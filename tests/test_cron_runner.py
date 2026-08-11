"""End-to-end behaviour of one cron refresh, without touching the network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import every_eval_ever.eval_types as ET
from every_eval_ever.cron import archive, publish, runner
from every_eval_ever.cron.fingerprint import output_fingerprint
from every_eval_ever.cron.schedule import CronAdapter, RawPolicy
from every_eval_ever.cron.stamp import (
    CRON_ADDITION_TYPE,
    TYPE_OF_ADDITION_KEY,
    StampError,
    stamp_tree,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    raw_capture,
    save_evaluation_log,
)

ADAPTER = 'vals_ai'


def _log(model_id: str = 'dev/model', score: float = 0.5):
    return ET.EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=f'vals/{model_id}/1700000000.0',
        retrieved_timestamp='1700000000.0',
        source_metadata=ET.SourceMetadata(
            source_name='Vals.ai',
            source_type='documentation',
            source_organization_name='Vals AI',
            evaluator_relationship='third_party',
        ),
        eval_library=ET.EvalLibrary(name='unknown', version='unknown'),
        model_info=ET.ModelInfo(
            name=model_id.split('/')[-1],
            id=model_id,
            developer='dev',
        ),
        evaluation_results=[
            ET.EvaluationResult(
                evaluation_name='finance_agent',
                source_data=ET.SourceDataUrl(
                    dataset_name='vals-ai',
                    source_type='url',
                    url=['https://vals.invalid/finance'],
                ),
                metric_config=ET.MetricConfig(
                    metric_name='accuracy',
                    lower_is_better=False,
                    score_type=ET.ScoreType.continuous,
                    min_score=0.0,
                    max_score=1.0,
                ),
                score_details=ET.ScoreDetails(score=score),
            )
        ],
    )


@pytest.fixture
def stub_adapter(monkeypatch):
    """Replace the subprocess call with an in-process writer."""

    def install(
        score: float = 0.5,
        raw: bytes | None = b'{"rows": 1}',
        verbatim: bool = False,
    ):
        def run_adapter(adapter, *, work_dir, raw_dir, environment):
            save_evaluation_log(
                _log(score=score),
                base_dir=work_dir / 'data' / 'vals-ai',
                developer='dev',
                model_name='model',
            )
            if raw is not None and verbatim:
                # What the shared fetch hook does: store the body as served.
                monkeypatch.setenv(
                    raw_capture.RAW_CAPTURE_DIR_ENV, str(raw_dir)
                )
                raw_capture.capture_response(
                    'https://vals.invalid/finance', raw
                )
            elif raw is not None:
                # What an adapter's own --save-raw-json flag leaves behind.
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / 'vals-ai.json').write_bytes(raw)
            return runner.AdapterOutcome(
                invocations=[
                    runner.Invocation(arguments=list(run), returncode=0)
                    for run in adapter.runs
                ]
            )

        monkeypatch.setattr(runner, 'run_adapter', run_adapter)

    return install


@pytest.fixture(autouse=True)
def _forget_recorded_payloads():
    raw_capture.reset_recorded_state()
    yield
    raw_capture.reset_recorded_state()


@pytest.fixture(autouse=True)
def _never_reach_the_hub(monkeypatch):
    """Archiving, the ledger read and publishing all talk to the Hub.

    None of them may reach it from a test, so all three are stubbed by default.
    Tests that care about one replace it with their own stub. Relying on each
    test to remember is how a test suite starts making live API calls.
    """
    monkeypatch.setattr(
        runner.archive_module,
        'archive',
        lambda raw_dir, **kwargs: archive.ArchiveResult(
            repo_id='evaleval/EEE_raw',
            ledger_path='ledger/stub.jsonl',
            uploaded=1,
        ),
    )
    monkeypatch.setattr(
        runner.archive_module,
        'last_gating_fingerprint',
        lambda adapter, **kwargs: None,
    )
    monkeypatch.setattr(
        runner.publish_module,
        'publish',
        lambda data_root, **kwargs: publish.PublishResult(
            pr_url='https://hf.invalid/discussions/0',
            pr_number=0,
            files=0,
            commits=0,
            reused_existing_pr=False,
        ),
    )


def _refresh(tmp_path: Path, **overrides):
    arguments = {
        'work_dir': tmp_path / 'work',
        'fingerprint_path': tmp_path / 'fingerprint' / ADAPTER,
        'summary_path': tmp_path / 'summary.json',
        'repo_id': 'evaleval/EEE_datastore',
        'run_url': 'https://github.invalid/run/1',
        'dry_run': True,
        'force': False,
        'environment': {},
    }
    arguments.update(overrides)
    return runner.refresh(ADAPTER, **arguments)


def test_a_dry_run_stamps_validates_and_reports_without_publishing(
    tmp_path: Path, stub_adapter
):
    stub_adapter()

    code, summary = _refresh(tmp_path)

    assert code == runner.EXIT_PUBLISHED
    assert summary.status == 'dry_run'
    assert summary.records == 1
    assert summary.collections == ['vals-ai']
    assert summary.validation == {'files': 1, 'errors': 0, 'warnings': 0}
    assert summary.pr_url is None

    record = next((tmp_path / 'work' / 'data').glob('*/*/*/*.json'))
    details = json.loads(record.read_text(encoding='utf-8'))['source_metadata'][
        'additional_details'
    ]
    assert details[TYPE_OF_ADDITION_KEY] == CRON_ADDITION_TYPE
    assert details['cron_run_date'] == summary.run_date
    assert details['cron_adapter'] == ADAPTER
    assert details['cron_run_url'] == 'https://github.invalid/run/1'


def test_the_summary_is_written_for_the_artifact(tmp_path: Path, stub_adapter):
    stub_adapter()
    _refresh(tmp_path)

    payload = json.loads((tmp_path / 'summary.json').read_text('utf-8'))

    assert payload['adapter'] == ADAPTER
    assert payload['status'] == 'dry_run'
    assert payload['raw_payloads'] == 1
    assert payload['fingerprint_source'] == 'output'


def test_a_dump_the_adapter_wrote_is_archived_but_does_not_gate_the_run(
    tmp_path: Path, stub_adapter
):
    # Archived for future reference, but it may carry its own fetch timestamp,
    # so the run is gated on its output instead.
    stub_adapter(raw=b'{"rows": 7}')

    _, summary = _refresh(tmp_path)

    assert summary.raw_payloads == 1
    assert summary.raw_bytes == len(b'{"rows": 7}')
    assert summary.raw_fingerprint is None
    assert summary.fingerprint_source == 'output'


def test_a_verbatim_wire_capture_gates_the_run(tmp_path: Path, stub_adapter):
    stub_adapter(raw=b'{"rows": 7}', verbatim=True)

    _, summary = _refresh(tmp_path)

    assert summary.raw_payloads == 1
    assert summary.raw_fingerprint
    assert summary.fingerprint_source == 'raw'


def test_output_fingerprint_is_used_when_no_raw_data_was_archived(
    tmp_path: Path, stub_adapter
):
    stub_adapter(raw=None)

    _, summary = _refresh(tmp_path)

    assert summary.raw_payloads == 0
    assert summary.raw_fingerprint is None
    assert summary.fingerprint_source == 'output'
    assert summary.output_fingerprint


def test_an_adapter_that_produces_no_records_is_not_a_failure(
    tmp_path: Path, monkeypatch
):
    def run_adapter(adapter, *, work_dir, raw_dir, environment):
        work_dir.mkdir(parents=True, exist_ok=True)
        return runner.AdapterOutcome(
            invocations=[runner.Invocation(arguments=[], returncode=0)]
        )

    monkeypatch.setattr(runner, 'run_adapter', run_adapter)

    code, summary = _refresh(tmp_path)

    assert code == runner.EXIT_NOTHING_NEW
    assert summary.status == 'nothing_produced'


def test_missing_credentials_skip_rather_than_fail(tmp_path: Path):
    code, summary = runner.refresh(
        'llm_stats',
        work_dir=tmp_path / 'work',
        fingerprint_path=None,
        summary_path=tmp_path / 'summary.json',
        repo_id='evaleval/EEE_datastore',
        run_url=None,
        dry_run=True,
        force=False,
        environment={},
    )

    assert code == runner.EXIT_NOTHING_NEW
    assert summary.status == 'skipped'
    assert 'LLM_STATS_API_KEY' in summary.detail


def test_a_failing_invocation_does_not_stop_the_others(
    tmp_path: Path, monkeypatch
):
    calls = []

    class Completed:
        def __init__(self, returncode):
            self.returncode = returncode

    def fake_run(argv, *, cwd, env, check):
        calls.append(argv[-1])
        # The second leaderboard is broken; the rest must still run.
        return Completed(1 if argv[-1] == 'HELM_Lite' else 0)

    monkeypatch.setattr(runner.subprocess, 'run', fake_run)
    adapter = CronAdapter(
        name='helm',
        raw_policy=RawPolicy.VIA_FETCH_HELPERS,
        runs=(
            ('--leaderboard_name', 'HELM_Capabilities'),
            ('--leaderboard_name', 'HELM_Lite'),
            ('--leaderboard_name', 'HELM_MMLU'),
        ),
    )

    outcome = runner.run_adapter(
        adapter,
        work_dir=tmp_path / 'work',
        raw_dir=tmp_path / 'raw',
        environment={},
    )

    assert calls == ['HELM_Capabilities', 'HELM_Lite', 'HELM_MMLU']
    assert len(outcome.invocations) == 3
    assert [item.arguments[-1] for item in outcome.failed] == ['HELM_Lite']
    assert not outcome.all_failed


def test_records_from_a_partial_refresh_are_still_published(
    tmp_path: Path, monkeypatch
):
    published = {}

    def run_adapter(adapter, *, work_dir, raw_dir, environment):
        save_evaluation_log(
            _log(),
            base_dir=work_dir / 'data' / 'vals-ai',
            developer='dev',
            model_name='model',
        )
        return runner.AdapterOutcome(
            invocations=[
                runner.Invocation(arguments=['--benchmark', 'a'], returncode=0),
                runner.Invocation(arguments=['--benchmark', 'b'], returncode=1),
            ]
        )

    def fake_publish(data_root, **kwargs):
        published.update(kwargs)
        published['files'] = len(list(Path(data_root).glob('*/*/*/*.json')))
        return publish.PublishResult(
            pr_url='https://hf.invalid/discussions/7',
            pr_number=7,
            files=published['files'],
            commits=1,
            reused_existing_pr=False,
        )

    monkeypatch.setattr(runner, 'run_adapter', run_adapter)
    monkeypatch.setattr(runner.publish_module, 'publish', fake_publish)

    code, summary = _refresh(tmp_path, dry_run=False)

    assert code == runner.EXIT_PUBLISHED
    assert summary.status == 'published_partial'
    assert published['files'] == 1
    assert summary.failed_invocations == [
        {'arguments': '--benchmark b', 'returncode': 1}
    ]
    # The reviewer of the PR is told the source is only partly represented.
    assert 'Partial refresh' in published['commit_description']


def test_no_records_and_a_failing_adapter_is_a_failure(
    tmp_path: Path, monkeypatch
):
    def run_adapter(adapter, *, work_dir, raw_dir, environment):
        work_dir.mkdir(parents=True, exist_ok=True)
        return runner.AdapterOutcome(
            invocations=[runner.Invocation(arguments=[], returncode=1)]
        )

    monkeypatch.setattr(runner, 'run_adapter', run_adapter)

    code, summary = _refresh(tmp_path)

    assert code == runner.EXIT_FAILED
    assert summary.status == 'failed'


def test_the_adapter_runs_in_the_scratch_tree_with_capture_enabled(
    tmp_path: Path, monkeypatch
):
    seen = {}

    class Completed:
        returncode = 0

    def fake_run(argv, *, cwd, env, check):
        seen['argv'] = argv
        seen['cwd'] = cwd
        seen['env'] = env
        return Completed()

    monkeypatch.setattr(runner.subprocess, 'run', fake_run)
    adapter = CronAdapter(
        name=ADAPTER,
        raw_policy=RawPolicy.VIA_ADAPTER_FLAG,
        raw_args=('--save-raw-json', '{raw_dir}/vals-ai.json'),
    )
    work_dir = tmp_path / 'work'
    raw_dir = tmp_path / 'raw'

    runner.run_adapter(
        adapter, work_dir=work_dir, raw_dir=raw_dir, environment={'A': 'b'}
    )

    assert seen['cwd'] == work_dir
    assert seen['argv'][1:3] == [
        '-m',
        'every_eval_ever.adapters.vals_ai.adapter',
    ]
    assert seen['argv'][-2:] == [
        '--save-raw-json',
        f'{raw_dir}/vals-ai.json',
    ]
    assert seen['env'][raw_capture.RAW_CAPTURE_DIR_ENV] == str(raw_dir)
    assert seen['env']['A'] == 'b'


def test_an_invalid_record_stops_the_refresh_and_names_the_file(
    tmp_path: Path, monkeypatch
):
    def run_adapter(adapter, *, work_dir, raw_dir, environment):
        # A file in the right place that is not a valid record.
        directory = work_dir / 'data' / 'vals-ai' / 'dev' / 'model'
        directory.mkdir(parents=True)
        (directory / f'{"a" * 8}-0000-4000-8000-000000000000.json').write_text(
            '{"schema_version": "0.3.0"}\n', encoding='utf-8'
        )
        return runner.AdapterOutcome(
            invocations=[runner.Invocation(arguments=[], returncode=0)]
        )

    monkeypatch.setattr(runner, 'run_adapter', run_adapter)

    with pytest.raises(StampError, match='not a valid EvaluationLog'):
        _refresh(tmp_path)


def test_output_fingerprint_ignores_the_regenerated_uuid_and_timestamp(
    tmp_path: Path,
):
    first = tmp_path / 'first' / 'data' / 'vals-ai'
    second = tmp_path / 'second' / 'data' / 'vals-ai'
    for base in (first, second):
        save_evaluation_log(
            _log(), base_dir=base, developer='dev', model_name='model'
        )

    assert output_fingerprint(first.parent) == output_fingerprint(second.parent)


def test_output_fingerprint_notices_a_changed_score(tmp_path: Path):
    first = tmp_path / 'first' / 'data' / 'vals-ai'
    second = tmp_path / 'second' / 'data' / 'vals-ai'
    for base, score in ((first, 0.5), (second, 0.6)):
        save_evaluation_log(
            _log(score=score),
            base_dir=base,
            developer='dev',
            model_name='model',
        )

    assert output_fingerprint(first.parent) != output_fingerprint(second.parent)


def test_a_second_run_with_the_same_source_publishes_nothing(
    tmp_path: Path, stub_adapter, monkeypatch
):
    published = []

    def fake_publish(data_root, **kwargs):
        published.append(kwargs['adapter'])
        return publish.PublishResult(
            pr_url='https://hf.invalid/discussions/7',
            pr_number=7,
            files=1,
            commits=1,
            reused_existing_pr=False,
        )

    monkeypatch.setattr(runner.publish_module, 'publish', fake_publish)
    fingerprint = tmp_path / 'fingerprint' / ADAPTER
    stub_adapter()

    first_code, first = _refresh(
        tmp_path,
        work_dir=tmp_path / 'first',
        fingerprint_path=fingerprint,
        dry_run=False,
    )
    second_code, second = _refresh(
        tmp_path,
        work_dir=tmp_path / 'second',
        fingerprint_path=fingerprint,
        dry_run=False,
    )

    assert first_code == runner.EXIT_PUBLISHED
    assert first.status == 'published'
    assert second_code == runner.EXIT_NOTHING_NEW
    assert second.status == 'unchanged'
    # Published once, not twice: the second day added no duplicate records.
    assert published == [ADAPTER]


def test_a_changed_source_publishes_again(
    tmp_path: Path, stub_adapter, monkeypatch
):
    published = []
    monkeypatch.setattr(
        runner.publish_module,
        'publish',
        lambda data_root, **kwargs: (
            published.append(kwargs['adapter'])
            or publish.PublishResult(
                pr_url='https://hf.invalid/discussions/7',
                pr_number=7,
                files=1,
                commits=1,
                reused_existing_pr=True,
            )
        ),
    )
    fingerprint = tmp_path / 'fingerprint' / ADAPTER

    stub_adapter(raw=b'{"rows": 1}', verbatim=True)
    _refresh(
        tmp_path,
        work_dir=tmp_path / 'first',
        fingerprint_path=fingerprint,
        dry_run=False,
    )
    stub_adapter(raw=b'{"rows": 2}', verbatim=True)
    code, summary = _refresh(
        tmp_path,
        work_dir=tmp_path / 'second',
        fingerprint_path=fingerprint,
        dry_run=False,
    )

    assert code == runner.EXIT_PUBLISHED
    assert summary.status == 'published'
    assert summary.previous_fingerprint != summary.raw_fingerprint
    assert published == [ADAPTER, ADAPTER]


def test_force_publishes_even_when_the_source_is_unchanged(
    tmp_path: Path, stub_adapter, monkeypatch
):
    published = []
    monkeypatch.setattr(
        runner.publish_module,
        'publish',
        lambda data_root, **kwargs: (
            published.append(kwargs['adapter'])
            or publish.PublishResult(
                pr_url='https://hf.invalid/discussions/7',
                pr_number=7,
                files=1,
                commits=1,
                reused_existing_pr=True,
            )
        ),
    )
    fingerprint = tmp_path / 'fingerprint' / ADAPTER
    stub_adapter()

    _refresh(
        tmp_path,
        work_dir=tmp_path / 'first',
        fingerprint_path=fingerprint,
        dry_run=False,
    )
    code, summary = _refresh(
        tmp_path,
        work_dir=tmp_path / 'second',
        fingerprint_path=fingerprint,
        dry_run=False,
        force=True,
    )

    assert code == runner.EXIT_PUBLISHED
    assert summary.status == 'published'
    assert published == [ADAPTER, ADAPTER]


def test_raw_data_is_archived_before_anything_is_published(
    tmp_path: Path, stub_adapter, monkeypatch
):
    order = []

    def fake_archive(raw_dir, **kwargs):
        order.append('archive')
        return archive.ArchiveResult(
            repo_id='evaleval/EEE_raw',
            ledger_path='ledger/vals_ai/2026-08-11-1-1.jsonl',
            uploaded=1,
            reused=0,
            uploaded_bytes=11,
        )

    def fake_publish(data_root, **kwargs):
        order.append('publish')
        return publish.PublishResult(
            pr_url='https://hf.invalid/discussions/7',
            pr_number=7,
            files=1,
            commits=1,
            reused_existing_pr=False,
        )

    monkeypatch.setattr(runner.archive_module, 'archive', fake_archive)
    monkeypatch.setattr(runner.publish_module, 'publish', fake_publish)
    stub_adapter()

    _, summary = _refresh(tmp_path, dry_run=False)

    # Records must not reach the datastore without their raw data stored.
    assert order == ['archive', 'publish']
    assert summary.raw_archive['status'] == 'archived'
    assert summary.raw_archive['ledger_path'].startswith('ledger/vals_ai/')


def test_a_failed_archive_stops_the_refresh_before_publishing(
    tmp_path: Path, stub_adapter, monkeypatch
):
    published = []

    def fake_archive(raw_dir, **kwargs):
        raise archive.ArchiveError('403 forbidden')

    monkeypatch.setattr(runner.archive_module, 'archive', fake_archive)
    monkeypatch.setattr(
        runner.publish_module,
        'publish',
        lambda *a, **k: published.append(1),
    )
    stub_adapter()

    with pytest.raises(archive.ArchiveError):
        _refresh(tmp_path, dry_run=False)

    assert published == []


def test_raw_data_is_archived_even_when_the_source_has_not_changed(
    tmp_path: Path, stub_adapter, monkeypatch
):
    # The ledger's value is knowing what the source looked like on each date,
    # and an unchanged payload costs nothing to keep.
    calls = []
    monkeypatch.setattr(
        runner.archive_module,
        'archive',
        lambda raw_dir, **kwargs: (
            calls.append(kwargs['run_date'])
            or archive.ArchiveResult(repo_id='r', ledger_path='l', reused=1)
        ),
    )
    monkeypatch.setattr(
        runner.publish_module,
        'publish',
        lambda *a, **k: publish.PublishResult(
            pr_url='u',
            pr_number=1,
            files=1,
            commits=1,
            reused_existing_pr=False,
        ),
    )
    fingerprint = tmp_path / 'fingerprint' / ADAPTER
    stub_adapter(verbatim=True)

    _refresh(
        tmp_path,
        work_dir=tmp_path / 'first',
        fingerprint_path=fingerprint,
        dry_run=False,
    )
    _, second = _refresh(
        tmp_path,
        work_dir=tmp_path / 'second',
        fingerprint_path=fingerprint,
        dry_run=False,
    )

    assert second.status == 'unchanged'
    assert len(calls) == 2


def test_raw_data_is_archived_even_when_no_records_were_produced(
    tmp_path: Path, monkeypatch
):
    calls = []

    def run_adapter(adapter, *, work_dir, raw_dir, environment):
        # What a crashing adapter leaves: the payload, and no records.
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / 'vals-ai.json').write_bytes(b'{"rows": 1}')
        return runner.AdapterOutcome(
            invocations=[runner.Invocation(arguments=[], returncode=1)]
        )

    monkeypatch.setattr(runner, 'run_adapter', run_adapter)
    monkeypatch.setattr(
        runner.archive_module,
        'archive',
        lambda raw_dir, **kwargs: (
            calls.append(1)
            or archive.ArchiveResult(repo_id='r', ledger_path='l', uploaded=1)
        ),
    )

    code, summary = _refresh(tmp_path, dry_run=False)

    assert code == runner.EXIT_FAILED
    assert calls == [1]
    assert summary.raw_archive['status'] == 'archived'


def test_a_dry_run_reports_the_archive_it_would_write(
    tmp_path: Path, stub_adapter
):
    stub_adapter()

    _, summary = _refresh(tmp_path, dry_run=True)

    assert summary.raw_archive == {
        'status': 'skipped_dry_run',
        'repo_id': archive.DEFAULT_RAW_REPO_ID,
        'payloads': 1,
        'skipped_payloads': 0,
    }


def test_an_adapter_with_no_raw_data_records_that_fact(
    tmp_path: Path, stub_adapter
):
    stub_adapter(raw=None)

    _, summary = _refresh(tmp_path)

    assert summary.raw_archive['status'] == 'nothing_captured'


def test_the_output_fingerprint_is_the_same_before_and_after_stamping(
    tmp_path: Path,
):
    # The runner computes it before stamping and archives it; the value has to
    # be the one a later run will compare against.
    base = tmp_path / 'data' / 'vals-ai'
    save_evaluation_log(
        _log(), base_dir=base, developer='dev', model_name='model'
    )
    before = output_fingerprint(tmp_path / 'data')

    stamp_tree(tmp_path / 'data', adapter=ADAPTER, run_date='2026-08-11')

    assert output_fingerprint(tmp_path / 'data') == before


def test_the_previous_fingerprint_comes_from_the_ledger(
    tmp_path: Path, stub_adapter, monkeypatch
):
    # No local fingerprint file and no build cache: the raw dataset's ledger is
    # what stops a rerun from republishing everything.
    stub_adapter(verbatim=True)
    _, first = _refresh(
        tmp_path, work_dir=tmp_path / 'first', fingerprint_path=None
    )

    monkeypatch.setattr(
        runner.archive_module,
        'last_gating_fingerprint',
        lambda adapter, **kwargs: first.raw_fingerprint,
    )
    monkeypatch.setattr(
        runner.publish_module,
        'publish',
        lambda *a, **k: publish.PublishResult(
            pr_url='u', pr_number=1, files=1, commits=1, reused_existing_pr=True
        ),
    )
    stub_adapter(verbatim=True)

    code, second = _refresh(
        tmp_path,
        work_dir=tmp_path / 'second',
        fingerprint_path=None,
        dry_run=False,
    )

    assert code == runner.EXIT_NOTHING_NEW
    assert second.status == 'unchanged'
    assert second.previous_fingerprint == first.raw_fingerprint


def test_a_dry_run_does_not_consult_the_ledger(
    tmp_path: Path, stub_adapter, monkeypatch
):
    consulted = []
    monkeypatch.setattr(
        runner.archive_module,
        'last_gating_fingerprint',
        lambda adapter, **kwargs: consulted.append(adapter),
    )
    stub_adapter()

    _refresh(tmp_path, fingerprint_path=None, dry_run=True)

    assert consulted == []


def test_an_oversized_payload_is_still_recorded_in_the_ledger(
    tmp_path: Path, monkeypatch
):
    # Nothing lands on disk, but the fetch happened and the ledger has to say so.
    # Guarding the archive on "did a payload land" skipped the row entirely.
    archived = []

    def run_adapter(adapter, *, work_dir, raw_dir, environment):
        monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(raw_dir))
        monkeypatch.setenv(raw_capture.RAW_CAPTURE_MAX_BYTES_ENV, '4')
        raw_capture.capture_response('https://vals.invalid/big', b'123456')
        save_evaluation_log(
            _log(),
            base_dir=work_dir / 'data' / 'vals-ai',
            developer='dev',
            model_name='model',
        )
        return runner.AdapterOutcome(
            invocations=[runner.Invocation(arguments=[], returncode=0)]
        )

    monkeypatch.setattr(runner, 'run_adapter', run_adapter)
    monkeypatch.setattr(
        runner.archive_module,
        'archive',
        lambda raw_dir, **kwargs: (
            archived.append(raw_dir)
            or archive.ArchiveResult(
                repo_id='evaleval/EEE_raw',
                ledger_path='ledger/vals_ai/x.jsonl',
                rows=archive.ledger_rows(
                    raw_dir,
                    adapter='vals_ai',
                    run_date='2026-08-11',
                    run_id='1',
                ),
            )
        ),
    )
    monkeypatch.setattr(
        runner.publish_module,
        'publish',
        lambda *a, **k: publish.PublishResult(
            pr_url='u',
            pr_number=1,
            files=1,
            commits=1,
            reused_existing_pr=False,
        ),
    )

    _, summary = _refresh(tmp_path, dry_run=False)

    assert summary.raw_payloads == 0
    assert summary.raw_skipped == 1
    assert len(archived) == 1, 'the ledger row was never archived'
    assert summary.raw_archive['status'] == 'archived'
    assert summary.raw_archive['ledger_rows'] == 1


def test_a_run_that_captured_nothing_reports_nothing_captured(
    tmp_path: Path, stub_adapter, monkeypatch
):
    archived = []
    monkeypatch.setattr(
        runner.archive_module,
        'archive',
        lambda raw_dir, **kwargs: archived.append(raw_dir),
    )
    stub_adapter(raw=None)

    _, summary = _refresh(tmp_path, dry_run=False)

    assert summary.raw_payloads == 0
    assert summary.raw_skipped == 0
    assert summary.raw_archive['status'] == 'nothing_captured'
    assert archived == []


def test_raw_bytes_counts_only_what_was_stored(tmp_path: Path, stub_adapter):
    stub_adapter(raw=b'{"rows": 7}', verbatim=True)

    _, summary = _refresh(tmp_path)

    assert summary.raw_payloads == 1
    assert summary.raw_bytes == len(b'{"rows": 7}')
    assert summary.raw_skipped == 0
