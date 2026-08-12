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
# An adapter whose registry entry declares NO capture route, for scenarios
# where producing records without raw payloads must be legitimate.
ZERO_CAPTURE_ADAPTER = 'mt_bench'


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
def state_store(monkeypatch):
    """Archiving, the state read/write and publishing all talk to the Hub.

    None of them may reach it from a test, so all are stubbed by default —
    relying on each test to remember is how a test suite starts making live
    API calls. The state stubs share a dict per test, so ``refresh()``'s real
    coupling (gate reads the state a previous *successful publish* wrote) is
    exercised rather than stubbed away; that stubbing is exactly what hid the
    run-reads-its-own-fingerprint bug.
    """
    store: dict[str, dict] = {}
    attempts: dict[str, dict] = {}
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
        'read_state',
        lambda adapter, **kwargs: store.get(adapter),
    )

    def write_state(adapter, state, **kwargs):
        store[adapter] = state
        if kwargs.get('clear_attempt'):
            attempts.pop(adapter, None)

    monkeypatch.setattr(runner.archive_module, 'write_state', write_state)
    monkeypatch.setattr(
        runner.archive_module,
        'read_attempt',
        lambda adapter, **kwargs: attempts.get(adapter),
    )
    monkeypatch.setattr(
        runner.archive_module,
        'write_attempt',
        lambda adapter, attempt, **kwargs: attempts.__setitem__(
            adapter, attempt
        ),
    )
    store['__attempts__'] = attempts  # visible to tests that need them
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
    return store


def _refresh(tmp_path: Path, **overrides):
    arguments = {
        'work_dir': tmp_path / 'work',
        'fingerprint_path': None,
        'summary_path': tmp_path / 'summary.json',
        'repo_id': 'evaleval/EEE_datastore',
        'run_url': 'https://github.invalid/run/1',
        'dry_run': True,
        'force': False,
        'environment': {},
    }
    adapter = overrides.pop('adapter', ADAPTER)
    arguments.update(overrides)
    return runner.refresh(adapter, **arguments)


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

    _, summary = _refresh(tmp_path, adapter=ZERO_CAPTURE_ADAPTER)

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


def test_missing_credentials_fail_the_adapters_own_run(tmp_path: Path):
    # An enabled adapter without its credential is broken configuration, not a
    # quiet day: its job fails in isolation while the others proceed.
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

    assert code == runner.EXIT_FAILED
    assert summary.status == 'missing_credentials'
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
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / 'vals-ai.json').write_bytes(b'{"rows": 1}')
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
            json.dumps({'schema_version': SCHEMA_VERSION}) + '\n',
            encoding='utf-8',
        )
        return runner.AdapterOutcome(
            invocations=[runner.Invocation(arguments=[], returncode=0)]
        )

    monkeypatch.setattr(runner, 'run_adapter', run_adapter)

    with pytest.raises(StampError, match='not a valid EvaluationLog'):
        _refresh(tmp_path, adapter=ZERO_CAPTURE_ADAPTER)


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

    _, summary = _refresh(tmp_path, adapter=ZERO_CAPTURE_ADAPTER)

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


def test_a_run_never_compares_against_its_own_fingerprint(
    tmp_path: Path, stub_adapter, state_store
):
    # The regression that shipped: the gate read the ledger this same run had
    # just archived and concluded "unchanged" — every adapter published
    # nothing, forever. The gate must only ever see what a previous successful
    # publish recorded, so a first run over an empty store publishes.
    stub_adapter(verbatim=True)

    code, summary = _refresh(tmp_path, dry_run=False)

    assert code == runner.EXIT_PUBLISHED
    assert summary.status == 'published'
    assert summary.previous_fingerprint is None
    # ...and only now, after the publish, is the fingerprint remembered.
    assert state_store[ADAPTER]['gating_fingerprint'] == (
        summary.raw_fingerprint
    )


def test_the_previous_fingerprint_comes_from_the_publish_state(
    tmp_path: Path, stub_adapter, state_store
):
    # No local fingerprint file and no build cache: the state a successful
    # publish wrote is what stops a rerun from republishing everything.
    stub_adapter(verbatim=True)
    _refresh(tmp_path, work_dir=tmp_path / 'first', dry_run=False)

    stub_adapter(verbatim=True)
    code, second = _refresh(
        tmp_path, work_dir=tmp_path / 'second', dry_run=False
    )

    assert code == runner.EXIT_NOTHING_NEW
    assert second.status == 'unchanged'
    assert (
        second.previous_fingerprint
        == (state_store[ADAPTER]['gating_fingerprint'])
    )


def test_a_failed_publish_does_not_update_the_state(
    tmp_path: Path, stub_adapter, state_store, monkeypatch
):
    # Recording the fingerprint before the publish succeeded would make the
    # next run skip as "unchanged" and the records would never be published.
    def failing_publish(data_root, **kwargs):
        raise publish.PublishError('504 gateway timeout')

    monkeypatch.setattr(runner.publish_module, 'publish', failing_publish)
    stub_adapter(verbatim=True)

    with pytest.raises(publish.PublishError):
        _refresh(tmp_path, work_dir=tmp_path / 'first', dry_run=False)

    assert ADAPTER not in state_store


def test_the_run_after_a_failed_publish_publishes(
    tmp_path: Path, stub_adapter, state_store, monkeypatch
):
    attempts = []

    def flaky_publish(data_root, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise publish.PublishError('504 gateway timeout')
        return publish.PublishResult(
            pr_url='u', pr_number=1, files=1, commits=1, reused_existing_pr=True
        )

    monkeypatch.setattr(runner.publish_module, 'publish', flaky_publish)
    stub_adapter(verbatim=True)

    with pytest.raises(publish.PublishError):
        _refresh(tmp_path, work_dir=tmp_path / 'first', dry_run=False)
    stub_adapter(verbatim=True)
    code, summary = _refresh(
        tmp_path, work_dir=tmp_path / 'second', dry_run=False
    )

    assert code == runner.EXIT_PUBLISHED
    assert summary.status == 'published'
    assert len(attempts) == 2


def test_a_zero_capture_adapter_still_skips_when_unchanged(
    tmp_path: Path, stub_adapter, state_store
):
    # hal, lexam, mt_bench and the upstream-versioned adapters archive no raw
    # payloads. Their output fingerprint must still be remembered, or they
    # would republish their whole set every single day.
    stub_adapter(raw=None)
    first_code, first = _refresh(
        tmp_path,
        work_dir=tmp_path / 'first',
        dry_run=False,
        adapter=ZERO_CAPTURE_ADAPTER,
    )
    stub_adapter(raw=None)
    second_code, second = _refresh(
        tmp_path,
        work_dir=tmp_path / 'second',
        dry_run=False,
        adapter=ZERO_CAPTURE_ADAPTER,
    )

    assert first_code == runner.EXIT_PUBLISHED
    assert first.fingerprint_source == 'output'
    assert second_code == runner.EXIT_NOTHING_NEW
    assert second.status == 'unchanged'


def _partial_run_adapter(*, failures, raw=b'{"rows": 1}'):
    """A run_adapter producing one record, a raw payload, and given failures."""

    def run_adapter(adapter, *, work_dir, raw_dir, environment):
        save_evaluation_log(
            _log(),
            base_dir=work_dir / 'data' / 'vals-ai',
            developer='dev',
            model_name='model',
        )
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / 'vals-ai.json').write_bytes(raw)
        return runner.AdapterOutcome(
            invocations=[
                runner.Invocation(arguments=['--benchmark', 'a'], returncode=0),
                *(
                    runner.Invocation(
                        arguments=['--benchmark', name], returncode=code
                    )
                    for name, code in failures
                ),
            ]
        )

    return run_adapter


def test_an_identical_partial_run_is_not_republished(
    tmp_path: Path, state_store, monkeypatch
):
    # Same records converted, same invocations failed the same way:
    # republishing would duplicate the successes without recovering anything.
    monkeypatch.setattr(
        runner, 'run_adapter', _partial_run_adapter(failures=[('b', 1)])
    )
    first_code, first = _refresh(
        tmp_path, work_dir=tmp_path / 'first', dry_run=False
    )
    second_code, second = _refresh(
        tmp_path, work_dir=tmp_path / 'second', dry_run=False
    )

    assert first_code == runner.EXIT_PUBLISHED
    assert first.status == 'published_partial'
    assert state_store[ADAPTER]['partial'] is True
    assert second_code == runner.EXIT_NOTHING_NEW
    assert second.status == 'unchanged'


def test_a_changed_failure_set_publishes_again(
    tmp_path: Path, state_store, monkeypatch
):
    monkeypatch.setattr(
        runner, 'run_adapter', _partial_run_adapter(failures=[('b', 1)])
    )
    _refresh(tmp_path, work_dir=tmp_path / 'first', dry_run=False)

    monkeypatch.setattr(
        runner,
        'run_adapter',
        _partial_run_adapter(failures=[('b', 1), ('c', 2)]),
    )
    code, summary = _refresh(
        tmp_path, work_dir=tmp_path / 'second', dry_run=False
    )

    assert code == runner.EXIT_PUBLISHED
    assert summary.status == 'published_partial'


def test_a_recovered_partial_run_publishes(
    tmp_path: Path, stub_adapter, state_store, monkeypatch
):
    # The failed invocation now converts: its records exist only in this run,
    # so the fingerprint differs and the run publishes.
    monkeypatch.setattr(
        runner, 'run_adapter', _partial_run_adapter(failures=[('b', 1)])
    )
    _refresh(tmp_path, work_dir=tmp_path / 'first', dry_run=False)

    def recovered(adapter, *, work_dir, raw_dir, environment):
        for model in ('model', 'second'):
            save_evaluation_log(
                _log(model_id=f'dev/{model}'),
                base_dir=work_dir / 'data' / 'vals-ai',
                developer='dev',
                model_name=model,
            )
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / 'vals-ai.json').write_bytes(b'{"rows": 2}')
        return runner.AdapterOutcome(
            invocations=[
                runner.Invocation(arguments=['--benchmark', 'a'], returncode=0),
                runner.Invocation(arguments=['--benchmark', 'b'], returncode=0),
            ]
        )

    monkeypatch.setattr(runner, 'run_adapter', recovered)
    code, summary = _refresh(
        tmp_path, work_dir=tmp_path / 'second', dry_run=False
    )

    assert code == runner.EXIT_PUBLISHED
    assert summary.status == 'published'
    assert state_store[ADAPTER]['partial'] is False


def test_a_dry_run_does_not_consult_the_state(
    tmp_path: Path, stub_adapter, state_store, monkeypatch
):
    consulted = []
    monkeypatch.setattr(
        runner.archive_module,
        'read_state',
        lambda adapter, **kwargs: consulted.append(adapter),
    )
    stub_adapter()

    _refresh(tmp_path, dry_run=True)

    assert consulted == []
    assert not {k: v for k, v in state_store.items() if k != '__attempts__'}


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

    _, summary = _refresh(tmp_path, dry_run=False, adapter=ZERO_CAPTURE_ADAPTER)

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


def test_a_capture_failure_stops_publication(
    tmp_path: Path, monkeypatch, state_store
):
    # Records without their source bytes archived must not be published.
    def run_adapter(adapter, *, work_dir, raw_dir, environment):
        save_evaluation_log(
            _log(),
            base_dir=work_dir / 'data' / 'vals-ai',
            developer='dev',
            model_name='model',
        )
        monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(raw_dir))
        raw_capture.capture_response('https://vals.invalid/ok', b'{"a": 1}')

        def broken(directory, url, body, content_type):
            raise OSError('disk full')

        monkeypatch.setattr(raw_capture, '_capture', broken)
        raw_capture.capture_response('https://vals.invalid/broken', b'{}')
        return runner.AdapterOutcome(
            invocations=[runner.Invocation(arguments=[], returncode=0)]
        )

    monkeypatch.setattr(runner, 'run_adapter', run_adapter)

    code, summary = _refresh(tmp_path, dry_run=False)

    # One capture succeeded — the mixed case must still fail.
    assert code == runner.EXIT_FAILED
    assert summary.status == 'failed'
    assert 'capture failure' in summary.detail
    assert summary.capture_errors == ['https://vals.invalid/broken']
    assert ADAPTER not in state_store


def test_records_without_any_captured_payload_stop_publication(
    tmp_path: Path, monkeypatch, state_store
):
    # vals_ai declares a capture route; records with an empty manifest mean
    # the route silently did not run — an unexplained provenance gap.
    def run_adapter(adapter, *, work_dir, raw_dir, environment):
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

    code, summary = _refresh(tmp_path, dry_run=False)

    assert code == runner.EXIT_FAILED
    assert 'captured nothing' in summary.detail
    assert ADAPTER not in state_store


def test_an_oversized_only_capture_still_publishes(
    tmp_path: Path, monkeypatch, state_store
):
    # A deliberate ceiling skip is recorded, not an error: the ledger row says
    # what happened and the run proceeds.
    def run_adapter(adapter, *, work_dir, raw_dir, environment):
        save_evaluation_log(
            _log(),
            base_dir=work_dir / 'data' / 'vals-ai',
            developer='dev',
            model_name='model',
        )
        monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(raw_dir))
        monkeypatch.setenv(raw_capture.RAW_CAPTURE_MAX_BYTES_ENV, '4')
        raw_capture.capture_response('https://vals.invalid/big', b'123456')
        return runner.AdapterOutcome(
            invocations=[runner.Invocation(arguments=[], returncode=0)]
        )

    monkeypatch.setattr(runner, 'run_adapter', run_adapter)

    code, summary = _refresh(tmp_path, dry_run=False)

    assert code == runner.EXIT_PUBLISHED
    assert summary.raw_skipped == 1
    assert summary.capture_errors == []


def test_the_write_token_never_reaches_the_adapter_subprocess(
    tmp_path: Path, monkeypatch
):
    # The cron's HF token can write to the datastore and the private raw
    # dataset; adapter code and its dependencies must never hold it.
    seen = {}

    class Completed:
        returncode = 0

    def fake_run(argv, *, cwd, env, check):
        seen.update(env)
        return Completed()

    monkeypatch.setattr(runner.subprocess, 'run', fake_run)
    adapter = CronAdapter(name=ADAPTER, raw_policy=RawPolicy.NOT_CAPTURED)

    runner.run_adapter(
        adapter,
        work_dir=tmp_path / 'work',
        raw_dir=tmp_path / 'raw',
        environment={
            'HF_TOKEN': 'write-capable',
            'HUGGING_FACE_HUB_TOKEN': 'write-capable',
            'HF_HUB_TOKEN': 'write-capable',
            'LLM_STATS_API_KEY': 'source-key',
            'PATH': '/usr/bin',
        },
    )

    for name in runner.WRITE_TOKEN_ENV_NAMES:
        assert name not in seen
    # Source-specific credentials and ordinary variables still pass through.
    assert seen['LLM_STATS_API_KEY'] == 'source-key'
    assert seen['PATH'] == '/usr/bin'


def test_a_source_hf_token_is_forwarded_only_when_declared(
    tmp_path: Path, monkeypatch
):
    seen = {}

    class Completed:
        returncode = 0

    def fake_run(argv, *, cwd, env, check):
        seen.update(env)
        return Completed()

    monkeypatch.setattr(runner.subprocess, 'run', fake_run)
    environment = {
        'HF_TOKEN': 'write-capable',
        runner.SOURCE_HF_TOKEN_ENV: 'read-only-source-token',
    }

    plain = CronAdapter(name=ADAPTER, raw_policy=RawPolicy.NOT_CAPTURED)
    runner.run_adapter(
        plain,
        work_dir=tmp_path / 'a',
        raw_dir=tmp_path / 'a-raw',
        environment=environment,
    )
    assert 'HF_TOKEN' not in seen

    seen.clear()
    declared = CronAdapter(
        name=ADAPTER,
        raw_policy=RawPolicy.NOT_CAPTURED,
        source_hf_token=True,
    )
    runner.run_adapter(
        declared,
        work_dir=tmp_path / 'b',
        raw_dir=tmp_path / 'b-raw',
        environment=environment,
    )
    # The forwarded value is the separate read token, never the cron's own.
    assert seen['HF_TOKEN'] == 'read-only-source-token'


def test_a_failed_publish_leaves_the_attempt_for_reconciliation(
    tmp_path: Path, stub_adapter, state_store, monkeypatch
):
    attempts = state_store['__attempts__']
    stale_seen = []

    calls = []

    def flaky_publish(data_root, *, stale_paths=None, **kwargs):
        calls.append(1)
        stale_seen.append(stale_paths)
        if len(calls) == 1:
            raise publish.PublishError('batch 2 of 3 failed')
        return publish.PublishResult(
            pr_url='u', pr_number=1, files=1, commits=1, reused_existing_pr=True
        )

    monkeypatch.setattr(runner.publish_module, 'publish', flaky_publish)
    stub_adapter(verbatim=True)

    with pytest.raises(publish.PublishError):
        _refresh(tmp_path, work_dir=tmp_path / 'first', dry_run=False)

    # The attempt record survives the failure — it is what the next run uses.
    assert ADAPTER in attempts
    left_behind = attempts[ADAPTER]['paths']
    assert left_behind and all(p.startswith('data/') for p in left_behind)

    stub_adapter(verbatim=True)
    code, summary = _refresh(
        tmp_path, work_dir=tmp_path / 'second', dry_run=False
    )

    assert code == runner.EXIT_PUBLISHED
    # The retry handed the incomplete attempt's paths to publish() for removal,
    # and success cleared the attempt in the same commit as the state.
    assert stale_seen[1] == left_behind
    assert ADAPTER not in attempts


def test_a_dangling_attempt_forces_a_publish_even_when_unchanged(
    tmp_path: Path, stub_adapter, state_store
):
    # Fingerprint equality must not skip while half an old attempt sits on the
    # PR: the reconciliation is the point of the run.
    attempts = state_store['__attempts__']
    stub_adapter(verbatim=True)
    _refresh(tmp_path, work_dir=tmp_path / 'first', dry_run=False)
    attempts[ADAPTER] = {'run_id': 'ghost', 'paths': ['data/x/y/z/a.json']}

    stub_adapter(verbatim=True)
    code, summary = _refresh(
        tmp_path, work_dir=tmp_path / 'second', dry_run=False
    )

    assert code == runner.EXIT_PUBLISHED
    assert summary.status == 'published'
    assert ADAPTER not in attempts


def test_a_persistent_state_write_failure_fails_the_run_keeping_the_pr(
    tmp_path: Path, stub_adapter, state_store, monkeypatch
):
    def broken_write(adapter, state, **kwargs):
        raise archive.ArchiveError('503 service unavailable')

    monkeypatch.setattr(runner.archive_module, 'write_state', broken_write)
    stub_adapter(verbatim=True)

    code, summary = _refresh(tmp_path, dry_run=False)

    assert code == runner.EXIT_FAILED
    assert summary.status == 'published_state_unrecorded'
    # The publish DID happen; the URL must stay in view.
    assert summary.pr_url == 'https://hf.invalid/discussions/0'
    assert '503' in summary.detail


def test_a_transient_state_write_failure_is_retried(
    tmp_path: Path, stub_adapter, state_store, monkeypatch
):
    attempts = []
    real_write = runner.archive_module.write_state

    def flaky_write(adapter, state, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise archive.ArchiveError('503 service unavailable')
        return real_write(adapter, state, **kwargs)

    monkeypatch.setattr(runner.archive_module, 'write_state', flaky_write)
    stub_adapter(verbatim=True)

    code, summary = _refresh(tmp_path, dry_run=False)

    assert code == runner.EXIT_PUBLISHED
    assert summary.status == 'published'
    assert len(attempts) == 2


def test_capture_failure_evidence_is_archived_before_the_run_fails(
    tmp_path: Path, monkeypatch, state_store
):
    # The successful sibling capture and the error row must be permanent
    # before the failure returns; the work directory is ephemeral.
    archived = []

    def recording_archive(raw_dir, **kwargs):
        archived.append(
            {
                'rows': archive.ledger_rows(
                    raw_dir, adapter='vals_ai', run_date='d', run_id='r'
                ),
                'reports': kwargs.get('reports'),
            }
        )
        return archive.ArchiveResult(repo_id='r', ledger_path='l', uploaded=1)

    monkeypatch.setattr(runner.archive_module, 'archive', recording_archive)

    def run_adapter(adapter, *, work_dir, raw_dir, environment):
        save_evaluation_log(
            _log(),
            base_dir=work_dir / 'data' / 'vals-ai',
            developer='dev',
            model_name='model',
        )
        reports = work_dir / 'adapter_reports'
        reports.mkdir(parents=True)
        (reports / 'vals-ai_failures.json').write_text('{}', encoding='utf-8')
        monkeypatch.setenv(raw_capture.RAW_CAPTURE_DIR_ENV, str(raw_dir))
        raw_capture.capture_response('https://vals.invalid/ok', b'{"a": 1}')

        def broken(directory, url, body, content_type):
            raise OSError('disk full')

        monkeypatch.setattr(raw_capture, '_capture', broken)
        raw_capture.capture_response('https://vals.invalid/broken', b'{}')
        return runner.AdapterOutcome(
            invocations=[runner.Invocation(arguments=[], returncode=0)]
        )

    monkeypatch.setattr(runner, 'run_adapter', run_adapter)

    code, summary = _refresh(tmp_path, dry_run=False)

    assert code == runner.EXIT_FAILED
    assert len(archived) == 1
    rows = archived[0]['rows']
    assert any(row['error'] for row in rows), 'the error row must be archived'
    assert any(row['file_name'] for row in rows), 'the sibling capture too'
    assert [r.name for r in archived[0]['reports']] == ['vals-ai_failures.json']
    assert ADAPTER not in state_store


def test_failure_reports_are_archived_even_with_no_captures(
    tmp_path: Path, monkeypatch, state_store
):
    # NOT_CAPTURED adapters still produce failure reports, which embed raw
    # source rows and must land in the private dataset, not a public artifact.
    archived = []
    monkeypatch.setattr(
        runner.archive_module,
        'archive',
        lambda raw_dir, **kwargs: (
            archived.append(kwargs.get('reports'))
            or archive.ArchiveResult(repo_id='r', ledger_path='l')
        ),
    )

    def run_adapter(adapter, *, work_dir, raw_dir, environment):
        save_evaluation_log(
            _log(),
            base_dir=work_dir / 'data' / 'vals-ai',
            developer='dev',
            model_name='model',
        )
        reports = work_dir / 'adapter_reports'
        reports.mkdir(parents=True)
        (reports / 'report.json').write_text('{}', encoding='utf-8')
        return runner.AdapterOutcome(
            invocations=[runner.Invocation(arguments=[], returncode=0)]
        )

    monkeypatch.setattr(runner, 'run_adapter', run_adapter)

    code, summary = _refresh(
        tmp_path, dry_run=False, adapter=ZERO_CAPTURE_ADAPTER
    )

    assert code == runner.EXIT_PUBLISHED
    assert [r.name for r in archived[0]] == ['report.json']
    assert summary.raw_archive['reports'] == 1
