"""What a scheduled run is and is not allowed to publish.

These drive the real pipeline — a real adapter subprocess, the real
validator, the real duplicate fingerprint — against a stand-in adapter whose
behaviour each test chooses. Nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from datetime import date
from pathlib import Path

import pytest

from every_eval_ever.adapters.registry import AdapterSpec
from every_eval_ever.cron import runner
from every_eval_ever.cron.provenance import is_cron_record
from every_eval_ever.validator.check_duplicate_entries import normalized_hash

FIXTURE_DIR = (
    Path(__file__).resolve().parent
    / 'data'
    / 'skill_reference_conversion'
    / 'data'
    / 'demo-source'
    / 'demo-org'
    / 'demo-model'
)
UUID_A = 'f3a1c0de-4b2e-4c1a-9f6d-1b7e5a2c8d40'
UUID_B = 'aa11bb22-cc33-4d44-8e55-ff6677889900'
RUN_DATE = date(2026, 8, 10)
COLLECTION = 'demo-source'

# A stand-in adapter. It writes exactly the files a test hands it, then
# reports whatever exit code the test asked for, which is how the tests
# exercise crash, partial-conversion, and misplaced-output paths.
STAND_IN = """
import argparse
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
FILES = json.loads((HERE / 'files.json').read_text('utf-8'))
BEHAVIOUR = json.loads((HERE / 'behaviour.json').read_text('utf-8'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    staging = args.output_dir.parent.parent
    for name, encoded in FILES.items():
        target = args.output_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(encoded))
    report = BEHAVIOUR.get('failure_report')
    if report:
        path = staging / 'adapter_reports' / 'demo-source_failures.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding='utf-8')
    stray = BEHAVIOUR.get('stray_file')
    if stray:
        path = staging / stray
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('stray', encoding='utf-8')
    print(BEHAVIOUR.get('stdout', 'stand-in adapter finished'))
    return BEHAVIOUR.get('exit_code', 0)


if __name__ == '__main__':
    sys.exit(main())
"""


def reference_record() -> dict:
    return json.loads((FIXTURE_DIR / f'{UUID_A}.json').read_text('utf-8'))


def record_without_samples(model: str = 'demo-model') -> dict:
    """A valid aggregate with no sidecar, retargeted onto ``model``."""
    record = reference_record()
    record.pop('detailed_evaluation_results', None)
    record['model_info']['name'] = model
    record['model_info']['id'] = f'demo-org/{model}'
    record['evaluation_id'] = f'demo-source/demo-org_{model}/1750000000'
    return record


def record_with_samples(uuid: str = UUID_A) -> tuple[dict, bytes]:
    """An aggregate and a sidecar whose checksum and row count agree.

    The bytes are rebuilt with LF endings rather than read from the fixture,
    so the pair stays self-consistent on a checkout that translates line
    endings.
    """
    record = reference_record()
    rows = [
        line
        for line in (FIXTURE_DIR / f'{UUID_A}_samples.jsonl')
        .read_text('utf-8')
        .splitlines()
        if line.strip()
    ]
    payload = ('\n'.join(rows) + '\n').encode('utf-8')
    detail = record['detailed_evaluation_results']
    detail['file_path'] = (
        f'data/{COLLECTION}/demo-org/demo-model/{uuid}_samples.jsonl'
    )
    detail['total_rows'] = len(rows)
    detail['checksum'] = hashlib.sha256(payload).hexdigest()
    return record, payload


def encode(value: dict | bytes | str) -> str:
    import base64

    if isinstance(value, dict):
        raw = (
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
            + '\n'
        ).encode('utf-8')
    elif isinstance(value, str):
        raw = value.encode('utf-8')
    else:
        raw = value
    return base64.b64encode(raw).decode('ascii')


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Return ``pipeline(files, **behaviour)`` -> :class:`runner.RunOutcome`.

    ``files`` maps a path relative to the adapter's ``--output-dir`` to the
    content to write there.
    """
    package = tmp_path / 'stand_in_pkg'
    package.mkdir()
    (package / '__init__.py').write_text('', encoding='utf-8')
    (package / 'adapter.py').write_text(
        textwrap.dedent(STAND_IN), encoding='utf-8'
    )
    runs = 0

    def go(files, *, run_kwargs=None, **behaviour):
        nonlocal runs
        runs += 1
        (package / 'files.json').write_text(
            json.dumps({name: encode(value) for name, value in files.items()}),
            encoding='utf-8',
        )
        (package / 'behaviour.json').write_text(
            json.dumps(behaviour), encoding='utf-8'
        )
        spec = AdapterSpec(
            key='demo',
            module='stand_in_pkg.adapter',
            collections=(COLLECTION,),
            timeout_minutes=5,
        )
        return runner.run(
            spec,
            tmp_path / f'work{runs}',
            run_date=RUN_DATE,
            base_env={'PYTHONPATH': str(tmp_path)},
            **(run_kwargs or {}),
        )

    return go


def published(outcome, *parts: str) -> Path:
    return outcome.upload_dir.joinpath('data', COLLECTION, *parts)


# --- the happy path ------------------------------------------------------


def test_a_clean_run_stages_validates_and_stamps(pipeline) -> None:
    record = record_without_samples()

    outcome = pipeline({f'demo-org/demo-model/{UUID_A}.json': record})

    assert outcome.status == 'completed'
    assert outcome.ok and outcome.has_upload
    assert [r.model_id for r in outcome.records] == ['demo-org/demo-model']
    assert (outcome.validation.valid, outcome.validation.invalid) == (1, 0)
    assert outcome.validation.warnings == 0

    result = json.loads(
        published(
            outcome, 'demo-org', 'demo-model', f'{UUID_A}.json'
        ).read_text('utf-8')
    )
    assert is_cron_record(result)
    details = result['source_metadata']['additional_details']
    assert details['cron_run_date'] == '2026-08-10'
    assert details['cron_adapter'] == 'demo'
    assert 'cron_run_url' not in details


def test_the_run_url_reaches_the_published_record(pipeline) -> None:
    outcome = pipeline(
        {f'demo-org/demo-model/{UUID_A}.json': record_without_samples()},
        run_kwargs={'run_url': 'https://ci.example/run/7'},
    )

    result = json.loads(
        published(
            outcome, 'demo-org', 'demo-model', f'{UUID_A}.json'
        ).read_text('utf-8')
    )
    assert (
        result['source_metadata']['additional_details']['cron_run_url']
        == 'https://ci.example/run/7'
    )


def test_nothing_but_provenance_changes_on_the_way_to_the_upload_tree(
    pipeline,
) -> None:
    record = record_without_samples()

    outcome = pipeline({f'demo-org/demo-model/{UUID_A}.json': record})

    result = json.loads(
        published(
            outcome, 'demo-org', 'demo-model', f'{UUID_A}.json'
        ).read_text('utf-8')
    )
    del result['source_metadata']
    expected = dict(record)
    del expected['source_metadata']
    assert result == expected


def test_a_samples_sidecar_is_published_byte_for_byte(pipeline) -> None:
    record, payload = record_with_samples()

    outcome = pipeline(
        {
            f'demo-org/demo-model/{UUID_A}.json': record,
            f'demo-org/demo-model/{UUID_A}_samples.jsonl': payload,
        }
    )

    assert outcome.status == 'completed', outcome.messages
    assert outcome.records[0].samples_repo_path == (
        f'data/{COLLECTION}/demo-org/demo-model/{UUID_A}_samples.jsonl'
    )
    assert (
        published(
            outcome, 'demo-org', 'demo-model', f'{UUID_A}_samples.jsonl'
        ).read_bytes()
        == payload
    )


# --- de-duplication ------------------------------------------------------


def test_the_fingerprint_is_taken_before_stamping(pipeline) -> None:
    """Otherwise the daily run date makes every record look new."""
    record = record_without_samples()

    outcome = pipeline({f'demo-org/demo-model/{UUID_A}.json': record})

    assert outcome.records[0].fingerprint == normalized_hash(record)


def test_a_record_seen_on_a_previous_run_is_skipped_and_listed(
    pipeline,
) -> None:
    record = record_without_samples()

    outcome = pipeline(
        {f'demo-org/demo-model/{UUID_A}.json': record},
        run_kwargs={'known_fingerprints': {normalized_hash(record)}},
    )

    assert outcome.status == 'completed'
    assert outcome.uploaded == []
    assert [r.model_id for r in outcome.skipped_unchanged] == [
        'demo-org/demo-model'
    ]
    assert not list(outcome.upload_dir.rglob('*.json'))
    assert 'unchanged since the last run' in outcome.coverage_line()
    assert outcome.to_manifest()['skipped_unchanged'][0]['model_id'] == (
        'demo-org/demo-model'
    )


def test_only_the_changed_record_of_a_batch_is_published(pipeline) -> None:
    unchanged = record_without_samples('model-a')
    changed = record_without_samples('model-b')

    outcome = pipeline(
        {
            f'demo-org/model-a/{UUID_A}.json': unchanged,
            f'demo-org/model-b/{UUID_B}.json': changed,
        },
        run_kwargs={'known_fingerprints': {normalized_hash(unchanged)}},
    )

    assert [r.model_id for r in outcome.uploaded] == ['demo-org/model-b']
    assert [r.model_id for r in outcome.skipped_unchanged] == [
        'demo-org/model-a'
    ]


def test_force_full_republishes_a_known_record(pipeline) -> None:
    record = record_without_samples()

    outcome = pipeline(
        {f'demo-org/demo-model/{UUID_A}.json': record},
        run_kwargs={
            'known_fingerprints': {normalized_hash(record)},
            'force_full': True,
        },
    )

    assert len(outcome.uploaded) == 1
    assert outcome.skipped_unchanged == []
    assert any('bypassed' in message for message in outcome.messages)


# --- refusals ------------------------------------------------------------


def test_an_invalid_record_blocks_the_whole_upload(pipeline) -> None:
    good = record_without_samples('model-a')
    bad = record_without_samples('model-b')
    bad['evaluation_results'][0]['score_details']['score'] = 42.0

    outcome = pipeline(
        {
            f'demo-org/model-a/{UUID_A}.json': good,
            f'demo-org/model-b/{UUID_B}.json': bad,
        }
    )

    assert outcome.status == 'failed'
    assert not outcome.ok
    assert outcome.validation.invalid == 1
    assert not list(outcome.upload_dir.rglob('*.json'))
    assert any('outside' in problem for problem in outcome.validation.problems)


def test_a_record_outside_a_declared_collection_is_refused(pipeline) -> None:
    outcome = pipeline(
        {
            f'../other-collection/demo-org/demo-model/{UUID_A}.json': (
                record_without_samples()
            )
        }
    )

    assert outcome.status == 'failed'
    assert any('is not declared' in message for message in outcome.messages)
    assert not list(outcome.upload_dir.rglob('*.json'))


def test_a_misshapen_datastore_path_is_refused(pipeline) -> None:
    outcome = pipeline({f'{UUID_A}.json': record_without_samples()})

    assert outcome.status == 'failed'
    assert any(
        'data/<collection>/<developer>/<model>/<file>' in message
        for message in outcome.messages
    )


def test_a_non_uuid_filename_is_refused(pipeline) -> None:
    outcome = pipeline(
        {'demo-org/demo-model/latest.json': record_without_samples()}
    )

    assert outcome.status == 'failed'
    assert any('uuid4' in message for message in outcome.messages)


def test_a_file_written_outside_the_staging_layout_is_refused(
    pipeline,
) -> None:
    outcome = pipeline(
        {f'demo-org/demo-model/{UUID_A}.json': record_without_samples()},
        stray_file='scratch/notes.txt',
    )

    assert outcome.status == 'failed'
    assert any(
        'outside its staging layout' in message for message in outcome.messages
    )


def test_a_samples_file_with_no_aggregate_is_refused(pipeline) -> None:
    _, payload = record_with_samples()

    outcome = pipeline(
        {
            f'demo-org/demo-model/{UUID_A}_samples.jsonl': payload,
            f'demo-org/demo-model/{UUID_B}.json': record_without_samples(),
        }
    )

    assert outcome.status == 'failed'
    assert any(
        'no sibling aggregate' in message for message in outcome.messages
    )


def test_duplicate_records_in_one_batch_block_the_upload(pipeline) -> None:
    record = record_without_samples()
    twin = json.loads(json.dumps(record))
    twin['evaluation_id'] = 'demo-source/demo-org_demo-model/1750000099'

    outcome = pipeline(
        {
            f'demo-org/demo-model/{UUID_A}.json': record,
            f'demo-org/demo-model/{UUID_B}.json': twin,
        }
    )

    assert outcome.status == 'failed'
    assert any('duplicate' in message for message in outcome.messages)
    assert not list(outcome.upload_dir.rglob('*.json'))


def test_an_empty_refresh_is_a_failure_not_a_quiet_success(pipeline) -> None:
    """ "0 valid, 0 failed" must never read as an up-to-date leaderboard."""
    outcome = pipeline({})

    assert outcome.status == 'failed'
    assert any('no records' in message for message in outcome.messages)


def test_a_crash_with_no_provenance_report_fails(pipeline) -> None:
    outcome = pipeline(
        {f'demo-org/demo-model/{UUID_A}.json': record_without_samples()},
        exit_code=1,
    )

    assert outcome.status == 'failed'
    assert any('exited 1' in message for message in outcome.messages)
    assert not list(outcome.upload_dir.rglob('*.json'))


# --- partial conversions -------------------------------------------------

PARTIAL_REPORT = {
    'source_name': 'demo',
    'total_source_records': 3,
    'converted_records': 1,
    'failed_record_count': 2,
    'excluded_record_count': 0,
    'failed_records': [
        {'source_ref': 'row 2', 'reason': 'missing model identity'}
    ],
    'excluded_records': [],
}


def test_a_partial_conversion_publishes_what_converted(pipeline) -> None:
    outcome = pipeline(
        {f'demo-org/demo-model/{UUID_A}.json': record_without_samples()},
        exit_code=1,
        failure_report=PARTIAL_REPORT,
    )

    assert outcome.status == 'partial'
    assert outcome.ok
    assert len(outcome.uploaded) == 1
    assert outcome.coverage['total_source_records'] == 3
    assert outcome.coverage['failed_record_count'] == 2
    assert outcome.coverage['example_reasons'] == ['missing model identity']
    line = outcome.coverage_line()
    assert '3 source row(s)' in line
    assert '2 dropped' in line
    assert '1 uploaded' in line


def test_an_adapter_that_forbids_partial_runs_publishes_nothing(
    tmp_path, pipeline
) -> None:
    outcome = pipeline(
        {f'demo-org/demo-model/{UUID_A}.json': record_without_samples()},
        exit_code=1,
        failure_report=PARTIAL_REPORT,
        run_kwargs={},
    )
    assert outcome.status == 'partial'

    # Same run, with the adapter marked as requiring a complete refresh.
    spec = AdapterSpec(
        key='demo',
        module='stand_in_pkg.adapter',
        collections=(COLLECTION,),
        timeout_minutes=5,
        allow_partial=False,
    )
    strict = runner.run(
        spec,
        tmp_path / 'strict',
        run_date=RUN_DATE,
        base_env={'PYTHONPATH': str(tmp_path)},
    )

    assert strict.status == 'failed'
    assert any(
        'require a complete one' in message for message in strict.messages
    )


# --- configuration gaps are not failures ---------------------------------


def test_a_missing_credential_is_its_own_outcome(tmp_path) -> None:
    spec = AdapterSpec(
        key='needs-key',
        module='every_eval_ever.adapters.hle.adapter',
        collections=('hle',),
        required_env=('DEMO_API_KEY',),
    )

    outcome = runner.run(
        spec, tmp_path / 'work', run_date=RUN_DATE, base_env={}
    )

    assert outcome.status == 'skipped_missing_credential'
    assert outcome.ok and not outcome.has_upload
    assert 'DEMO_API_KEY' in outcome.messages[0]
    assert outcome.process is None


def test_a_missing_optional_dependency_is_its_own_outcome(tmp_path) -> None:
    spec = AdapterSpec(
        key='needs-package',
        module='every_eval_ever.adapters.hle.adapter',
        collections=('hle',),
        with_packages=('a_package_that_does_not_exist',),
    )

    outcome = runner.run(
        spec, tmp_path / 'work', run_date=RUN_DATE, base_env={}
    )

    assert outcome.status == 'skipped_missing_dependency'
    assert outcome.ok
    assert 'a_package_that_does_not_exist' in outcome.messages[0]


# --- isolation and plumbing ----------------------------------------------


def test_an_adapter_is_not_handed_another_adapters_credentials(
    tmp_path,
) -> None:
    spec = AdapterSpec(
        key='one-key',
        module='every_eval_ever.adapters.llm_stats.adapter',
        collections=('llm-stats',),
        required_env=('LLM_STATS_API_KEY',),
    )

    env = runner.adapter_environment(
        spec,
        raw_dir=tmp_path / 'raw',
        base_env={
            'LLM_STATS_API_KEY': 'mine',
            'ARTIFICIAL_ANALYSIS_API_KEY': 'not mine',
            'PATH': '/usr/bin',
        },
    )

    assert env['LLM_STATS_API_KEY'] == 'mine'
    assert 'ARTIFICIAL_ANALYSIS_API_KEY' not in env
    assert env['PATH'] == '/usr/bin'
    assert env['EEE_RAW_CAPTURE_DIR'] == str(tmp_path / 'raw')


def test_adapter_output_encoding_does_not_depend_on_the_platform(
    tmp_path,
) -> None:
    """An adapter must not die on the arrow in its own summary line."""
    package = tmp_path / 'unicode_pkg'
    package.mkdir()
    (package / '__init__.py').write_text('', encoding='utf-8')
    (package / 'adapter.py').write_text(
        "print('done \u2192 data/demo')\n", encoding='utf-8'
    )
    spec = AdapterSpec(
        key='chatty',
        module='unicode_pkg.adapter',
        collections=(COLLECTION,),
    )

    process = runner.run_adapter(
        spec,
        data_root=tmp_path / 'data',
        raw_dir=tmp_path / 'raw',
        base_env={'PYTHONPATH': str(tmp_path)},
    )

    assert process.ok, process.stderr
    assert 'done → data/demo' in process.stdout


def test_raw_capture_is_pointed_at_this_runs_own_directory(pipeline) -> None:
    outcome = pipeline(
        {f'demo-org/demo-model/{UUID_A}.json': record_without_samples()}
    )

    assert outcome.raw_dir.name == 'raw'
    assert outcome.raw_dir.parent == outcome.staging_dir.parent


def test_a_hanging_adapter_is_killed_and_reported(tmp_path) -> None:
    """A stuck scrape must end the job, not occupy the runner forever."""
    package = tmp_path / 'hang_pkg'
    package.mkdir()
    (package / '__init__.py').write_text('', encoding='utf-8')
    (package / 'adapter.py').write_text(
        'import time\ntime.sleep(120)\n', encoding='utf-8'
    )
    spec = AdapterSpec(
        key='slow',
        module='hang_pkg.adapter',
        collections=(COLLECTION,),
        # A fraction of a minute, so the test exercises the timeout without
        # waiting for the smallest budget a real adapter would be given.
        timeout_minutes=0.02,
    )

    process = runner.run_adapter(
        spec,
        data_root=tmp_path / 'data',
        raw_dir=tmp_path / 'raw',
        base_env={'PYTHONPATH': str(tmp_path)},
    )

    assert process.argv[0] == sys.executable
    assert process.argv[1:3] == ['-m', 'hang_pkg.adapter']
    assert process.timed_out
    assert not process.ok
    assert process.returncode != 0
    assert 'timed out' in process.stderr


def test_validation_of_an_empty_tree_is_not_publishable(tmp_path) -> None:
    staging = tmp_path / 'staging'
    (staging / 'data').mkdir(parents=True)

    summary = runner.validate_staging(staging)

    assert not summary.publishable
    assert summary.valid == 0
