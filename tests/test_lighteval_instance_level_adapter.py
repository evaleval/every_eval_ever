"""Convert a committed lighteval run and require the output to pass the gate.

The fixture read here was produced by lighteval itself
(`scripts/upstream_smoke/lighteval_smoke.py`, with their DummyModelConfig), not
hand-written, so the parquet layout, the numpy dtypes and the task-key spelling
are all upstream's rather than ours.

Scope: shape, not semantics. A metric that switched from percent to proportion
upstream passes this, as does a changed prompt template. What it does catch is a
converter that stops finding the details file, stops resolving the gold, or
starts emitting records the datastore would reject.
"""

import json

import pytest

pytest.importorskip(
    'pyarrow',
    reason='no parquet engine; install with: uv sync --extra lighteval',
)

from pathlib import Path

from every_eval_ever import cli
from every_eval_ever.converters.lighteval.utils import find_details_file

FIXTURE_DIR = Path(__file__).parent / 'data' / 'lighteval_smoke'
RESULTS_FILE = (
    FIXTURE_DIR
    / 'results'
    / 'eee-smoke'
    / 'dummy-model'
    / 'results_2026-08-10T17-22-11.385681.json'
)

# One multiple-choice task and one generative task. Their details rows differ in
# every way that matters here: the MC row answers by scoring choices and carries
# no generated text, the generative row is the reverse.
MC_TASK = 'anli:r1|0'
GENERATIVE_TASK = 'squad_v2|0'


def _convert(tmp_path: Path) -> Path:
    """Run the real CLI, as a user would, and return the published data dir."""
    data_dir = tmp_path / 'data'
    exit_code = cli.main(
        [
            'convert',
            'lighteval',
            '--log_path',
            str(RESULTS_FILE),
            '--include_details',
            '--output_dir',
            str(data_dir),
        ]
    )
    assert exit_code == 0
    return data_dir


def _samples_by_task(data_dir: Path) -> dict[str, list[dict]]:
    """Group every published instance-level record by the task it came from."""
    grouped: dict[str, list[dict]] = {}
    for path in sorted(data_dir.glob('*/*/*/*_samples.jsonl')):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding='utf-8').splitlines()
            if line
        ]
        assert rows, f'{path} is empty'
        grouped.setdefault(rows[0]['evaluation_name'], []).extend(rows)
    return grouped


def test_details_file_is_found_for_each_task():
    for task_key in (MC_TASK, GENERATIVE_TASK):
        found = find_details_file(
            RESULTS_FILE, task_key, 'eee-smoke/dummy-model'
        )
        assert found is not None, f'no details file located for {task_key}'
        assert found.name.startswith(f'details_{task_key}_')


def test_conversion_publishes_aggregates_and_samples(tmp_path):
    data_dir = _convert(tmp_path)

    aggregates = sorted(data_dir.glob('*/*/*/*.json'))
    samples = sorted(data_dir.glob('*/*/*/*_samples.jsonl'))
    assert len(aggregates) == 2
    assert len(samples) == 2

    for aggregate in aggregates:
        record = json.loads(aggregate.read_text(encoding='utf-8'))
        detailed = record['detailed_evaluation_results']
        assert detailed is not None
        # The sidecar's declared repository path has to be the one publication
        # actually used, or a submission points at a file that is not there.
        assert (
            Path(detailed['file_path']).name
            == f'{aggregate.stem}_samples.jsonl'
        )
        assert detailed['total_rows'] == 2


def test_published_records_pass_the_datastore_gate(tmp_path):
    data_dir = _convert(tmp_path)
    for pattern in ('*.json', '*.jsonl'):
        # Exit 0 is clean; 2 is warning-only, which is valid locally but not
        # merge-ready, so anything but 0 fails here.
        assert (
            cli.main(
                [
                    'validate',
                    str(data_dir / '*' / '*' / '*' / pattern),
                    '--format',
                    'rich',
                ]
            )
            == 0
        )


def test_multiple_choice_row_keeps_its_options_and_gold(tmp_path):
    rows = _samples_by_task(_convert(tmp_path))[MC_TASK]
    for row in rows:
        # gold_index is an int here, and reference is resolved through choices,
        # so an empty reference means the index was dropped rather than read.
        assert row['input']['reference']
        assert row['input']['choices']
        assert row['input']['reference'][0] in row['input']['choices']
        attribution = row['answer_attribution'][0]
        assert attribution['extraction_method'] == 'argmax_choice_logprob'
        assert attribution['extracted_value'] in row['input']['choices']
        assert json.loads(row['metadata']['choice_logprobs'])
        assert row['metadata']['lighteval_sampling_methods'] == 'LOGPROBS'


def test_generative_row_reports_gold_as_reference_not_as_choices(tmp_path):
    rows = _samples_by_task(_convert(tmp_path))[GENERATIVE_TASK]
    for row in rows:
        # gold_index is a numpy array of numpy ints on this row. Reading it with
        # an isinstance check against Python's own int drops every element and
        # leaves reference empty, which is why this asserts on content.
        assert row['input']['reference']
        # lighteval stores a generative task's gold answers in doc.choices.
        # Publishing them as input.choices would say the model was shown the
        # answer to choose from.
        assert row['input']['choices'] is None
        assert row['output']['raw']
        assert row['metadata']['lighteval_sampling_methods'] == 'GENERATIVE'


def test_missing_details_tree_is_reported_not_swallowed(tmp_path):
    """A run made without save_details must not look like a clean conversion."""
    from every_eval_ever.helpers.io import SourceRecordsError

    results_file = (
        Path(__file__).parent
        / 'data'
        / 'lighteval'
        / 'results'
        / 'HuggingFaceTB'
        / 'SmolLM2-1.7B-Instruct'
        / 'results_2026-01-21T03-44-18.458309.json'
    )
    data_dir = tmp_path / 'data'
    with pytest.raises(SourceRecordsError):
        cli.main(
            [
                'convert',
                'lighteval',
                '--log_path',
                str(results_file),
                '--include_details',
                '--output_dir',
                str(data_dir),
            ]
        )
    # The aggregates are still published, as they are for lm-eval: a missing
    # sidecar is a partial conversion, not a reason to discard usable records.
    assert sorted(data_dir.glob('*/*/*/*.json'))
    assert not sorted(data_dir.glob('*/*/*/*_samples.jsonl'))
    report = tmp_path / 'adapter_reports' / 'lighteval_details_failures.json'
    assert report.is_file()


def test_score_and_correctness_agree_with_the_recorded_metric(tmp_path):
    for rows in _samples_by_task(_convert(tmp_path)).values():
        for row in rows:
            metrics = json.loads(row['metadata']['lighteval_metrics'])
            primary = row['metadata']['primary_metric']
            assert primary in metrics
            assert row['evaluation']['score'] == pytest.approx(metrics[primary])
            assert row['evaluation']['is_correct'] is (
                row['evaluation']['score'] == 1.0
            )
