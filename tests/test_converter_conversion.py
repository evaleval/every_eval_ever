"""Convert a committed upstream log with the real CLI and put it through the gate.

Someone ran an evaluation: can this repo still turn that output into a record the
datastore accepts? Each case converts a log the upstream tool wrote — never a
hand-built one — and requires the published files to pass `validate` with the semantic
checks on and no warnings, since a warning here is one every user inherits.

A *new version* of the upstream tool is out of reach here; that is what a
`tools/upstream_smoke/` script covers, on a schedule. Neither layer checks semantics: a
metric that changes from a percentage to a proportion upstream passes both.

Cases live in `tests/converter_cases.py`.
"""

from __future__ import annotations

import json

import pytest

from tests.converter_cases import (
    CASES,
    ConverterCase,
    convert,
    gate_complaints,
    missing_paths,
    unavailable,
)

CASE_IDS = [case.id for case in CASES]


@pytest.fixture(params=CASES, ids=CASE_IDS)
def case(request) -> ConverterCase:
    reason = unavailable(request.param)
    if reason is not None:
        pytest.skip(reason)
    return request.param


def test_conversion_passes_the_merge_gate(case, tmp_path, capsys):
    """Everything the converter publishes must be submittable as-is."""
    paths = convert(case, tmp_path)
    complaints = gate_complaints(paths, capsys)

    assert not complaints, (
        f'{case.source} conversion no longer passes the merge gate:\n'
        + json.dumps(complaints, indent=2)
        + f'\nFix every_eval_ever/converters/{case.source}/, or the schema, '
        'validator or publisher change that caused it.'
    )


def test_conversion_yields_the_expected_records(case, tmp_path):
    """Guard against a conversion that validates but quietly loses data."""
    paths = convert(case, tmp_path)
    aggregates = [path for path in paths if path.suffix == '.json']
    sidecars = [path for path in paths if path.suffix == '.jsonl']

    assert len(aggregates) == case.aggregates, (
        f'{case.source} produced {len(aggregates)} aggregate record(s), '
        f'expected {case.aggregates}: {[path.name for path in aggregates]}'
    )
    assert len(sidecars) == case.sidecars, (
        f'{case.source} produced {len(sidecars)} instance-level sidecar(s), '
        f'expected {case.sidecars}'
    )

    logs = [json.loads(path.read_text(encoding='utf-8')) for path in aggregates]
    if case.model_id is not None:
        assert {log['model_info']['id'] for log in logs} == {case.model_id}
    if case.results is not None:
        converted = sum(len(log['evaluation_results']) for log in logs)
        assert converted == case.results, (
            f'{case.source} converted {converted} result(s), '
            f'expected {case.results}'
        )
    if case.scores is not None:
        scored = [
            (
                f'{result["evaluation_name"]}/'
                f'{result.get("evaluation_result_id") or result["metric_config"]["evaluation_description"]}',
                result['score_details']['score'],
            )
            for log in logs
            for result in log['evaluation_results']
        ]
        # Collected as pairs, not straight into a dict: two results sharing a key
        # is the quietly-lost data this test exists to catch, and a dict would
        # merge them before the comparison below could see the collision.
        keys = [key for key, _ in scored]
        assert len(keys) == len(set(keys)), (
            f'{case.source} produced two results with the same '
            f'evaluation_name/evaluation_result_id: '
            f'{sorted(key for key in set(keys) if keys.count(key) > 1)}'
        )
        assert dict(scored) == case.scores
    if case.metric_names is not None:
        # `.get`, because an unset `metric_name` is absent from the record rather than
        # null — which is the case this assertion exists to report.
        converted = {
            result['metric_config'].get('metric_name')
            for log in logs
            for result in log['evaluation_results']
        }
        assert converted == case.metric_names, (
            f'{case.source} named its metrics {sorted(converted, key=str)}, '
            f'expected {sorted(case.metric_names)}. The metric belongs in '
            '`metric_config.metric_name`; `None` here means the converter left it '
            'unset, and `evaluation_name` is for the evaluation.'
        )

    for log, path in zip(logs, aggregates, strict=True):
        detailed = log.get('detailed_evaluation_results')
        if detailed is None:
            continue
        # The sidecar pointer must name the file that was written beside it, since
        # the gate resolves it as a repository path.
        assert detailed['file_path'].endswith(f'{path.stem}_samples.jsonl')
        assert detailed['total_rows'] > 0

    if case.sidecar_rows is not None:
        declared = sum(
            log['detailed_evaluation_results']['total_rows']
            for log in logs
            if log.get('detailed_evaluation_results')
        )
        written = sum(
            len(
                [
                    line
                    for line in path.read_text(encoding='utf-8').splitlines()
                    if line.strip()
                ]
            )
            for path in sidecars
        )
        assert declared == case.sidecar_rows, (
            f'{case.source} reported {declared} instance-level row(s), expected '
            f'{case.sidecar_rows}. A row is owed per aggregate result a sample '
            'contributed to, so this changes when the results do.'
        )
        # `total_rows` is what a reader trusts without opening the sidecar.
        assert written == declared, (
            f'{case.source} wrote {written} row(s) but reported {declared}'
        )


def test_required_source_paths_are_present_in_the_fixture(case):
    """The keys the converter reads must still exist in the committed log.

    A fixture refreshed from a newer upstream release fails here by name, instead of
    as a stack trace from inside the converter.
    """
    if not case.required_source_paths:
        pytest.skip(f'{case.source} declares no required source paths')

    absent = missing_paths(case.source_payload(), case.required_source_paths)

    assert not absent, (
        f'{case.source} reads upstream keys that {case.log_path.name} does not have: '
        f'{absent}\nEither the fixture came from an incompatible version, or the '
        f'converter and tests/converter_cases.py disagree about what it reads.'
    )
