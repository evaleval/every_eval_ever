"""Tests for edge cases discovered during real-world Inspect log conversion.

These tests exercise scenarios encountered when converting logs from:
- CVE-Bench (agentic, sandbox-based) — samples with empty output.choices
- inspect_evals dashboard — datasets with local filesystem paths in
  dataset.location and misleading dataset names

Fixtures are stripped-down real logs with events removed and messages trimmed.

Tests that exercise already-fixed issues serve as regression guards;
tests that still xfail document issues in the current adapter that are
yet to be addressed.
"""

import pytest

pytest.importorskip(
    'inspect_ai',
    reason='inspect-ai not installed; install with: uv sync --extra inspect',
)

import json
import tempfile
from pathlib import Path

from every_eval_ever.converters.inspect.adapter import InspectAIAdapter
from every_eval_ever.eval_types import (
    EvaluatorRelationship,
    SourceDataHf,
    SourceDataPrivate,
)
from every_eval_ever.instance_level_types import InstanceLevelEvaluationLog

FIXTURES = Path(__file__).resolve().parent / 'data/inspect'
METADATA_ARGS = {
    'source_organization_name': 'TestOrg',
    'evaluator_relationship': EvaluatorRelationship.first_party,
}


def _load_eval_and_instances(filepath, metadata_args=None):
    """Load a log through the adapter and return (EvaluationLog, [InstanceLevelEvaluationLog])."""
    adapter = InspectAIAdapter()
    if metadata_args is None:
        metadata_args = {}
    args = {**METADATA_ARGS, **metadata_args}
    args.setdefault('file_uuid', 'test-uuid')

    with tempfile.TemporaryDirectory() as tmpdir:
        args['parent_eval_output_dir'] = tmpdir
        converted = adapter.transform_from_file(str(filepath), args)

        instance_logs = []
        if converted.detailed_evaluation_results:
            instance_path = Path(
                converted.detailed_evaluation_results.file_path
            )
            if instance_path.exists():
                with instance_path.open('r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            instance_logs.append(
                                InstanceLevelEvaluationLog.model_validate(
                                    json.loads(line)
                                )
                            )

    return converted, instance_logs


# -----------------------------------------------------------------------
# 1. Empty output.choices
#    Some agentic samples (e.g. sandbox failures) have no model output,
#    resulting in sample.output.choices == [].  The instance-level
#    adapter indexes choices[0] unconditionally, raising IndexError.
#    A possible fix: guard the access with `if sample.output.choices:`
#    and fall back to empty content.
# -----------------------------------------------------------------------


class TestEmptyOutputChoices:
    """Tests for samples where output.choices is an empty list.

    The fixture contains a single sample from a successful CVE-Bench run
    whose output.choices is [] because the sandbox failed to start.
    """

    def test_conversion_does_not_crash(self):
        """Converting a sample with empty choices must not raise."""
        converted, instances = _load_eval_and_instances(
            FIXTURES / 'data_cvebench_empty_choices.json'
        )
        assert len(instances) == 1
        # stop_reason should be empty when choices is empty; the field is
        # documented as reflecting the first choice.
        assert instances[0].metadata['stop_reason'] == ''


# -----------------------------------------------------------------------
# 2. Local filesystem path in dataset.location → hf_repo
#    When Inspect evals use local/cached datasets, dataset.location
#    contains an absolute filesystem path (e.g. /root/.cache/...) which
#    gets passed through to hf_repo verbatim.  The adapter should
#    detect non-HF paths and either use a different SourceData variant
#    or omit the field.
# -----------------------------------------------------------------------


class TestLocalFilesystemDatasetPath:
    """When dataset.location is a local path, source_data should be a
    SourceDataPrivate rather than a SourceDataHf with a local hf_repo."""

    def test_intercode_ctf_source_not_hf(self):
        """gdm_intercode_ctf uses a cached local JSON file — source_type
        should be 'other' (SourceDataPrivate)."""
        converted, _ = _load_eval_and_instances(
            FIXTURES / 'data_intercode_ctf_local_path.json'
        )
        sd = converted.evaluation_results[0].source_data
        assert isinstance(sd, SourceDataPrivate), (
            f'Expected SourceDataPrivate, got {type(sd).__name__}'
        )
        assert not isinstance(sd, SourceDataHf)
        # The original location should be preserved for traceability.
        assert 'inspect_dataset_location' in (sd.additional_details or {})

    def test_cvebench_source_not_hf(self):
        """CVE-Bench uses a local challenges directory — source_type
        should be 'other' (SourceDataPrivate)."""
        converted, _ = _load_eval_and_instances(
            FIXTURES / 'data_cvebench_empty_choices.json'
        )
        sd = converted.evaluation_results[0].source_data
        assert isinstance(sd, SourceDataPrivate), (
            f'Expected SourceDataPrivate, got {type(sd).__name__}'
        )
        assert not isinstance(sd, SourceDataHf)
        assert 'inspect_dataset_location' in (sd.additional_details or {})


# -----------------------------------------------------------------------
# 3. Misleading dataset_name from Inspect log metadata
#    Some inspect_evals use internal filenames as the dataset name
#    (e.g. 'challenges' for cyberseceval_2 vulnerability_exploit,
#    'ic_ctf' for gdm_intercode_ctf).  The adapter should prefer the
#    task name when the dataset name is ambiguous or generic.
# -----------------------------------------------------------------------


class TestMisleadingDatasetNames:
    """dataset_name should identify the benchmark, not an internal filename."""

    @pytest.mark.xfail(
        reason="adapter uses dataset.name ('challenges') instead of the "
        "task name ('cyse2_vulnerability_exploit')",
        strict=True,
    )
    def test_cyse2_vuln_exploit_dataset_name(self):
        """CybersecEval2 vulnerability_exploit should not be named
        'challenges'."""
        converted, _ = _load_eval_and_instances(
            FIXTURES / 'data_cyse2_vuln_exploit_challenges.json'
        )
        sd = converted.evaluation_results[0].source_data
        assert sd.dataset_name != 'challenges', (
            'dataset_name should reflect the benchmark, not the internal '
            f'filename — got {sd.dataset_name!r}'
        )

    @pytest.mark.xfail(
        reason="adapter uses dataset.name ('ic_ctf') instead of the "
        "task name ('gdm_intercode_ctf')",
        strict=True,
    )
    def test_intercode_ctf_dataset_name(self):
        """gdm_intercode_ctf should not be named 'ic_ctf'."""
        converted, _ = _load_eval_and_instances(
            FIXTURES / 'data_intercode_ctf_local_path.json'
        )
        sd = converted.evaluation_results[0].source_data
        assert sd.dataset_name != 'ic_ctf', (
            'dataset_name should reflect the benchmark, not the internal '
            f'filename — got {sd.dataset_name!r}'
        )
