"""Unit tests for the MMLU-Pro adapter."""

from __future__ import annotations

from every_eval_ever.eval_types import EvaluationLog
from utils.mmlu_pro import adapter


def sample_rows() -> list[dict]:
    return [
        {
            'Models': 'GPT-4o (2024-05-13)',
            'Data Source': 'TIGER-Lab',
            'Model Size(B)': 'unk',
            'Overall': '0.7255',
            'Biology': '0.8675',
            'Business': '0.7858',
            'Chemistry': '0.7393',
            'Computer Science': '0.7829',
            'Economics': '0.808',
            'Engineering': '0.55',
            'Health': '0.7212',
            'History': '0.7007',
            'Law': '0.5104',
            'Math': '0.7609',
            'Philosophy': '0.7014',
            'Physics': '0.7467',
            'Psychology': '0.7919',
            'Other': '0.7748',
        },
        # Same model, different data source, different score — both are
        # legitimate measurements and should produce two distinct logs.
        {
            'Models': 'Claude-3.5-Sonnet (2024-10-22)',
            'Data Source': 'TIGER-LAb',  # typo in upstream CSV
            'Model Size(B)': 'unk',
            'Overall': '0.7764',
        },
        {
            'Models': 'Claude-3.5-Sonnet (2024-10-22)',
            'Data Source': 'Self-Reported',
            'Model Size(B)': 'unk',
            'Overall': '0.780',
        },
        # Exact duplicate — should be dropped.
        {
            'Models': 'LLaDA',
            'Data Source': 'Self-Reported',
            'Model Size(B)': '8',
            'Overall': '0.370',
        },
        {
            'Models': 'LLaDA',
            'Data Source': 'Self-Reported',
            'Model Size(B)': '8',
            'Overall': '0.370',
        },
        # Missing overall score — should be skipped.
        {
            'Models': 'broken-model',
            'Data Source': 'Self-Reported',
            'Model Size(B)': 'unk',
            'Overall': '',
        },
        # EXAONE — exercises DEVELOPER_OVERRIDES.
        {
            'Models': 'EXAONE-3.5-2.4B-Instruct',
            'Data Source': 'TIGER-Lab',
            'Model Size(B)': '2.4',
            'Overall': '0.391',
        },
    ]


def test_make_logs_validate_against_schema():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    # GPT-4o + 2 Claude variants + LLaDA (deduped) + EXAONE = 5 logs;
    # broken-model is dropped, second LLaDA is dropped.
    assert len(bundles) == 5
    for log, _, _ in bundles:
        validated = EvaluationLog.model_validate(log.model_dump())
        assert validated.schema_version == '0.2.2'
        assert validated.source_metadata.source_organization_name == 'TIGER-Lab'
        assert validated.source_metadata.source_type.value == 'documentation'


def test_overall_and_subject_results_per_model():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    by_id = {b[0].model_info.id: b[0] for b in bundles}
    gpt4o = by_id['openai/gpt-4o-2024-05-13']
    ids = {r.evaluation_result_id for r in gpt4o.evaluation_results}
    # Overall + 14 subjects.
    assert 'mmlu_pro/overall' in ids
    assert 'mmlu_pro/biology' in ids
    assert 'mmlu_pro/computer_science' in ids
    assert 'mmlu_pro/other' in ids
    assert len(ids) == 15


def test_two_data_sources_for_same_model_yield_two_logs():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    claude_logs = [
        b[0]
        for b in bundles
        if b[0].model_info.id == 'anthropic/claude-3.5-sonnet-2024-10-22'
    ]
    assert len(claude_logs) == 2
    eval_ids = {log.evaluation_id for log in claude_logs}
    # Each log carries the data-source slug in its evaluation_id.
    assert any('tiger-lab' in eid for eid in eval_ids)
    assert any('self-reported' in eid for eid in eval_ids)


def test_data_source_typos_are_normalized():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    claude_logs = [
        b[0]
        for b in bundles
        if b[0].model_info.id == 'anthropic/claude-3.5-sonnet-2024-10-22'
    ]
    sources = {
        (log.source_metadata.additional_details or {}).get(
            'leaderboard_data_source'
        )
        for log in claude_logs
    }
    # 'TIGER-LAb' must be canonicalized to 'TIGER-Lab'.
    assert sources == {'TIGER-Lab', 'Self-Reported'}

    # The raw upstream value must also be preserved when canonicalization
    # changed it, so reviewers can audit the normalization.
    raw_typo_log = next(
        log
        for log in claude_logs
        if (log.source_metadata.additional_details or {}).get(
            'leaderboard_data_source'
        )
        == 'TIGER-Lab'
        and 'raw_leaderboard_data_source'
        in (log.source_metadata.additional_details or {})
    )
    assert (
        raw_typo_log.source_metadata.additional_details[
            'raw_leaderboard_data_source'
        ]
        == 'TIGER-LAb'
    )
    assert (
        raw_typo_log.model_info.additional_details[
            'raw_leaderboard_data_source'
        ]
        == 'TIGER-LAb'
    )

    # When the canonical and raw values match (no typo), the raw key is
    # NOT added — avoids redundant noise.
    self_reported = next(
        log
        for log in claude_logs
        if (log.source_metadata.additional_details or {}).get(
            'leaderboard_data_source'
        )
        == 'Self-Reported'
    )
    assert (
        'raw_leaderboard_data_source'
        not in (self_reported.source_metadata.additional_details or {})
    )


def test_exact_duplicate_rows_are_dropped():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    llada = [b for b in bundles if 'llada' in b[0].model_info.id.lower()]
    assert len(llada) == 1


def test_developer_override_for_exaone():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    exaone = next(
        b[0] for b in bundles if 'exaone' in b[0].model_info.id.lower()
    )
    assert exaone.model_info.developer == 'lg-ai'


def test_metric_config_uses_zero_to_one_continuous_scale():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    log = bundles[0][0]
    overall = next(
        r
        for r in log.evaluation_results
        if r.evaluation_result_id == 'mmlu_pro/overall'
    )
    assert overall.metric_config.min_score == 0.0
    assert overall.metric_config.max_score == 1.0
    assert overall.metric_config.lower_is_better is False
    assert overall.metric_config.metric_kind == 'accuracy'
    assert overall.metric_config.metric_unit == 'proportion'


def test_model_size_captured_when_known():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    exaone = next(
        b[0] for b in bundles if 'exaone' in b[0].model_info.id.lower()
    )
    assert (
        exaone.model_info.additional_details['size_billions_parameters']
        == '2.4'
    )

    gpt4o = next(
        b[0]
        for b in bundles
        if b[0].model_info.id == 'openai/gpt-4o-2024-05-13'
    )
    assert 'size_billions_parameters' not in (
        gpt4o.model_info.additional_details or {}
    )


def test_source_data_is_hf_dataset_pointing_at_results_repo():
    bundles = adapter.make_logs(sample_rows(), retrieved_timestamp='123.0')
    log = bundles[0][0]
    overall = log.evaluation_results[0]
    assert overall.source_data.source_type == 'hf_dataset'
    assert overall.source_data.hf_repo == adapter.RESULTS_HF_REPO
