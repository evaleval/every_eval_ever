"""Unit tests for the MT-Bench adapter."""

from __future__ import annotations

from every_eval_ever.eval_types import EvaluationLog
from utils.mt_bench import adapter


def sample_rows() -> list[dict]:
    return [
        # Two complete sets of turn-1 + turn-2 judgments for two models.
        # gpt-4 averages: turn1=9.0, turn2=8.0, overall=8.5
        {
            'question_id': 81,
            'model': 'gpt-4',
            'judge': ['gpt-4', 'single-v1'],
            'score': 9,
            'turn': 1,
            'tstamp': 1687000000.0,
        },
        {
            'question_id': 81,
            'model': 'gpt-4',
            'judge': ['gpt-4', 'single-v1-multi-turn'],
            'score': 8,
            'turn': 2,
            'tstamp': 1687000010.0,
        },
        # vicuna-13b-v1.3 averages: turn1=6.0, turn2=4.0, overall=5.0
        {
            'question_id': 81,
            'model': 'vicuna-13b-v1.3',
            'judge': ['gpt-4', 'single-v1'],
            'score': 6,
            'turn': 1,
            'tstamp': 1687000020.0,
        },
        {
            'question_id': 81,
            'model': 'vicuna-13b-v1.3',
            'judge': ['gpt-4', 'single-v1-multi-turn'],
            'score': 4,
            'turn': 2,
            'tstamp': 1687000030.0,
        },
        # Score == -1 should be filtered (matches FastChat's show_result.py).
        {
            'question_id': 82,
            'model': 'gpt-4',
            'judge': ['gpt-4', 'single-v1'],
            'score': -1,
            'turn': 1,
            'tstamp': 1687000040.0,
        },
        # Row with no model -> ignored.
        {'question_id': 99, 'score': 5, 'turn': 1},
    ]


def test_make_logs_validate_against_schema():
    bundles = adapter.make_logs(
        sample_rows(), retrieved_timestamp='1234567890.0'
    )
    assert len(bundles) == 2

    for log, _, _ in bundles:
        validated = EvaluationLog.model_validate(log.model_dump())
        assert validated.schema_version == '0.2.2'
        assert validated.source_metadata.source_organization_name == 'LMSYS'
        assert validated.source_metadata.source_type.value == 'documentation'
        assert (
            validated.source_metadata.evaluator_relationship.value
            == 'third_party'
        )
        assert validated.eval_library.name == 'FastChat (llm_judge)'


def test_aggregation_matches_show_result_means():
    bundles = dict(
        (log.model_info.id, log)
        for log, _, _ in adapter.make_logs(
            sample_rows(), retrieved_timestamp='1234567890.0'
        )
    )

    gpt4 = bundles['openai/gpt-4']
    by_id = {r.evaluation_result_id: r for r in gpt4.evaluation_results}
    assert by_id['mt_bench/overall'].score_details.score == 8.5
    assert by_id['mt_bench/turn_1'].score_details.score == 9.0
    assert by_id['mt_bench/turn_2'].score_details.score == 8.0
    # The -1 score for question 82 must be filtered out.
    assert by_id['mt_bench/overall'].score_details.uncertainty.num_samples == 2


def test_metric_config_uses_one_to_ten_scale():
    bundles = adapter.make_logs(
        sample_rows(), retrieved_timestamp='1234567890.0'
    )
    log = bundles[0][0]
    overall = next(
        r
        for r in log.evaluation_results
        if r.evaluation_result_id == 'mt_bench/overall'
    )
    assert overall.metric_config.min_score == 1.0
    assert overall.metric_config.max_score == 10.0
    assert overall.metric_config.lower_is_better is False
    assert overall.metric_config.metric_kind == 'judge_score'
    assert overall.metric_config.metric_unit == 'points'
    assert overall.metric_config.llm_scoring is not None
    judge = overall.metric_config.llm_scoring.judges[0]
    assert judge.model_info.id == 'openai/gpt-4'


def test_developer_overrides_used_for_fastchat_models():
    bundles = adapter.make_logs(
        sample_rows(), retrieved_timestamp='1234567890.0'
    )
    by_dev = {bundle[1] for bundle in bundles}
    # vicuna-13b-v1.3 should land under 'lmsys' via DEVELOPER_OVERRIDES,
    # not under the generic 'unknown' fallback.
    assert 'lmsys' in by_dev
    assert 'openai' in by_dev
    assert 'unknown' not in by_dev


def test_evaluation_id_uses_sanitized_model_id():
    bundles = adapter.make_logs(
        sample_rows(), retrieved_timestamp='1234567890.0'
    )
    ids = {log.evaluation_id for log, _, _ in bundles}
    assert 'mt-bench/openai_gpt-4/1234567890.0' in ids
    assert 'mt-bench/lmsys_vicuna-13b-v1.3/1234567890.0' in ids


def test_source_data_url_contains_judgment_url():
    bundles = adapter.make_logs(
        sample_rows(), retrieved_timestamp='1234567890.0'
    )
    log = bundles[0][0]
    overall = log.evaluation_results[0]
    assert adapter.JUDGMENT_URL in overall.source_data.url
    assert overall.source_data.source_type == 'url'


def test_judge_model_not_duplicated_in_additional_details():
    """Per reviewer feedback on HF datastore PR (akornilo on #125): judge
    model info should live in metric_config.llm_scoring.judges[] only
    (schema L586), not be scattered across multiple additional_details
    bags. Mirrors the same guard added to the HLE adapter.
    """
    bundles = adapter.make_logs(
        sample_rows(), retrieved_timestamp='1234567890.0'
    )
    for log, _, _ in bundles:
        meta_extras = log.source_metadata.additional_details or {}
        assert 'judge_model' not in meta_extras
        assert 'judge_models_json' not in meta_extras
        for result in log.evaluation_results:
            metric_extras = result.metric_config.additional_details or {}
            assert 'judge_model' not in metric_extras
            assert 'judge_models_json' not in metric_extras
            source_extras = result.source_data.additional_details or {}
            assert 'judge_model' not in source_extras
            assert 'judge_models_json' not in source_extras
            # Sanity: judge IS in the prescribed slot.
            judges = result.metric_config.llm_scoring.judges
            assert len(judges) == 1
            assert judges[0].model_info.id == 'openai/gpt-4'


def test_judge_prompt_templates_appear_only_once():
    """The verbatim FastChat template identifiers ('single-v1',
    'single-v1-multi-turn') previously appeared in BOTH source_metadata
    and metric_config additional_details. They should now appear in only
    one place — metric_config.additional_details — to avoid the same
    'scattered duplicates' issue akornilo flagged for judge_model.
    """
    bundles = adapter.make_logs(
        sample_rows(), retrieved_timestamp='1234567890.0'
    )
    for log, _, _ in bundles:
        meta_extras = log.source_metadata.additional_details or {}
        assert 'judge_prompt_templates_json' not in meta_extras
        for result in log.evaluation_results:
            metric_extras = result.metric_config.additional_details or {}
            assert 'judge_prompt_templates_json' in metric_extras
