"""Unit tests for the LEXam adapter."""

from pathlib import Path

import pytest

from every_eval_ever.converters.lexam.adapter import (
    LEXamAdapter,
    _clean_model_name,
    _extract_section_rows,
    _model_identity,
)
from every_eval_ever.eval_types import EvaluationLog

FIXTURE_HTML = (
    Path(__file__).parent / 'data' / 'lexam' / 'leaderboard.html'
).read_text(encoding='utf-8')


def test_clean_model_name_strips_medals() -> None:
    assert _clean_model_name('GPT-5🥇') == 'GPT-5'
    assert _clean_model_name('Claude-3.7-Sonnet🥉') == 'Claude-3.7-Sonnet'


def test_extract_section_rows_open_questions() -> None:
    rows = _extract_section_rows(
        FIXTURE_HTML,
        'Leaderboard on LEXam – Open Questions',
    )
    assert len(rows) == 3
    assert rows[0].model_name == 'GPT-5'
    assert rows[0].score == 70.20


def test_extract_section_rows_mcq() -> None:
    rows = _extract_section_rows(
        FIXTURE_HTML,
        'Leaderboard on LEXam – Multiple-Choice Questions',
    )
    assert len(rows) == 3
    assert rows[2].model_name == 'Phi-4'
    assert rows[2].score == 25.0


def test_extract_section_rows_missing_section_raises() -> None:
    with pytest.raises(ValueError, match='Leaderboard section not found'):
        _extract_section_rows(FIXTURE_HTML, 'Missing Section')


def test_fetch_leaderboard_combines_metrics_per_model() -> None:
    logs = LEXamAdapter().fetch_leaderboard(html=FIXTURE_HTML)
    by_name = {log.model_info.name: log for log in logs}

    assert len(logs) == 4
    assert len(by_name['GPT-5'].evaluation_results) == 2
    assert len(by_name['Gemini-3-Pro-preview'].evaluation_results) == 1
    assert len(by_name['Phi-4'].evaluation_results) == 1


def test_fetch_leaderboard_open_question_score() -> None:
    logs = LEXamAdapter().fetch_leaderboard(html=FIXTURE_HTML)
    gpt5 = next(log for log in logs if log.model_info.name == 'GPT-5')
    results = {r.evaluation_name: r for r in gpt5.evaluation_results}

    assert results['Open Question Judge Score'].score_details.score == 70.20
    assert results['Multiple-Choice Accuracy'].score_details.score == 62.65


def test_fetch_leaderboard_source_metadata_is_documentation() -> None:
    logs = LEXamAdapter().fetch_leaderboard(html=FIXTURE_HTML)
    assert logs[0].source_metadata.source_type.value == 'documentation'
    assert logs[0].source_metadata.source_name == 'LEXam Leaderboard'


def test_fetch_leaderboard_uses_hf_dataset_source() -> None:
    logs = LEXamAdapter().fetch_leaderboard(html=FIXTURE_HTML)
    gpt5 = next(log for log in logs if log.model_info.name == 'GPT-5')
    open_result = next(
        r
        for r in gpt5.evaluation_results
        if r.evaluation_name == 'Open Question Judge Score'
    )
    mcq_result = next(
        r
        for r in gpt5.evaluation_results
        if r.evaluation_name == 'Multiple-Choice Accuracy'
    )

    assert open_result.source_data.hf_repo == 'LEXam-Benchmark/LEXam'
    assert open_result.source_data.hf_split == 'test'
    assert open_result.source_data.samples_number == 2541
    assert mcq_result.source_data.samples_number == 4696


def test_fetch_leaderboard_metric_ids() -> None:
    logs = LEXamAdapter().fetch_leaderboard(html=FIXTURE_HTML)
    gpt5 = next(log for log in logs if log.model_info.name == 'GPT-5')
    by_name = {r.evaluation_name: r for r in gpt5.evaluation_results}

    assert (
        by_name['Open Question Judge Score'].metric_config.metric_id
        == 'lexam.open_question_judge_score'
    )
    assert (
        by_name['Multiple-Choice Accuracy'].metric_config.metric_id
        == 'lexam.mcq_accuracy'
    )


def test_fetch_leaderboard_open_metric_has_llm_scoring() -> None:
    logs = LEXamAdapter().fetch_leaderboard(html=FIXTURE_HTML)
    gpt5 = next(log for log in logs if log.model_info.name == 'GPT-5')
    open_result = next(
        r
        for r in gpt5.evaluation_results
        if r.evaluation_name == 'Open Question Judge Score'
    )

    llm_scoring = open_result.metric_config.llm_scoring
    assert llm_scoring is not None
    assert len(llm_scoring.judges) == 3


def test_fetch_leaderboard_model_developer_inference() -> None:
    logs = LEXamAdapter().fetch_leaderboard(html=FIXTURE_HTML)
    gpt5 = next(log for log in logs if log.model_info.name == 'GPT-5')

    assert gpt5.model_info.developer == 'openai'
    assert gpt5.model_info.id == 'openai/GPT-5'


def test_unknown_model_identity_raises() -> None:
    with pytest.raises(ValueError, match='No model identity mapping'):
        _model_identity('New-Unmapped-Model')


def test_fetch_leaderboard_output_validates_as_evaluation_log() -> None:
    logs = LEXamAdapter().fetch_leaderboard(html=FIXTURE_HTML)
    for log in logs:
        validated = EvaluationLog.model_validate(
            log.model_dump(mode='json', exclude_none=True)
        )
        assert validated.schema_version == log.schema_version
