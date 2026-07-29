from utils.rewardbench.adapter import (
    convert_rewardbench_v1_rows,
    extract_hf_model_id_from_html,
)


def test_random_baseline_is_recorded_as_non_failing_exclusion(tmp_path):
    random_row = {
        'Model': 'random',
        'Score': '50',
    }
    result = convert_rewardbench_v1_rows(
        [random_row],
        retrieved_timestamp='1234',
        output_dir=tmp_path / 'data' / 'reward-bench',
    )

    assert result.records == []
    assert result.failures == []
    assert len(result.exclusions) == 1
    assert result.exclusions[0].source_record == random_row
    result.raise_if_incomplete()


def test_invalid_metric_does_not_discard_other_model_metrics(tmp_path):
    row = {
        'Model': (
            '<a href="https://huggingface.co/example/model-a">'
            'Model A</a>'
        ),
        'Score': '75',
        'Chat': 'not-a-score',
    }
    result = convert_rewardbench_v1_rows(
        [row],
        retrieved_timestamp='1234',
        output_dir=tmp_path / 'data' / 'reward-bench',
    )

    assert len(result.records) == 1
    assert [
        metric.evaluation_name
        for metric in result.records[0].eval_log.evaluation_results
    ] == ['Score']
    assert len(result.failures) == 1
    assert result.failures[0].source_record == {
        'model': row['Model'],
        'metric': 'Chat',
        'value': 'not-a-score',
    }
    assert result.records[0].developer == 'example'
    assert result.records[0].model_name == 'model-a'


def test_external_model_link_is_not_treated_as_hugging_face_id(tmp_path):
    row = {
        'Model': '<a href="https://openai.com/gpt-4">GPT-4</a>',
        'Score': '75',
    }

    result = convert_rewardbench_v1_rows(
        [row],
        retrieved_timestamp='1234',
        output_dir=tmp_path / 'data' / 'reward-bench',
    )

    assert result.failures == []
    assert result.records[0].developer == 'openai'
    assert result.records[0].model_name == 'GPT-4'
    assert result.records[0].eval_log.model_info.id == 'openai/GPT-4'


def test_closed_model_identifier_falls_back_to_display_name(tmp_path):
    row = {
        'Model': (
            '<a href="https://www.anthropic.com/claude">'
            'Anthropic/claude-3-opus</a>'
        ),
        'Score': '75',
    }

    result = convert_rewardbench_v1_rows(
        [row],
        retrieved_timestamp='1234',
        output_dir=tmp_path / 'data' / 'reward-bench',
    )

    assert result.failures == []
    assert result.records[0].developer == 'Anthropic'
    assert result.records[0].model_name == 'claude-3-opus'
    assert (
        result.records[0].eval_log.model_info.id
        == 'Anthropic/claude-3-opus'
    )


def test_hugging_face_id_extraction_rejects_external_and_lookalike_hosts():
    assert (
        extract_hf_model_id_from_html(
            '<a href="https://huggingface.co/example/model">Model</a>'
        )
        == 'example/model'
    )
    assert (
        extract_hf_model_id_from_html(
            '<a href="/example/model">Model</a>'
        )
        == 'example/model'
    )
    assert (
        extract_hf_model_id_from_html(
            '<a href="https://anthropic.com/claude">Claude</a>'
        )
        is None
    )
    assert (
        extract_hf_model_id_from_html(
            '<a href="https://huggingface.co.example.com/org/model">'
            'Model</a>'
        )
        is None
    )
