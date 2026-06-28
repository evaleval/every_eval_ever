from __future__ import annotations

import json

from every_eval_ever.eval_types import EvaluationLog
from utils.tau_bench import adapter


def sample_records() -> list[adapter.TauBenchSubmission]:
    return [
        adapter.TauBenchSubmission(
            submission_id='gpt-5-5_sierra_2026-05-05',
            manifest_section='submissions',
            source_url=adapter.submission_source_url(
                'gpt-5-5_sierra_2026-05-05'
            ),
            submission={
                'model_name': 'GPT-5.5',
                'model_organization': 'OpenAI',
                'submitting_organization': 'Sierra',
                'submission_date': '2026-05-05',
                'submission_type': 'standard',
                'modality': 'text',
                'contact_info': {
                    'email': 'research@example.com',
                    'name': 'Sierra Research Team',
                },
                'is_new': True,
                'trajectories_available': True,
                'trajectory_files': {
                    'banking_knowledge': (
                        'gpt-5.5_xhigh_banking_knowledge_gpt-5.2_4trials.json'
                    )
                },
                'references': [],
                'results': {
                    'airline': None,
                    'retail': None,
                    'telecom': None,
                    'banking_knowledge': {
                        'pass_1': 37.37,
                        'pass_2': 27.84,
                        'pass_3': None,
                        'pass_4': None,
                        'cost': 1.988,
                        'retrieval_config': 'alltools',
                    },
                },
                'reasoning_effort': 'xhigh',
                'methodology': {
                    'evaluation_date': '2026-05-06',
                    'tau2_bench_version': '0.2.1-dev',
                    'user_simulator': 'gpt-5.2',
                    'notes': 'AllTools retrieval, 4 trials.',
                    'verification': {
                        'modified_prompts': False,
                        'omitted_questions': False,
                    },
                },
                'model_release': {'release_date': '2026-04-22'},
            },
        ),
        adapter.TauBenchSubmission(
            submission_id='gpt-realtime-1-0_openai_2026-04-13',
            manifest_section='voice_submissions',
            source_url=adapter.submission_source_url(
                'gpt-realtime-1-0_openai_2026-04-13'
            ),
            submission={
                'model_name': 'GPT Realtime 1.0',
                'model_organization': 'OpenAI',
                'submitting_organization': 'OpenAI',
                'submission_date': '2026-04-13',
                'submission_type': 'standard',
                'modality': 'voice',
                'contact_info': {'email': 'research@example.com'},
                'results': {
                    'retail': {'pass_1': 55.5},
                    'airline': None,
                    'telecom': None,
                    'banking_knowledge': None,
                },
                'methodology': {
                    'evaluation_date': '2026-04-13',
                    'tau2_bench_version': '0.2.1-dev',
                    'user_simulator': 'voice-user-sim-v1',
                },
                'voice_config': {
                    'provider': 'openai',
                    'model': 'gpt-realtime-1.0',
                    'tick_duration_seconds': 1.0,
                    'max_steps_seconds': 900.0,
                    'user_tts_provider': 'elevenlabs/eleven_v3',
                    'pipeline': {'asr': 'deepgram', 'tts': 'elevenlabs'},
                },
            },
        ),
    ]


def test_make_logs_validate_against_schema():
    bundles = adapter.make_logs(
        sample_records(), retrieved_timestamp='1234567890.0'
    )

    assert len(bundles) == 2
    for bundle in bundles:
        validated = EvaluationLog.model_validate(bundle.log.model_dump())
        assert validated.schema_version == '0.2.2'
        assert validated.source_metadata.source_name == 'tau-bench Leaderboard'
        assert validated.source_metadata.source_type.value == 'documentation'
        assert validated.eval_library.name == 'tau2-bench'


def test_text_submission_maps_domain_metrics_and_cost():
    bundles = adapter.make_logs(
        sample_records(), retrieved_timestamp='1234567890.0'
    )
    text = next(
        bundle.log
        for bundle in bundles
        if bundle.log.model_info.id == 'openai/gpt-5.5'
    )

    assert text.evaluation_timestamp == '2026-05-06'
    assert text.source_metadata.evaluator_relationship.value == 'third_party'
    assert text.model_info.additional_details['reasoning_effort'] == 'xhigh'

    by_id = {
        result.evaluation_result_id: result
        for result in text.evaluation_results
    }
    assert set(by_id) == {
        'tau_bench:gpt-5-5_sierra_2026-05-05:banking_knowledge:pass_1',
        'tau_bench:gpt-5-5_sierra_2026-05-05:banking_knowledge:pass_2',
        'tau_bench:gpt-5-5_sierra_2026-05-05:banking_knowledge:cost',
    }

    pass_1 = by_id[
        'tau_bench:gpt-5-5_sierra_2026-05-05:banking_knowledge:pass_1'
    ]
    assert pass_1.evaluation_name == ('tau_bench.text.banking_knowledge.pass_1')
    assert pass_1.metric_config.metric_id == 'tau_bench.pass_at_k'
    assert pass_1.metric_config.metric_parameters == {'k': 1}
    assert pass_1.metric_config.metric_unit == 'percent'
    assert pass_1.metric_config.min_score == 0
    assert pass_1.metric_config.max_score == 100
    assert pass_1.score_details.score == 37.37
    assert (
        pass_1.source_data.additional_details['retrieval_config'] == 'alltools'
    )
    assert (
        pass_1.generation_config.additional_details['user_simulator']
        == 'gpt-5.2'
    )

    cost = by_id['tau_bench:gpt-5-5_sierra_2026-05-05:banking_knowledge:cost']
    assert cost.metric_config.lower_is_better is True
    assert cost.metric_config.metric_unit == 'usd_per_trajectory'
    assert cost.metric_config.score_type is None
    assert cost.score_details.score == 1.988


def test_voice_submission_preserves_voice_metadata():
    bundles = adapter.make_logs(
        sample_records(), retrieved_timestamp='1234567890.0'
    )
    voice = next(
        bundle.log
        for bundle in bundles
        if bundle.log.model_info.id == 'openai/gpt-realtime-1.0'
    )

    assert voice.source_metadata.evaluator_relationship.value == 'first_party'
    result = voice.evaluation_results[0]
    assert result.evaluation_name == 'tau_bench.voice.retail.pass_1'
    assert result.score_details.score == 55.5
    assert (
        result.generation_config.additional_details['voice_provider']
        == 'openai'
    )
    assert (
        result.generation_config.additional_details['voice_model']
        == 'gpt-realtime-1.0'
    )
    assert json.loads(
        result.generation_config.additional_details['voice_pipeline']
    ) == {'asr': 'deepgram', 'tts': 'elevenlabs'}


def test_load_submissions_from_local_manifest(tmp_path):
    root = tmp_path / 'submissions'
    root.mkdir()
    manifest = {
        'submissions': ['gpt-5-5_sierra_2026-05-05'],
        'voice_submissions': ['gpt-realtime-1-0_openai_2026-04-13'],
        'legacy_submissions': ['ignored-legacy'],
    }
    (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')

    for record in sample_records():
        submission_dir = root / record.submission_id
        submission_dir.mkdir()
        (submission_dir / 'submission.json').write_text(
            json.dumps(record.submission),
            encoding='utf-8',
        )

    records = adapter.load_submissions_from_dir(
        root, sections=['submissions', 'voice_submissions']
    )
    assert [record.submission_id for record in records] == [
        'gpt-5-5_sierra_2026-05-05',
        'gpt-realtime-1-0_openai_2026-04-13',
    ]


def test_non_numeric_score_fails_with_context():
    record = sample_records()[0]
    record.submission['results']['banking_knowledge']['pass_1'] = 'not-a-score'

    try:
        adapter.make_logs([record], retrieved_timestamp='1234567890.0')
    except ValueError as exc:
        assert 'gpt-5-5_sierra_2026-05-05/banking_knowledge/pass_1' in str(exc)
    else:
        raise AssertionError('expected non-numeric score to fail')
