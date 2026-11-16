from __future__ import annotations

import hashlib
import json

import pytest

from every_eval_ever.adapters.tau_bench import adapter
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.helpers import FetchError


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
                'interaction_metrics': {
                    'version': '1.0',
                    'config': {
                        'tick_duration_sec': 1.0,
                        'no_yield_window_sec': 2.0,
                    },
                    'domains': {
                        'retail': {
                            'response_latency_mean': 1.23,
                            'yield_latency_mean': 0.45,
                            'response_rate': 0.98,
                            'yield_rate': 0.91,
                            'agent_interruption_rate': 0.04,
                            'selectivity_backchannel': 0.87,
                            'selectivity_vocal_tic': 0.79,
                            'selectivity_non_directed': 0.82,
                            'counts': {
                                'n_simulations': 50,
                                'response_total': 300,
                            },
                        },
                    },
                    'overall': {
                        'response_latency_mean': 1.23,
                        'response_rate': 0.98,
                        'agent_interruption_rate': 0.04,
                        'counts': {'n_simulations': 50},
                    },
                },
            },
        ),
    ]


def test_make_logs_validate_against_schema():
    bundles = adapter.make_logs(
        sample_records(), retrieved_timestamp='1234567890.0'
    )

    for bundle in bundles:
        validated = EvaluationLog.model_validate(bundle.log.model_dump())
        assert validated.schema_version == adapter.SCHEMA_VERSION
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
    assert pass_1.metric_config.metric_id == 'tau_bench.pass_hat_k'
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


def test_voice_interaction_metrics_are_emitted_with_units_and_direction():
    """Voice submissions carry a per-domain interaction panel that must not be
    dropped, and each metric needs its own unit, scale, and direction."""
    bundles = adapter.make_logs(
        sample_records(), retrieved_timestamp='1234567890.0'
    )
    voice = next(
        bundle.log
        for bundle in bundles
        if bundle.log.model_info.id == 'openai/gpt-realtime-1.0'
    )
    prefix = 'tau_bench:gpt-realtime-1-0_openai_2026-04-13'
    by_id = {
        result.evaluation_result_id: result
        for result in voice.evaluation_results
    }

    latency = by_id[f'{prefix}:retail:response_latency_mean']
    assert (
        latency.evaluation_name
        == 'tau_bench.voice.retail.response_latency_mean'
    )
    assert (
        latency.metric_config.metric_id
        == 'tau_bench.interaction.response_latency_mean'
    )
    assert latency.metric_config.metric_unit == 'seconds'
    assert latency.metric_config.lower_is_better is True
    assert latency.metric_config.max_score == float('inf')
    assert latency.score_details.score == 1.23
    assert (
        json.loads(latency.score_details.details['counts'])['n_simulations']
        == 50
    )
    assert (
        latency.metric_config.additional_details[
            'interaction_metrics_version'
        ]
        == '1.0'
    )
    assert (
        'interaction_metrics_config'
        in latency.metric_config.additional_details
    )

    response_rate = by_id[f'{prefix}:retail:response_rate']
    assert response_rate.metric_config.metric_unit == 'proportion'
    assert response_rate.metric_config.lower_is_better is False
    assert response_rate.metric_config.max_score == 1.0
    assert response_rate.score_details.score == 0.98

    interruption = by_id[f'{prefix}:retail:agent_interruption_rate']
    assert interruption.metric_config.lower_is_better is True

    # The overall aggregate panel is kept under a synthetic `overall` domain.
    assert f'{prefix}:overall:response_rate' in by_id


def test_out_of_range_interaction_rate_is_rejected():
    """A proportion the source could never report is invalid data."""
    record = sample_records()[1]
    record.submission['interaction_metrics']['domains']['retail'][
        'response_rate'
    ] = 1.5
    with pytest.raises(ValueError) as caught:
        adapter.make_logs([record], retrieved_timestamp='1234567890.0')
    assert 'retail/response_rate' in str(caught.value)


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


def test_pass_metric_names_pass_hat_k_and_says_so():
    """Pass^k is not pass@k, and the record has to distinguish them."""
    bundles = adapter.make_logs(
        sample_records(), retrieved_timestamp='1234567890.0'
    )
    pass_2 = next(
        result
        for bundle in bundles
        for result in bundle.log.evaluation_results
        if result.metric_config.metric_parameters == {'k': 2}
    )

    assert pass_2.metric_config.metric_id == 'tau_bench.pass_hat_k'
    assert pass_2.metric_config.metric_name == 'Pass^2'
    semantics = pass_2.metric_config.additional_details['metric_semantics']
    assert 'all 2 trials succeed' in semantics
    assert 'Not pass@k' in semantics


def test_one_provider_spelled_two_ways_gets_one_developer():
    """`Z.ai` and `Zhipu AI` are the same provider on this leaderboard."""
    assert adapter.organization_slug('Z.ai') == 'zhipu-ai'
    assert adapter.organization_slug('Zhipu AI') == 'zhipu-ai'
    assert adapter.organization_slug('z.AI ') == 'zhipu-ai'
    # A provider that is genuinely new still gets a slug of its own.
    assert adapter.organization_slug('Some New Lab') == 'some-new-lab'


def test_replay_inputs_cannot_smuggle_impossible_scores():
    """--input-dir and --base-url accept what the leaderboard never serves."""
    for value in (101.0, -0.5, True, float('inf'), 'nan'):
        record = sample_records()[0]
        record.submission['results']['banking_knowledge']['pass_1'] = value
        with pytest.raises(ValueError) as caught:
            adapter.make_logs([record], retrieved_timestamp='1234567890.0')
        assert 'banking_knowledge/pass_1' in str(caught.value)

    record = sample_records()[0]
    record.submission['results']['banking_knowledge']['cost'] = -1.0
    with pytest.raises(ValueError) as caught:
        adapter.make_logs([record], retrieved_timestamp='1234567890.0')
    assert 'banking_knowledge/cost' in str(caught.value)


def test_a_malformed_methodology_block_does_not_crash_the_submission():
    """Older and hand-built submissions carry strings where objects belong."""
    record = sample_records()[0]
    record.submission['methodology'] = 'AllTools retrieval, 4 trials.'
    record.submission['voice_config'] = 'none'
    record.submission['model_release'] = 'unreleased'
    record.submission['trajectory_files'] = 'see attachment'
    record.submission['references'] = 'https://example.com/paper'

    bundles = adapter.make_logs([record], retrieved_timestamp='1234567890.0')

    log = bundles[0].log
    assert log.eval_library.version == 'unknown'
    # `submission_date` is the fallback once methodology carries no date.
    assert log.evaluation_timestamp == '2026-05-05'


def test_remote_run_pins_the_ref_and_records_the_bytes(monkeypatch):
    """A record must name the exact input that produced its scores."""
    sha = 'a' * 40
    fetched: list[str] = []

    def fake_fetch_json(url, *args, **kwargs):
        fetched.append(url)
        if url.startswith('https://api.github.com/'):
            return {'sha': sha}
        if url.endswith('manifest.json'):
            return {'submissions': ['gpt-5-5_sierra_2026-05-05']}
        return sample_records()[0].submission

    monkeypatch.setattr(adapter, 'fetch_json', fake_fetch_json)

    records = adapter.load_submissions_from_url(
        adapter.RAW_SUBMISSIONS_BASE_URL, sections=['submissions']
    )

    assert fetched[0] == (
        'https://api.github.com/repos/sierra-research/tau2-bench/commits/main'
    )
    assert all('/main/' not in url for url in fetched[1:])
    assert records[0].source_commit == sha
    assert sha in records[0].source_url
    assert records[0].content_sha256 == adapter._canonical_sha256(
        records[0].submission
    )

    log = adapter.make_logs(records, retrieved_timestamp='1.0')[0].log
    details = log.evaluation_results[0].source_data.additional_details
    assert details['submission_commit'] == sha
    assert details['submission_sha256'] == records[0].content_sha256


def test_an_unresolvable_ref_stops_the_run_unless_it_is_allowed(monkeypatch):
    def failing_fetch_json(url, *args, **kwargs):
        if url.startswith('https://api.github.com/'):
            raise FetchError('rate limited')
        if url.endswith('manifest.json'):
            return {'submissions': []}
        raise AssertionError(f'unexpected fetch: {url}')

    monkeypatch.setattr(adapter, 'fetch_json', failing_fetch_json)

    with pytest.raises(FetchError):
        adapter.load_submissions_from_url(
            adapter.RAW_SUBMISSIONS_BASE_URL, sections=['submissions']
        )

    assert (
        adapter.load_submissions_from_url(
            adapter.RAW_SUBMISSIONS_BASE_URL,
            sections=['submissions'],
            allow_unpinned_source=True,
        )
        == []
    )


def test_local_replay_does_not_claim_an_upstream_url(tmp_path):
    root = tmp_path / 'submissions'
    root.mkdir()
    record = sample_records()[0]
    (root / 'manifest.json').write_text(
        json.dumps({'submissions': [record.submission_id]}), encoding='utf-8'
    )
    submission_dir = root / record.submission_id
    submission_dir.mkdir()
    payload = json.dumps(record.submission)
    (submission_dir / 'submission.json').write_text(payload, encoding='utf-8')

    loaded = adapter.load_submissions_from_dir(root, sections=['submissions'])

    assert loaded[0].source_url is None
    assert loaded[0].local_path == str(submission_dir / 'submission.json')
    assert loaded[0].content_sha256 == hashlib.sha256(
        payload.encode('utf-8')
    ).hexdigest()

    log = adapter.make_logs(loaded, retrieved_timestamp='1.0')[0].log
    source_data = log.evaluation_results[0].source_data
    assert source_data.url == [adapter.LEADERBOARD_URL]
    assert (
        source_data.additional_details['local_input_path']
        == loaded[0].local_path
    )


def test_limit_bounds_the_download_not_just_the_output(monkeypatch):
    submission_ids = [f'model-{index}' for index in range(5)]
    fetched: list[str] = []

    def fake_fetch_json(url, *args, **kwargs):
        fetched.append(url)
        if url.startswith('https://api.github.com/'):
            return {'sha': 'b' * 40}
        if url.endswith('manifest.json'):
            return {'submissions': submission_ids}
        return sample_records()[0].submission

    monkeypatch.setattr(adapter, 'fetch_json', fake_fetch_json)

    records = adapter.load_submissions_from_url(
        adapter.RAW_SUBMISSIONS_BASE_URL, sections=['submissions'], limit=2
    )

    assert len(records) == 2
    submission_fetches = [
        url for url in fetched if url.endswith('submission.json')
    ]
    assert len(submission_fetches) == 2


def test_conversion_accounts_for_every_submission_it_does_not_publish():
    scored, empty = sample_records()
    empty.submission['results'] = {domain: None for domain in adapter.DOMAINS}
    # A row with nothing to publish also has no interaction panel.
    empty.submission.pop('interaction_metrics', None)
    broken = sample_records()[0]
    broken.submission.pop('model_name')

    result = adapter.convert_logs(
        [scored, empty, broken], retrieved_timestamp='1.0'
    )

    assert result.total_records == 3
    assert len(result.records) == 1
    assert [exclusion.source_ref for exclusion in result.exclusions] == [
        empty.submission_id
    ]
    assert [failure.source_ref for failure in result.failures] == [
        broken.submission_id
    ]
    assert 'model_name' in result.failures[0].reason
    with pytest.raises(Exception):
        result.raise_if_incomplete()


def test_export_writes_one_datastore_path_per_submission(tmp_path):
    output_dir = tmp_path / 'data' / 'tau-bench'
    bundles = adapter.make_logs(
        sample_records(), retrieved_timestamp='1234567890.0'
    )

    paths = adapter.export_logs(bundles, output_dir)

    assert len(paths) == len(bundles)
    for path in paths:
        assert path.parent.parent.parent == output_dir
        assert path.suffix == '.json'
        EvaluationLog.model_validate(json.loads(path.read_text()))


def test_a_failing_record_leaves_no_partial_publication(tmp_path):
    output_dir = tmp_path / 'data' / 'tau-bench'
    bundles = adapter.make_logs(
        sample_records(), retrieved_timestamp='1234567890.0'
    )
    broken = bundles[-1]
    object.__setattr__(broken, 'developer', '')

    with pytest.raises(ValueError):
        adapter.export_logs(bundles, output_dir)

    assert not output_dir.exists()


def test_pin_base_url_falls_back_to_git_ls_remote_on_403(monkeypatch):
    import requests

    sha = 'c' * 40

    def rate_limited_fetch(url, *args, **kwargs):
        resp = requests.Response()
        resp.status_code = 403
        http_err = requests.HTTPError('403 Forbidden', response=resp)
        raise FetchError('GitHub rate limit') from http_err

    monkeypatch.setattr(adapter, 'fetch_json', rate_limited_fetch)
    monkeypatch.setattr(
        adapter, '_git_ls_remote_commit', lambda owner, repo, ref: sha
    )

    pinned_url, resolved = adapter.pin_base_url(adapter.RAW_SUBMISSIONS_BASE_URL)
    assert resolved == sha
    assert sha in pinned_url


def test_git_ls_remote_prefers_peeled_tag_and_emit_source_version(
    monkeypatch, capsys
):
    tag_obj = '1' * 40
    peeled_commit = '2' * 40

    def fake_run(cmd, **kwargs):
        out = f'{tag_obj}\trefs/tags/v1.0\n{peeled_commit}\trefs/tags/v1.0^{{}}\n'
        return type('Proc', (), {'returncode': 0, 'stdout': out})()

    monkeypatch.setattr(adapter.subprocess, 'run', fake_run)
    assert (
        adapter._git_ls_remote_commit('sierra-research', 'tau2-bench', 'v1.0')
        == peeled_commit
    )

    monkeypatch.setattr(
        adapter,
        '_probe_submissions_commit',
        lambda owner, repo, ref, path: peeled_commit,
    )
    assert adapter.run(adapter.parse_args(['--emit-source-version'])) == 0
    assert capsys.readouterr().out.strip() == peeled_commit
