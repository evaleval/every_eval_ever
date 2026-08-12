from __future__ import annotations

import json
from pathlib import Path

import pytest

from every_eval_ever.adapters.mercor_eval import adapter
from every_eval_ever.eval_types import EvaluationLog
from every_eval_ever.validate import validate_file

FIXTURE_PATH = (
    Path(__file__).parent / 'data' / 'mercor_eval' / 'api_payload.json'
)


def fixture_payload() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding='utf-8'))


def make_fixture_bundles():
    return adapter.make_bundles(
        fixture_payload(),
        retrieved_timestamp='1234567890.0',
        base_url=adapter.DEFAULT_BASE_URL,
    )


def test_make_bundles_maps_all_metrics_and_validates():
    bundles = make_fixture_bundles()

    assert len(bundles) == 1
    log = bundles[0].log
    EvaluationLog.model_validate(log.model_dump())
    assert log.model_info.id == 'google/gemini-3-flash-preview'
    assert log.model_info.additional_details['mercor_model_id'] == 'model_test'
    assert log.evaluation_timestamp == '2026-01-08T04:10:41Z'
    assert len(log.evaluation_results) == 6

    def find_result(
        evaluation_name: str,
        metric_id: str,
        metric_parameters: dict | None = None,
    ):
        return next(
            result
            for result in log.evaluation_results
            if result.evaluation_name == evaluation_name
            and result.metric_config.metric_id == metric_id
            and result.metric_config.metric_parameters == metric_parameters
        )

    pass_at_1 = find_result('Overall', 'pass_at_k', {'k': 1})
    assert pass_at_1.score_details.score == 0.24
    assert pass_at_1.score_details.uncertainty.num_samples == 480
    assert (
        pass_at_1.score_details.uncertainty.confidence_interval.lower == 0.208
    )
    pass_at_8 = find_result('Overall', 'pass_at_k', {'k': 8})
    assert pass_at_8.score_details.score == 0.367
    assert (
        find_result(
            'Overall',
            'pass_hat_k',
            {'k': 8, 'estimator': 'naive'},
        ).score_details.score
        == 0.134
    )
    assert find_result('Overall', 'mean_score').score_details.score == 0.422
    assert (
        find_result(
            'Investment Banking',
            'pass_at_k',
            {'k': 1},
        ).score_details.score
        == 0.267
    )
    assert (
        find_result('Law', 'pass_at_k', {'k': 1}).score_details.score == 0.259
    )


def test_make_bundles_preserves_run_configuration():
    bundle = make_fixture_bundles()[0]
    result = bundle.log.evaluation_results[0]
    args = result.generation_config.generation_args

    assert args.temperature == 1.0
    assert args.max_tokens == 65536
    assert args.eval_limits.time_limit == 10800
    assert args.eval_limits.message_limit == 250
    assert (
        args.agentic_eval_config.additional_details['agent_name']
        == 'ReAct Toolbelt Agent'
    )
    assert (
        args.agentic_eval_config.additional_details['agent_config_id']
        == 'react_toolbelt_agent'
    )
    assert (
        result.generation_config.additional_details['reasoning_effort']
        == 'high'
    )
    assert result.generation_config.additional_details['verbosity'] == 'medium'


def test_source_and_evaluation_metadata_are_preserved():
    log = make_fixture_bundles()[0].log

    assert log.evaluation_id == (
        'apex-agents/google_gemini-3-flash-preview/1234567890.0'
    )
    assert log.source_metadata.source_type.value == 'evaluation_run'
    assert log.source_metadata.source_name == 'Mercor APEX-Agents Leaderboard'
    assert log.source_metadata.source_organization_name == 'Mercor'
    assert (
        log.source_metadata.source_organization_url == 'https://www.mercor.com'
    )
    assert log.source_metadata.evaluator_relationship.value == 'first_party'
    assert (
        log.source_metadata.additional_details['data_as_of']
        == '2026-06-30T20:32:54Z'
    )
    assert log.eval_library.name == 'Mercor Evaluation Exports API'
    assert log.eval_library.version == '1.0'

    source_data = log.evaluation_results[0].source_data
    assert source_data.dataset_name == 'apex-agents'
    assert source_data.source_type == 'hf_dataset'
    assert source_data.hf_repo == 'mercor/apex-agents'
    assert source_data.additional_details['benchmark_id'] == 'camp_test'
    assert source_data.additional_details['num_tasks'] == '480'
    assert source_data.additional_details['api_url'] == (
        f'{adapter.DEFAULT_BASE_URL}/leaderboards?benchmarkId=camp_test'
    )


def test_provider_namespace_falls_back_to_sanitized_provider():
    assert adapter.canonical_developer('Mystery Provider') == 'mystery-provider'
    assert adapter.canonical_developer('gemini') == 'google'
    assert adapter.canonical_developer('moonshotai') == 'moonshot'


def test_namespaced_model_uses_model_developer_not_inference_provider():
    payload = fixture_payload()
    model = payload['leaderboards']['rows'][0]['model']
    model['name'] = 'openai/gpt-oss-120b'
    model['config']['model'] = 'openai/gpt-oss-120b'
    model['config']['provider'] = 'baseten'

    bundle = adapter.make_bundles(
        payload,
        retrieved_timestamp='123',
    )[0]

    assert bundle.log.model_info.name == 'openai/gpt-oss-120b'
    assert bundle.log.model_info.id == 'openai/gpt-oss-120b'
    assert bundle.log.model_info.developer == 'openai'
    assert bundle.log.model_info.inference_platform == 'baseten'
    assert bundle.developer == 'openai'
    assert bundle.model == 'gpt-oss-120b'
    assert bundle.benchmark_slug == 'apex-agents'


def test_namespaced_moonshot_model_uses_datastore_slug():
    payload = fixture_payload()
    model = payload['leaderboards']['rows'][0]['model']
    model['name'] = 'moonshotai/Kimi-K2-Thinking'
    model['config']['model'] = 'moonshotai/Kimi-K2-Thinking'
    model['config']['provider'] = 'baseten'

    bundle = adapter.make_bundles(
        payload,
        retrieved_timestamp='123',
    )[0]

    assert bundle.log.model_info.name == 'moonshotai/Kimi-K2-Thinking'
    assert bundle.log.model_info.id == 'moonshot/Kimi-K2-Thinking'
    assert bundle.log.model_info.developer == 'moonshot'
    assert bundle.log.model_info.inference_platform == 'baseten'
    assert bundle.developer == 'moonshot'
    assert bundle.model == 'kimi-k2-thinking'


def test_unknown_benchmark_reference_fails():
    payload = fixture_payload()
    payload['leaderboards']['rows'][0]['benchmark']['id'] = 'camp_missing'

    with pytest.raises(ValueError, match='camp_missing'):
        adapter.make_bundles(payload, retrieved_timestamp='123')


def test_unsupported_schema_version_fails():
    payload = fixture_payload()
    payload['benchmarks']['schemaVersion'] = '2.0'

    with pytest.raises(ValueError, match='schemaVersion'):
        adapter.make_bundles(payload, retrieved_timestamp='123')


def test_empty_leaderboard_fails():
    payload = fixture_payload()
    payload['leaderboards']['rows'] = []
    payload['leaderboards']['total'] = 0

    with pytest.raises(ValueError, match='leaderboard'):
        adapter.make_bundles(payload, retrieved_timestamp='123')


def test_load_payload_requires_an_object(tmp_path: Path):
    payload_path = tmp_path / 'payload.json'
    payload_path.write_text('[]', encoding='utf-8')

    with pytest.raises(ValueError, match='JSON object'):
        adapter.load_payload(payload_path)


def test_export_paths_validate(tmp_path: Path):
    output_dir = tmp_path / 'data'
    paths = adapter.export_bundles(make_fixture_bundles(), output_dir)

    assert len(paths) == 1
    assert paths[0].parent.parent.parent.parent == output_dir
    assert paths[0].parent == (
        output_dir / 'apex-agents' / 'google' / 'gemini-3-flash-preview'
    )
    assert validate_file(paths[0]).valid


def test_fetch_payload_authenticates_and_paginates():
    calls = []
    responses = [
        {
            'schemaVersion': '1.0',
            'benchmarks': fixture_payload()['benchmarks']['benchmarks'],
            'dataAsOf': '2026-06-30T20:32:54Z',
        },
        {
            'schemaVersion': '1.0',
            'rows': [{'evaluationId': 'first'}],
            'total': 2,
            'limit': 1,
            'offset': 0,
            'dataAsOf': '2026-06-30T20:32:54Z',
        },
        {
            'schemaVersion': '1.0',
            'rows': [{'evaluationId': 'second'}],
            'total': 2,
            'limit': 1,
            'offset': 1,
            'dataAsOf': '2026-06-30T20:32:54Z',
        },
    ]

    def fake_fetch(url, *, headers, params=None, timeout):
        calls.append((url, headers, params, timeout))
        return responses.pop(0)

    payload = adapter.fetch_payload(
        'secret',
        base_url='https://example.test/v1',
        page_size=1,
        fetch_page=fake_fetch,
    )

    assert [call[2] for call in calls[1:]] == [
        {'limit': 1, 'offset': 0},
        {'limit': 1, 'offset': 1},
    ]
    assert all(call[1] == {'X-API-Key': 'secret'} for call in calls)
    assert len(payload['leaderboards']['rows']) == 2
    assert payload['leaderboards']['dataAsOf'] == '2026-06-30T20:32:54Z'


def test_fetch_payload_rejects_empty_page_before_total():
    responses = [
        {
            'schemaVersion': '1.0',
            'benchmarks': fixture_payload()['benchmarks']['benchmarks'],
            'dataAsOf': '2026-06-30T20:32:54Z',
        },
        {
            'schemaVersion': '1.0',
            'rows': [],
            'total': 1,
            'limit': 1,
            'offset': 0,
            'dataAsOf': '2026-06-30T20:32:54Z',
        },
    ]

    def fake_fetch(_url, **_kwargs):
        return responses.pop(0)

    with pytest.raises(ValueError, match='before total'):
        adapter.fetch_payload(
            'secret',
            base_url='https://example.test/v1',
            page_size=1,
            fetch_page=fake_fetch,
        )


def test_resolve_api_key_prefers_explicit_value(monkeypatch):
    monkeypatch.setenv(adapter.API_KEY_ENV, 'environment-key')

    assert adapter.resolve_api_key('explicit-key') == 'explicit-key'


def test_resolve_api_key_reads_environment(monkeypatch):
    monkeypatch.setenv(adapter.API_KEY_ENV, 'environment-key')

    assert adapter.resolve_api_key(None) == 'environment-key'


def test_resolve_api_key_fails_when_missing(monkeypatch):
    monkeypatch.delenv(adapter.API_KEY_ENV, raising=False)

    with pytest.raises(ValueError, match=adapter.API_KEY_ENV):
        adapter.resolve_api_key(None)


def _api_down(*args, **kwargs):
    import requests

    raise requests.ConnectionError('name resolution failed')


def test_an_unreachable_api_is_the_source_being_unavailable(monkeypatch):
    """Transport failures are Mercor being down, not the adapter crashing."""
    monkeypatch.setattr(adapter.requests, 'get', _api_down)

    with pytest.raises(adapter.SourceUnavailableError):
        adapter.request_json(
            f'{adapter.DEFAULT_BASE_URL}/benchmarks', headers={}
        )


def test_a_non_json_body_is_the_source_being_unavailable(monkeypatch):
    """A failing API serving an HTML error page is still an outage."""

    class HtmlResponse:
        url = f'{adapter.DEFAULT_BASE_URL}/benchmarks'
        content = b'<html>502</html>'
        headers = {'Content-Type': 'text/html'}

        def raise_for_status(self):
            return None

        def json(self):
            raise ValueError('not json')

    monkeypatch.setattr(adapter.requests, 'get', lambda *a, **k: HtmlResponse())

    with pytest.raises(adapter.SourceUnavailableError):
        adapter.request_json(
            f'{adapter.DEFAULT_BASE_URL}/benchmarks', headers={}
        )


def test_an_unavailable_api_exits_with_the_outage_code(monkeypatch, capsys):
    """Exit 75 is what lets a scheduled run report an outage, not a crash.

    The schema-contract failures deliberately do not take this path: an API
    that answers with the wrong shape needs a person, and stays a hard error
    (see test_unsupported_schema_version_fails).
    """
    monkeypatch.setenv(adapter.API_KEY_ENV, 'a-key')
    monkeypatch.setattr(adapter.requests, 'get', _api_down)
    monkeypatch.setattr(
        adapter.sys, 'argv', ['every_eval_ever.adapters.mercor_eval.adapter']
    )

    with pytest.raises(SystemExit) as caught:
        adapter.main()

    assert caught.value.code == adapter.SOURCE_UNAVAILABLE_EXIT == 75
    assert 'Mercor API unavailable' in capsys.readouterr().err


def test_a_rejected_key_is_a_hard_error_not_an_outage(monkeypatch):
    """An expired secret reported as "source down" would stay green for as
    long as nobody looked. Credentials are this side's configuration."""
    import requests

    class Rejected:
        status_code = 401

        def raise_for_status(self):
            error = requests.HTTPError('401 Unauthorized')
            error.response = self
            raise error

    monkeypatch.setattr(adapter.requests, 'get', lambda *a, **k: Rejected())

    with pytest.raises(RuntimeError, match=adapter.API_KEY_ENV):
        adapter.request_json(
            f'{adapter.DEFAULT_BASE_URL}/benchmarks', headers={}
        )
