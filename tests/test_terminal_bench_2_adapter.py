from __future__ import annotations

import json

import pytest

from utils.terminal_bench_2 import adapter


def _source_row(**overrides):
    row = {
        'agent': 'Example Agent',
        'model': ['Example Model'],
        'agentOrganization': 'Example Agent Org',
        'modelOrganization': ['Example Model Org'],
        'date': '2026-07-25',
        'accuracy': 0.8471910112359551,
        'stderr': 0.010659362899443975,
        'integrationMethod': 'API',
        'agentUrl': 'https://example.com/agent',
        'verified': True,
        'agentName': 'example-agent',
        'agentVersion': '1.0.0',
        'modelNames': ['example-model'],
        'modelProviders': ['example-provider'],
        'key': 'example-agent__example-model',
    }
    row.update(overrides)
    return row


def _flight_html(rows: list[dict]) -> str:
    message = f'26:[["$",null,null,{{"rows":{json.dumps(rows)}}}]]'
    argument = json.dumps([1, message])
    return f'<html><script>self.__next_f.push({argument})</script></html>'


def test_extracts_structured_flight_rows():
    rows = adapter.extract_leaderboard_rows(_flight_html([_source_row()]))

    assert rows == [
        {
            'rank': 1,
            'agent': 'Example Agent',
            'model': 'Example Model',
            'date': '2026-07-25',
            'agent_org': 'Example Agent Org',
            'model_org': 'Example Model Org',
            'accuracy': 84.7,
            'stderr': 1.065936,
            'verified': True,
            'agent_name': 'example-agent',
            'agent_version': '1.0.0',
            'agent_url': 'https://example.com/agent',
            'integration_method': 'API',
            'model_names': ['example-model'],
            'model_providers': ['example-provider'],
        }
    ]


def test_multiple_models_use_existing_multiple_identity():
    rows = adapter.extract_leaderboard_rows(
        _flight_html(
            [
                _source_row(
                    model=['Model A', 'Model B'],
                    modelOrganization=['Org A', 'Org B'],
                    modelNames=['model-a', 'model-b'],
                    modelProviders=['provider-a', 'provider-b'],
                )
            ]
        )
    )

    assert rows[0]['model'] == 'Multiple'
    assert rows[0]['model_org'] == 'Multiple'


def test_missing_or_malformed_rows_fail_closed():
    with pytest.raises(ValueError, match='exactly one'):
        adapter.extract_leaderboard_rows('<html></html>')

    with pytest.raises(ValueError, match='accuracy'):
        adapter.extract_leaderboard_rows(
            _flight_html([_source_row(accuracy='84.7')])
        )


def test_conversion_uses_live_source_metadata():
    entry = adapter.extract_leaderboard_rows(_flight_html([_source_row()]))[0]

    log = adapter.convert_entry(entry, '1234.5')
    result = log.evaluation_results[0]

    assert result.score_details.score == 84.7
    assert result.score_details.uncertainty is not None
    assert result.score_details.uncertainty.standard_error.value == 1.065936
    assert result.score_details.uncertainty.num_samples == 445
    assert (
        result.generation_config.generation_args.execution_command
        == 'harbor run -d terminal-bench/terminal-bench-2 '
        '-a "Example Agent" -m "Example Model" -k 5'
    )
    assert log.model_info.additional_details['leaderboard_verified'] == 'true'
