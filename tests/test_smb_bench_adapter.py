"""Offline tests for the SMB Bench adapter.

The fixture is an excerpt of the published dataset ``oro-agents/smb-bench``
(CC BY 4.0, revision 23486462): every run row of three models, one with all
54 runs scored (gpt-6-astra), one with two failed runs (gemini-4-hydrogen)
and one open-weights model with one failed run (glm-5.3), plus their rows of
the ``models`` and ``prices`` tables.
"""

import json
import math
from pathlib import Path

import pytest

from every_eval_ever.adapters.smb_bench import adapter
from every_eval_ever.helpers import eval_card_registry as registry_mod
from every_eval_ever.helpers.io import SourceRecordsError
from every_eval_ever.validate import validate_file

FIXTURE_DIR = Path(__file__).parent / 'data' / 'smb_bench'
REVISION = '23486462ddbb916dc241adb1c4f8f8150f9bc1db'
RETRIEVED = '1234567890.0'


@pytest.fixture
def source() -> dict:
    return adapter.load_source(FIXTURE_DIR, REVISION)


def _convert(source, tmp_path, **kwargs):
    return adapter.convert_source(
        source,
        tmp_path / 'data' / adapter.COLLECTION,
        retrieved_timestamp=RETRIEVED,
        **kwargs,
    )


def _log(result, model_name):
    for output in result.records:
        if output.eval_log.model_info.name == model_name:
            return output.eval_log
    raise AssertionError(f'no converted log for {model_name!r}')


def _result(log, suffix):
    for result in log.evaluation_results:
        if result.evaluation_result_id.endswith(f'#{suffix}'):
            return result
    raise AssertionError(f'no result {suffix!r}')


def test_models_convert_and_validate(source, tmp_path):
    result = _convert(source, tmp_path)

    assert result.total_records == 3
    assert len(result.records) == 3
    assert not result.failures
    result.raise_if_incomplete()

    paths = adapter.save_evaluation_logs(result.records)
    collection = tmp_path / 'data' / adapter.COLLECTION
    assert sorted(path.parent.relative_to(collection) for path in paths) == [
        Path('google/gemini-4-hydrogen'),
        Path('openai/gpt-6-astra'),
        Path('zai-org/GLM-5.3'),
    ]
    for path in paths:
        report = validate_file(path)
        assert report.valid, report.errors


def test_headline_is_the_task_mean_of_run_means(source, tmp_path):
    """Hand-computed from the fixture rows, so the aggregation cannot drift.

    The benchmark's figure is the mean over the 18 tasks of the mean of the
    model's scored runs on each task; 7748.98 is the value its leaderboard
    shows for this model.
    """
    log = _log(_convert(source, tmp_path), 'gpt-6-astra')
    headline = _result(log, 'mean-profit-usd')

    assert headline.evaluation_name == 'smb-bench-v1'
    assert headline.score_details.score == pytest.approx(7748.980185185185)
    assert round(headline.score_details.score, 2) == 7748.98
    assert headline.score_details.details['fixed_rule_mean_profit_usd'] == (
        json.dumps(6411.9127777777785)
    )


def test_standard_error_is_over_tasks(source, tmp_path):
    """The task is the sampling unit: sd of 18 task means over sqrt(18)."""
    log = _log(_convert(source, tmp_path), 'gpt-6-astra')
    uncertainty = _result(log, 'mean-profit-usd').score_details.uncertainty

    task_mean_sd = 6905.524627476393
    assert uncertainty.num_samples == 18
    assert uncertainty.standard_deviation == pytest.approx(task_mean_sd)
    assert uncertainty.standard_error.value == pytest.approx(
        task_mean_sd / math.sqrt(18)
    )
    assert uncertainty.standard_error.method.startswith('analytic')

    gain = _result(log, 'mean-profit-vs-fixed-rule-usd')
    assert gain.score_details.score == pytest.approx(1337.0674074074077)
    assert gain.score_details.uncertainty.standard_error.value == (
        pytest.approx(862.2837902742256)
    )


def test_failed_runs_are_counted_and_left_out_of_every_mean(source, tmp_path):
    """A failed run has no score upstream; it is never counted as zero."""
    log = _log(_convert(source, tmp_path), 'gemini-4-hydrogen')
    headline = _result(log, 'mean-profit-usd')
    details = headline.score_details.details

    assert headline.score_details.score == pytest.approx(2560.4862962962966)
    assert (details['runs'], details['scored_runs']) == ('54', '52')
    assert details['unscored_runs'] == '2'
    assert json.loads(details['run_status_counts']) == {
        'failed': 2,
        'valid': 52,
    }
    assert [
        item['cell_id'] for item in json.loads(details['unscored_run_ids'])
    ] == [
        'gemini-4-hydrogen__smb-bench-control-4__r3',
        'gemini-4-hydrogen__smb-bench-public-10__r2',
    ]
    # Every task still has a scored run, so all 18 enter the task mean.
    assert details['tasks_with_a_scored_run'] == '18'

    cost = _result(log, 'cost-per-task')
    assert cost.score_details.score == pytest.approx(25.436920102884617)
    assert cost.score_details.details['scored_runs'] == '52'
    # The failure is reported by run id and status only.
    assert set(json.loads(details['unscored_run_ids'])[0]) == {
        'cell_id',
        'status',
    }


def test_cost_states_how_the_source_priced_each_model(source, tmp_path):
    """Not every model has a list price; the price table says which."""
    result = _convert(source, tmp_path)

    listed = _result(_log(result, 'gpt-6-astra'), 'cost-per-task')
    assert listed.score_details.details['cost_method'] == (
        'tokens times the list prices in this row'
    )
    unlisted = _result(_log(result, 'glm-5.3'), 'cost-per-task')
    assert (
        'no list price was published'
        in (unlisted.score_details.details['cost_method'])
    )
    for cost in (listed, unlisted):
        assert 'list price' not in cost.metric_config.evaluation_description


def test_three_kinds_of_quarter_partition_the_headline(source, tmp_path):
    log = _log(_convert(source, tmp_path), 'gpt-6-astra')

    names = [result.evaluation_name for result in log.evaluation_results]
    assert names == [
        'smb-bench-v1',
        'smb-bench-v1.normal-quarters',
        'smb-bench-v1.customer-quarters',
        'smb-bench-v1.supplier-quarters',
        'smb-bench-v1',
        'smb-bench-v1',
    ]
    assert len({r.evaluation_result_id for r in log.evaluation_results}) == 6

    overall = _result(log, 'mean-profit-usd')
    kinds = [
        _result(log, f'mean-profit-usd.{kind}-quarters')
        for kind in ('normal', 'customer', 'supplier')
    ]
    assert overall.score_details.details['aggregation_level'] == 'overall'
    # The kinds partition the 18 tasks, so a consumer that took both levels
    # would double-count; each result says which level it is.
    assert [k.score_details.details['aggregation_level'] for k in kinds] == [
        'kind:normal',
        'kind:customer',
        'kind:supplier',
    ]
    assert [k.score_details.details['tasks'] for k in kinds] == ['6'] * 3
    assert [k.source_data.samples_number for k in kinds] == [6] * 3
    assert overall.source_data.samples_number == 18
    assert [k.score_details.uncertainty.num_samples for k in kinds] == [6] * 3
    assert sum(k.score_details.score for k in kinds) / 3 == pytest.approx(
        overall.score_details.score
    )
    assert [round(k.score_details.score, 2) for k in kinds] == [
        10144.82,
        8808.76,
        4293.37,
    ]


def test_profit_is_unbounded_and_cost_is_bounded_below(source, tmp_path):
    result = _convert(source, tmp_path)
    log = _log(result, 'glm-5.3')

    profit = _result(log, 'mean-profit-usd')
    assert profit.score_details.score < 0
    assert profit.metric_config.metric_id == 'smb-bench.mean-profit-usd'
    assert profit.metric_config.metric_unit == 'usd'
    assert profit.metric_config.lower_is_better is False
    assert profit.metric_config.min_score == -math.inf
    assert profit.metric_config.max_score == math.inf

    cost = _result(log, 'cost-per-task').metric_config
    assert cost.metric_id == 'cost-per-task'
    assert cost.lower_is_better is True
    assert (cost.min_score, cost.max_score) == (0.0, math.inf)

    # Unbounded sides are written as the schema's strings, not bare tokens.
    paths = adapter.save_evaluation_logs(result.records)
    for path in paths:
        text = path.read_text(encoding='utf-8')
        saved = json.loads(text)  # strict JSON: a bare Infinity would fail
        metric = saved['evaluation_results'][0]['metric_config']
        assert (metric['min_score'], metric['max_score']) == (
            '-Infinity',
            'Infinity',
        )


def test_model_ids_come_from_the_registry_snapshot(source, tmp_path):
    result = _convert(source, tmp_path)

    glm = _log(result, 'glm-5.3').model_info
    assert glm.id == 'zai-org/GLM-5.3'
    assert glm.developer == 'zai'
    assert glm.additional_details['model_availability'] == 'open_weights'
    assert glm.additional_details['deployment_type'] == 'externally_managed'
    assert glm.additional_details['served_model_id'] == 'zai-org/GLM-5.3'
    assert glm.additional_details['model_id_source'] == 'eval-card-registry'
    assert glm.additional_details['model_registry_matched_field'] == (
        'model_id'
    )
    # The source's `provider` column is not published as a platform.
    assert 'provider' not in glm.additional_details

    astra = _log(result, 'gpt-6-astra').model_info
    assert astra.id == 'openai/gpt-6-astra'
    assert astra.additional_details['model_availability'] == 'closed_weights'
    assert (
        astra.additional_details['model_registry_revision']
        == (adapter.load_registry_snapshot()['_meta']['registry_revision'])
    )


def test_registry_snapshot_covers_every_published_model():
    """Refreshed against the full ``models`` table, not just the fixture."""
    snapshot = adapter.load_registry_snapshot()

    assert snapshot['_meta']['count'] == len(snapshot['models']) == 16
    assert snapshot['_meta']['unresolved'] == []
    ids = [entry['canonical_id'] for entry in snapshot['models'].values()]
    assert len(set(ids)) == 16
    assert all('/' in model_id for model_id in ids)
    assert '-dirty' not in snapshot['_meta']['registry_revision']
    # A served id with no organization resolves under the label's, so the
    # two versioned Muse ids come out the same way.
    assert snapshot['models']['muse-ember']['canonical_id'] == (
        'meta/muse-ember-1-0'
    )
    assert snapshot['models']['muse-wonder-1.0']['canonical_id'] == (
        'meta/muse-wonder-1-0'
    )
    # Matched on the benchmark's label, because the served `-Fast` id is
    # not in the registry.
    assert snapshot['models']['deepseek-v4.1-flash-1017']['matched_field'] == (
        'model'
    )


def test_no_registry_resolve_keeps_source_spellings_unverified(
    source, tmp_path
):
    result = _convert(
        source, tmp_path, registry=registry_mod.Registry(enabled=False)
    )

    assert not result.failures
    model = _log(result, 'gpt-6-astra').model_info
    assert model.id == 'openai/gpt-6-astra'
    assert model.additional_details['model_id_verified'] == 'false'
    assert model.additional_details['model_registry_strategy'] == (
        'registry_disabled'
    )
    assert model.additional_details['model_availability'] == 'unknown'


def test_evaluation_id_is_stable_and_not_keyed_on_the_registry_id(
    source, tmp_path
):
    """Keyed on the source's model label and the run table's content."""
    first = _convert(source, tmp_path)
    second = adapter.convert_source(
        source,
        tmp_path / 'other' / adapter.COLLECTION,
        retrieved_timestamp='9999999999.0',
        registry=registry_mod.Registry(enabled=False),
    )

    ids = [output.eval_log.evaluation_id for output in first.records]
    assert ids == [output.eval_log.evaluation_id for output in second.records]
    assert ids[1].startswith('smb-bench-v1/smb-bench-buyer__glm-5.3/runs-')
    assert RETRIEVED not in ids[1]
    assert 'zai-org' not in ids[1]


def test_adding_a_model_leaves_the_other_evaluation_ids_alone(source, tmp_path):
    """The id digests a model's own rows, not the whole run table."""
    before = {
        output.eval_log.model_info.name: output.eval_log.evaluation_id
        for output in _convert(source, tmp_path).records
    }
    source['runs'] = [
        row for row in source['runs'] if row['model'] != 'gpt-6-astra'
    ]
    after = {
        output.eval_log.model_info.name: output.eval_log.evaluation_id
        for output in _convert(source, tmp_path / 'fewer').records
    }

    assert set(after) == {'gemini-4-hydrogen', 'glm-5.3'}
    assert all(before[name] == after[name] for name in after)

    # A changed run of a model does change that model's id, and only its.
    for row in source['runs']:
        if row['cell_id'] == 'glm-5-3__smb-bench-public-1__r1':
            row['business_return_cents'] += 1
    changed = {
        output.eval_log.model_info.name: output.eval_log.evaluation_id
        for output in _convert(source, tmp_path / 'changed').records
    }
    assert changed['glm-5.3'] != after['glm-5.3']
    assert changed['gemini-4-hydrogen'] == after['gemini-4-hydrogen']


def test_run_settings_come_from_the_models_table(source, tmp_path):
    log = _log(_convert(source, tmp_path), 'gemini-4-hydrogen')
    config = log.evaluation_results[0].generation_config

    assert config.generation_args.max_tokens == 65536
    assert config.generation_args.reasoning is True
    assert config.generation_args.temperature is None
    assert config.additional_details['reasoning_setting'] == (
        'thinking level high'
    )
    assert config.additional_details['temperature'] == 'not sent'
    assert config.additional_details['prompt_version'] == 'buyer-prompt-v11'
    assert config.additional_details['max_tool_calls_per_day'] == '60'
    assert config.additional_details['simulated_days_per_run'] == '[90]'
    assert 'provider' not in config.additional_details
    assert log.source_metadata.source_type.value == 'evaluation_run'
    assert log.source_metadata.evaluator_relationship.value == 'third_party'
    assert log.source_metadata.additional_details['hf_revision'] == REVISION
    assert log.evaluation_timestamp == '2026-11-13T00:58:07.471648+00:00'


def test_an_unknown_task_fails_its_model_and_the_run(source, tmp_path):
    """A row that cannot be placed is reported, never silently dropped."""
    for row in source['runs']:
        if row['cell_id'] == 'glm-5-3__smb-bench-public-1__r1':
            row['public_task_id'] = 'smb-bench-public-99'

    result = _convert(source, tmp_path)

    assert len(result.records) == 2
    assert [failure.source_ref for failure in result.failures] == [
        "model 'glm-5.3' (54 run rows)"
    ]
    assert 'smb-bench-public-99' in result.failures[0].reason
    with pytest.raises(SourceRecordsError):
        result.raise_if_incomplete()


def test_a_repeated_run_row_fails_its_model(source, tmp_path):
    """A duplicated row would shift a task mean without any other sign."""
    source['runs'].append(
        dict(
            next(
                row
                for row in source['runs']
                if row['cell_id'] == 'gpt-6-astra__smb-bench-public-1__r1'
            )
        )
    )

    result = _convert(source, tmp_path)

    assert len(result.records) == 2
    assert 'appears more than once' in result.failures[0].reason


def test_a_task_with_no_scored_run_is_not_described_as_covered(
    source, tmp_path
):
    for row in source['runs']:
        if (
            row['model'] == 'gpt-6-astra'
            and row['public_task_id'] == 'smb-bench-public-1'
        ):
            row['status'] = 'failed'

    headline = _result(
        _log(_convert(source, tmp_path), 'gpt-6-astra'), 'mean-profit-usd'
    )

    assert headline.score_details.uncertainty.num_samples == 17
    assert headline.score_details.details['tasks_with_a_scored_run'] == '17'
    assert 'the 17 of 18 tasks that have a scored run' in (
        headline.metric_config.evaluation_description
    )


def test_only_a_named_effort_or_thinking_level_counts_as_reasoning(
    source, tmp_path
):
    source['models']['glm-5.3']['reasoning'] = 'the provider default'

    result = _convert(source, tmp_path)

    default = _log(result, 'glm-5.3').evaluation_results[0].generation_config
    assert default.generation_args.reasoning is None
    assert default.additional_details['reasoning_setting'] == (
        'the provider default'
    )
    named = _log(result, 'gpt-6-astra').evaluation_results[0]
    assert named.generation_config.generation_args.reasoning is True


def test_a_model_missing_from_the_models_table_is_a_failure(source, tmp_path):
    del source['models']['gpt-6-astra']

    result = _convert(source, tmp_path)

    assert len(result.records) == 2
    assert 'not in the models table' in result.failures[0].reason


def test_a_kind_listed_under_the_wrong_split_is_rejected(source, tmp_path):
    """The task listing is checked against the run table's own split."""
    for row in source['runs']:
        if row['public_task_id'] == 'smb-bench-control-1':
            row['split'] = 'leaderboard'

    result = _convert(source, tmp_path)

    assert not result.records
    assert len(result.failures) == 3
    assert 'normal quarter' in result.failures[0].reason
