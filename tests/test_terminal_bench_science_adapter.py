"""Offline tests for the Terminal-Bench-Science leaderboard adapter.

The fixture is three rows of the live payload: one closed-weights row, one
open-weights row, and one the leaderboard is not publishing.
"""

import json
import math
from pathlib import Path

import pytest

from every_eval_ever.adapters.terminal_bench_science import adapter
from every_eval_ever.helpers import eval_card_registry as registry_mod
from every_eval_ever.helpers.io import SourceRecordsError
from every_eval_ever.validate import validate_file

FIXTURE = (
    Path(__file__).parent
    / 'data'
    / 'terminal_bench_science'
    / 'leaderboard.json'
)
RETRIEVED = '1234567890.0'


@pytest.fixture
def payload() -> dict:
    return json.loads(FIXTURE.read_text(encoding='utf-8'))


def _convert(payload, tmp_path, **kwargs):
    return adapter.convert_payload(
        payload,
        tmp_path / 'data' / adapter.COLLECTION,
        retrieved_timestamp=RETRIEVED,
        **kwargs,
    )


def _log(result, model_name):
    for output in result.records:
        if output.eval_log.model_info.name == model_name:
            return output.eval_log
    raise AssertionError(f'no converted log for {model_name!r}')


def test_published_rows_convert_and_validate(payload, tmp_path):
    result = _convert(payload, tmp_path)

    assert result.total_records == 3
    assert len(result.records) == 2
    assert not result.failures

    paths = adapter.save_evaluation_logs(result.records)
    assert len(paths) == 2
    for path in paths:
        report = validate_file(path)
        assert report.valid, report.errors


def test_an_unpublished_row_is_excluded_not_dropped(payload, tmp_path):
    """A hidden row is reported, but it does not fail the refresh."""
    result = _convert(payload, tmp_path)

    assert [exclusion.source_ref for exclusion in result.exclusions] == [
        'leaderboard row 3'
    ]
    assert 'hidden' in result.exclusions[0].reason
    # Exclusions are not failures: the command still exits zero.
    result.raise_if_incomplete()


def test_a_malformed_row_is_reported_and_fails_the_run(payload, tmp_path):
    payload['rows'][0]['metadata']['model_display'] = {'url': 'x'}

    result = _convert(payload, tmp_path)

    assert len(result.records) == 1
    assert [failure.source_ref for failure in result.failures] == [
        'leaderboard row 1'
    ]
    with pytest.raises(SourceRecordsError):
        result.raise_if_incomplete()


def test_overall_and_five_domains_each_carry_their_own_n(payload, tmp_path):
    log = _log(_convert(payload, tmp_path), 'Fable 5.1')

    names = [result.evaluation_name for result in log.evaluation_results]
    assert names == [
        'terminal-bench-science-0.1',
        'terminal-bench-science-0.1.life-sciences',
        'terminal-bench-science-0.1.physical-sciences',
        'terminal-bench-science-0.1.earth-sciences',
        'terminal-bench-science-0.1.mathematical-sciences',
        'terminal-bench-science-0.1.engineering-sciences',
    ]

    overall, *domains = log.evaluation_results
    assert overall.score_details.details['aggregation_level'] == 'overall'
    assert overall.score_details.uncertainty.num_samples == 210
    # The parts sum to the whole, so a consumer that took both would
    # double-count; each result says which level it is.
    assert (
        sum(int(domain.score_details.details['trials']) for domain in domains)
        == 210
    )
    assert sum(
        int(domain.score_details.details['trials_passed']) for domain in domains
    ) == int(overall.score_details.details['trials_passed'])
    for domain in domains:
        assert domain.score_details.details['aggregation_level'].startswith(
            'domain:'
        )


def test_published_standard_error_is_the_binomial_one(payload, tmp_path):
    """Hand-computed, so a source change to a different estimator shows up.

    The leaderboard's figure is sqrt(p(1-p)/n) on the percent scale over the
    trials, not a resampled spread, which is why it is published as analytic.
    """
    log = _log(_convert(payload, tmp_path), 'Fable 5.1')
    overall = log.evaluation_results[0]

    passes, trials = 84, 210
    rate = passes / trials
    expected = 100.0 * math.sqrt(rate * (1.0 - rate) / trials)

    assert overall.score_details.score == pytest.approx(100.0 * rate)
    assert overall.score_details.uncertainty.standard_error.value == (
        pytest.approx(expected)
    )
    assert overall.score_details.uncertainty.standard_error.method == (
        'analytic'
    )


def test_the_metric_is_namespaced_on_the_leaderboards_percent_scale(
    payload, tmp_path
):
    """A trial resolution rate must not join to the registry's `accuracy`.

    The registry's `accuracy` is a 0-1 proportion over items; this is the share
    of 3-trial attempts a verifier accepted, published as a percentage. The id
    is namespaced the same way the sibling `terminal_bench_2` adapter does, so
    the two Terminal-Bench leaderboards join with each other and with nothing
    else.
    """
    log = _log(_convert(payload, tmp_path), 'Fable 5.1')

    for result in log.evaluation_results:
        metric = result.metric_config
        assert metric.metric_id == 'terminal-bench-science.accuracy'
        assert metric.metric_unit == 'percent'
        assert (metric.min_score, metric.max_score) == (0.0, 100.0)
        assert metric.lower_is_better is False


def test_evaluation_id_is_stable_and_separates_scaffold_variants(
    payload, tmp_path
):
    """Keyed on the leaderboard's own row id, never on `now`."""
    first = _convert(payload, tmp_path)
    second = adapter.convert_payload(
        payload,
        tmp_path / 'other' / adapter.COLLECTION,
        retrieved_timestamp='9999999999.0',
    )

    ids = [output.eval_log.evaluation_id for output in first.records]
    assert ids == [output.eval_log.evaluation_id for output in second.records]
    assert ids[0] == (
        'terminal-bench-science-0.1/claude-code__fable-5.1__max/'
        '1264a994-632f-4742-a143-a848a6ad4114'
    )
    # The agent and effort ride in the id, so one model under two scaffolds
    # stays two records rather than collapsing into one.
    assert RETRIEVED not in ids[0]


def test_deployment_axes_are_set_per_organization(payload, tmp_path):
    result = _convert(payload, tmp_path)

    closed = _log(result, 'Fable 5.1').model_info
    assert closed.id == 'anthropic/fable-5.1'
    assert closed.additional_details['deployment_type'] == (
        'externally_managed'
    )
    assert closed.additional_details['model_availability'] == 'closed_weights'

    open_weights = _log(result, 'Kimi K3').model_info
    assert open_weights.id == 'moonshotai/kimi-k3'
    assert open_weights.additional_details['model_availability'] == (
        'open_weights'
    )


def test_an_unmapped_organization_yields_unknown_not_a_guess(payload, tmp_path):
    payload['rows'][0]['metadata']['model_org'] = {
        'label': 'Some New Lab',
        'url': 'https://example.com',
    }

    log = _log(_convert(payload, tmp_path), 'Fable 5.1')

    assert log.model_info.additional_details['model_availability'] == (
        'unknown'
    )


def test_model_ids_are_published_as_unverified(payload, tmp_path):
    """The registry has no model entry for these releases yet.

    Only the organization half is canonicalized; the record says so rather than
    implying the whole id was resolved.
    """
    log = _log(_convert(payload, tmp_path), 'Fable 5.1')
    details = log.model_info.additional_details

    assert details['model_id_verified'] == 'false'
    assert details['model_id_source'] == 'leaderboard_labels'
    assert details['developer_registry_id'] == 'anthropic'
    assert details['developer_registry_review_status'] == 'reviewed'
    # Harbor is not a canonical harness either. There is no id to publish, so
    # the strategy is what carries that: the record says the registry was
    # asked and had no answer, rather than staying silent about it.
    assert log.eval_library.name == 'harbor'
    harness = log.eval_library.additional_details
    assert 'harness_registry_id' not in harness
    assert harness['harness_registry_strategy'] == 'no_canonical'


def test_registry_can_be_turned_off_without_changing_the_path(
    payload, tmp_path
):
    result = _convert(
        payload, tmp_path, registry=registry_mod.Registry(enabled=False)
    )

    log = _log(result, 'Fable 5.1')
    # Falls back to the source's own spelling, marked as having had no
    # registry opinion rather than quietly looking source-derived.
    assert log.model_info.id == 'anthropic/fable-5.1'
    assert (
        log.model_info.additional_details['developer_registry_strategy']
        == 'registry_disabled'
    )


def test_source_version_tracks_published_state_only(payload):
    baseline = adapter.source_version(payload)

    assert adapter.source_version(json.loads(json.dumps(payload))) == baseline

    payload['rows'][0]['metrics']['accuracy'] = 41.0
    assert adapter.source_version(payload) != baseline


def test_api_url_is_the_endpoint_the_leaderboard_page_reads():
    assert adapter.leaderboard_api_url() == (
        'https://terminal-bench-science.ai/api/leaderboard'
        '?package=terminal-bench-science%2Fterminal-bench-science'
        '&name=v0-1-eval'
    )


@pytest.mark.parametrize(
    'flag,value',
    [('--package', 'other/benchmark'), ('--leaderboard-name', 'v0-2-eval')],
)
def test_cli_rejects_sources_that_would_be_mislabeled_as_release_01(
    flag, value
):
    with pytest.raises(SystemExit) as error:
        adapter.parse_args([flag, value])
    assert error.value.code == 2


@pytest.mark.parametrize(
    'field,value',
    [('package', 'other/benchmark'), ('name', 'v0-2-eval'), ('name', None)],
)
def test_replay_and_version_probe_reject_an_unsupported_leaderboard(
    payload, tmp_path, field, value
):
    payload['leaderboard'][field] = value
    with pytest.raises(ValueError, match='leaderboard'):
        _convert(payload, tmp_path)
    with pytest.raises(ValueError, match='leaderboard'):
        adapter.source_version(payload)


@pytest.mark.parametrize('domains', [None, {}, [], {'life': {}}])
def test_incomplete_domains_fail_the_row_instead_of_publishing_partial_results(
    payload, tmp_path, domains
):
    payload['rows'][0]['metrics']['domain_metrics'] = domains
    result = _convert(payload, tmp_path)
    assert len(result.records) == 1
    assert len(result.failures) == 1
    assert 'domain' in result.failures[0].reason
    with pytest.raises(SourceRecordsError):
        result.raise_if_incomplete()


@pytest.mark.parametrize(
    'domain', ['life', 'physical', 'earth', 'mathematical', 'engineering']
)
def test_each_domain_is_required(payload, tmp_path, domain):
    del payload['rows'][0]['metrics']['domain_metrics'][domain]
    result = _convert(payload, tmp_path)
    assert len(result.records) == 1
    assert len(result.failures) == 1
    assert domain in result.failures[0].reason


@pytest.mark.parametrize('scope', ['overall', 'life'])
@pytest.mark.parametrize(
    'field,value',
    [
        ('tasks', 210.5),
        ('tasks', 209),
        ('tasks', 0),
        ('passes', 20.5),
        ('passes', -0.5),
    ],
)
def test_invalid_counts_fail_without_rounding(
    payload, tmp_path, scope, field, value
):
    metrics = payload['rows'][0]['metrics']
    if scope != 'overall':
        metrics = metrics['domain_metrics'][scope]
    metrics[field] = value
    result = _convert(payload, tmp_path)
    assert len(result.records) == 1
    assert len(result.failures) == 1
    assert field in result.failures[0].reason
    with pytest.raises(SourceRecordsError):
        result.raise_if_incomplete()


def test_cli_writes_a_failure_report_for_missing_domains(payload, tmp_path):
    del payload['rows'][0]['metrics']['domain_metrics']
    source = tmp_path / 'source.json'
    source.write_text(json.dumps(payload))
    report = tmp_path / 'failures.json'
    with pytest.raises(SourceRecordsError):
        adapter.main(
            [
                '--input-json',
                str(source),
                '--output-dir',
                str(tmp_path / 'data' / adapter.COLLECTION),
                '--failure-report',
                str(report),
            ]
        )
    failures = json.loads(report.read_text())
    assert failures['failed_record_count'] == 1
    assert failures['converted_records'] == 1
    assert failures['failed_records'][0]['source_ref'] == 'leaderboard row 1'
    assert 'domain' in failures['failed_records'][0]['reason']
