"""Unit tests for the AlpacaEval adapter.

The adapter reads a *snapshot* of upstream ``tatsu-lab/alpaca_eval`` — the two
leaderboard CSVs plus the judge configs, judge prompts and per-model configs
that make those numbers interpretable. These tests build such a snapshot
in-memory, so nothing here touches the network.
"""

import json

import pytest

from every_eval_ever import cli
from every_eval_ever.converters.alpaca_eval import identity as identity_mod
from every_eval_ever.converters.alpaca_eval import (
    refresh_hf_canonical_ids as refresh_mod,
)
from every_eval_ever.converters.alpaca_eval import upstream as upstream_mod
from every_eval_ever.converters.alpaca_eval.adapter import (
    LEADERBOARDS,
    AlpacaEvalAdapter,
    model_slug_from_row,
)
from every_eval_ever.converters.alpaca_eval.upstream import (
    DEFAULT_UPSTREAM_REF,
    LeaderboardSnapshot,
    UpstreamSnapshot,
    parse_single_entry_yaml,
    raw_url,
    resolve_ref,
)
from every_eval_ever.helpers import eval_card_registry as registry_mod
from every_eval_ever.helpers.fetch import FetchError
from every_eval_ever.helpers.io import SourceRecordsError

# ---------------------------------------------------------------------------
# Fixture snapshot
# ---------------------------------------------------------------------------

_V1_ROW = {
    '': 'gpt4',
    'win_rate': '95.28',
    'standard_error': '0.68',
    'n_wins': '767',
    'n_wins_base': '38',
    'n_draws': '0',
    'n_total': '805',
    'mode': 'minimal',
    'avg_length': '1365',
    'discrete_win_rate': '95.28',
    'length_controlled_winrate': '',
}

_V2_ROW = {
    '': 'gpt4_turbo',
    'win_rate': '50.0',
    'standard_error': '0.0',
    'n_wins': '0',
    'n_wins_base': '0',
    'n_draws': '805',
    'n_total': '805',
    'discrete_win_rate': '50.0',
    'mode': 'minimal',
    'avg_length': '2049',
    'length_controlled_winrate': '55.12',
    'lc_standard_error': '0.72',
}

_V1_JUDGE_PROMPT = '<|im_start|>system\nRank the models.\n<|im_end|>\n'
_V2_JUDGE_PROMPT = '<|im_start|>system\nPick the better model.\n<|im_end|>\n'

_V1_ANNOTATOR = {
    'prompt_template': 'alpaca_eval_gpt4/alpaca_eval.txt',
    'fn_completions': 'openai_completions',
    'completions_kwargs': {
        'model_name': 'gpt-4',
        'max_tokens': 100,
        'temperature': 0,
    },
    'fn_completion_parser': 'ranking_parser',
}

_V2_ANNOTATOR = {
    'prompt_template': 'alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt',
    'fn_completions': 'openai_completions',
    'completions_kwargs': {
        'model_name': 'gpt-4-1106-preview',
        'max_tokens': 1,
        'temperature': 1,
        'logprobs': True,
        'top_logprobs': 5,
    },
    'fn_completion_parser': 'logprob_parser',
    'completion_parser_kwargs': {
        'numerator_token': 'm',
        'denominator_tokens': ['m', 'M'],
        'is_binarize': False,
    },
}

_MODEL_CONFIGS = {
    'gpt4': {
        'prompt_template': 'gpt4/chatml_prompt.txt',
        'fn_completions': 'openai_completions',
        'completions_kwargs': {
            'model_name': 'gpt-4',
            'max_tokens': 2048,
            'temperature': 0.7,
            'top_p': 1.0,
        },
        'pretty_name': 'GPT-4',
    },
    'gpt4_turbo': {
        'prompt_template': 'gpt4_turbo/chatml_prompt.txt',
        'fn_completions': 'openai_completions',
        'completions_kwargs': {
            'model_name': 'gpt-4-1106-preview',
            'max_tokens': 4096,
        },
        'pretty_name': 'GPT-4 Turbo',
    },
    # Upstream serves this through Together, and links the Meta repo: the id
    # prefix is the HuggingFace namespace, ``meta-llama``.
    'Meta-Llama-3-70B-Instruct': {
        'prompt_template': 'Mixtral-8x7B-Instruct-v0.1/togetherai_prompt.txt',
        'fn_completions': 'openai_completions',
        'completions_kwargs': {
            'model_name': 'meta-llama/Llama-3-70b-chat-hf',
            'max_tokens': 4096,
            'client_kwargs': {'base_url': 'https://api.together.xyz'},
        },
        'pretty_name': 'Llama 3 70B Instruct',
        'link': 'https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct',
    },
    # Ran from a local checkout, so only the link names the repo — and that repo
    # has since been renamed (``WizardLM`` -> ``WizardLMTeam``).
    'wizardlm-13b-v1.2': {
        'prompt_template': 'wizardlm-13b/prompt.txt',
        'fn_completions': 'huggingface_local_completions',
        'completions_kwargs': {
            'model_name': './wizardlm_13b-v1.2',
            'max_new_tokens': 4096,
            'temperature': 0.7,
        },
        'pretty_name': 'WizardLM 13B V1.2',
        'link': 'https://huggingface.co/WizardLM/WizardLM-13B-V1.2',
    },
    'vicuna-7b': {
        'prompt_template': 'vicuna-7b/prompt.txt',
        'fn_completions': 'huggingface_local_completions',
        'completions_kwargs': {'temperature': 0.7, 'max_new_tokens': 2048},
        'link': 'https://huggingface.co/lmsys/vicuna-7b-delta-v1.1',
        'pretty_name': 'Vicuna 7B',
    },
}

#: Verbatim prompt text, keyed by the ``prompt_template`` paths above. Every
#: config here names one except ``gpt4_turbo``, which stands in for a template
#: upstream did not serve.
_MODEL_PROMPTS = {
    'gpt4/chatml_prompt.txt': '<|im_start|>user\n{instruction}<|im_end|>\n',
    'Mixtral-8x7B-Instruct-v0.1/togetherai_prompt.txt': '[INST] {instruction}',
    'wizardlm-13b/prompt.txt': 'USER: {instruction} ASSISTANT:',
    'vicuna-7b/prompt.txt': 'USER: {instruction}\nASSISTANT:',
}


def _snapshot(v1_rows=None, v2_rows=None) -> UpstreamSnapshot:
    """Build an in-memory upstream snapshot for the given leaderboard rows."""
    boards = {}
    if v1_rows is not None:
        boards['v1'] = LeaderboardSnapshot(
            rows=[dict(row) for row in v1_rows],
            annotator_config=dict(_V1_ANNOTATOR),
            judge_prompt=_V1_JUDGE_PROMPT,
            judge_prompt_path=(
                'src/alpaca_eval/evaluators_configs/'
                'alpaca_eval_gpt4/alpaca_eval.txt'
            ),
        )
    if v2_rows is not None:
        boards['v2'] = LeaderboardSnapshot(
            rows=[dict(row) for row in v2_rows],
            annotator_config=dict(_V2_ANNOTATOR),
            judge_prompt=_V2_JUDGE_PROMPT,
            judge_prompt_path=(
                'src/alpaca_eval/evaluators_configs/'
                'alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt'
            ),
        )
    return UpstreamSnapshot(
        ref=DEFAULT_UPSTREAM_REF,
        package_version='0.6.6',
        leaderboards=boards,
        model_configs={k: dict(v) for k, v in _MODEL_CONFIGS.items()},
        model_prompts=dict(_MODEL_PROMPTS),
    )


def _adapter(v1_rows=None, v2_rows=None, registry=None) -> AlpacaEvalAdapter:
    return AlpacaEvalAdapter(
        snapshot=_snapshot(v1_rows, v2_rows), registry=registry
    )


def _by_metric(log):
    return {r.metric_config.metric_name: r for r in log.evaluation_results}


@pytest.fixture
def snapshot_file(tmp_path):
    """A ``--save-raw-json``-shaped snapshot on disk, for CLI replay tests."""

    def _write(v1_rows=None, v2_rows=None):
        path = tmp_path / 'upstream.json'
        path.write_text(
            json.dumps(_snapshot(v1_rows, v2_rows).to_payload()),
            encoding='utf-8',
        )
        return path

    return _write


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------


def test_model_slug_from_unnamed_column():
    assert (
        model_slug_from_row({'': 'my_model', 'win_rate': '50'}) == 'my_model'
    )


def test_model_slug_fallback_to_first_value():
    assert model_slug_from_row({'x': 'fallback'}) == 'fallback'


# ---------------------------------------------------------------------------
# Snapshot plumbing
# ---------------------------------------------------------------------------


def test_snapshot_round_trips_through_payload():
    original = _snapshot([_V1_ROW], [_V2_ROW])
    restored = UpstreamSnapshot.from_payload(original.to_payload())

    assert restored.ref == original.ref
    assert restored.package_version == '0.6.6'
    assert restored.leaderboards['v1'].rows == original.leaderboards['v1'].rows
    assert restored.leaderboards['v2'].judge_prompt == _V2_JUDGE_PROMPT
    assert restored.model_configs == original.model_configs
    # Without this, replaying a saved snapshot loses every prompt.
    assert restored.model_prompts == original.model_prompts


def test_parse_single_entry_yaml_accepts_mismatched_top_level_key():
    # Some upstream configs key the entry by a name other than the directory.
    body = parse_single_entry_yaml('other_name:\n  fn_completions: x\n', 'slug')
    assert body == {'fn_completions': 'x'}


def test_parse_single_entry_yaml_rejects_non_mapping():
    with pytest.raises(ValueError):
        parse_single_entry_yaml('- a\n- b\n', 'slug')


def test_leaderboard_urls_point_at_the_pinned_csvs():
    for version, cfg in LEADERBOARDS.items():
        assert cfg['url'] == raw_url(cfg['csv_path'])
        assert version[1:] in cfg['leaderboard_version']


def test_offline_snapshot_without_the_requested_board_is_an_error():
    adapter = _adapter(v2_rows=[_V2_ROW])
    with pytest.raises(ValueError, match='snapshot has no'):
        adapter.fetch_leaderboard_result('v1')


def test_unknown_version_raises():
    with pytest.raises(ValueError, match='Unknown version'):
        AlpacaEvalAdapter().fetch_leaderboard('v99')


# ---------------------------------------------------------------------------
# Record identity
# ---------------------------------------------------------------------------


def test_v1_row_produces_a_log_with_resolved_model_identity():
    logs = _adapter(v1_rows=[_V1_ROW]).fetch_leaderboard('v1')

    assert len(logs) == 1
    log = logs[0]
    assert log.model_info.name == 'gpt4'
    assert log.model_info.id == 'openai/gpt-4'
    assert log.model_info.developer == 'openai'
    details = log.model_info.additional_details
    assert details['identity_source'] == 'vendor_api'
    assert details['leaderboard_slug'] == 'gpt4'
    assert details['pretty_name'] == 'GPT-4'
    assert details['deployment_type'] == 'externally_managed'
    assert details['model_availability'] == 'closed_weights'


def test_eval_library_version_comes_from_upstream_package():
    log = _adapter(v1_rows=[_V1_ROW]).fetch_leaderboard('v1')[0]
    assert log.eval_library.name == 'alpaca_eval'
    # Not the leaderboard version: that is recorded separately.
    assert log.eval_library.version == '0.6.6'
    assert log.eval_library.additional_details['leaderboard_version'] == '1.0'
    assert (
        log.eval_library.additional_details['upstream_ref']
        == DEFAULT_UPSTREAM_REF
    )


def test_evaluation_id_is_stable_and_pins_the_upstream_revision():
    first = _adapter(v1_rows=[_V1_ROW]).fetch_leaderboard('v1')[0]
    second = _adapter(v1_rows=[_V1_ROW]).fetch_leaderboard('v1')[0]

    assert first.evaluation_id == second.evaluation_id
    assert first.evaluation_id == (
        f'alpaca_eval_v1/gpt4@{DEFAULT_UPSTREAM_REF[:12]}'
    )
    assert first.evaluation_results[0].evaluation_result_id == (
        f'{first.evaluation_id}/win_rate'
    )


def test_variant_slugs_stay_distinct_records_for_one_model_id():
    concise = dict(_V2_ROW, **{'': 'gpt4_turbo_concise'})
    adapter = AlpacaEvalAdapter(
        snapshot=UpstreamSnapshot(
            ref=DEFAULT_UPSTREAM_REF,
            package_version='0.6.6',
            leaderboards=_snapshot(v2_rows=[_V2_ROW, concise]).leaderboards,
            model_configs={
                **_MODEL_CONFIGS,
                'gpt4_turbo_concise': dict(_MODEL_CONFIGS['gpt4_turbo']),
            },
        )
    )
    logs = adapter.fetch_leaderboard('v2')

    assert len({log.model_info.id for log in logs}) == 1
    assert len({log.evaluation_id for log in logs}) == 2


def test_model_developer_is_never_unknown_for_a_published_row():
    log = _adapter(v1_rows=[_V1_ROW]).fetch_leaderboard('v1')[0]
    assert log.model_info.developer not in ('', 'unknown', None)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_win_rate_takes_the_registrys_canonical_metric_id():
    log = _adapter(v2_rows=[_V2_ROW]).fetch_leaderboard('v2')[0]
    win_rate = _by_metric(log)['win_rate'].metric_config

    assert win_rate.metric_id == 'win-rate'
    assert win_rate.additional_details['metric_registry_id'] == 'win-rate'
    assert win_rate.additional_details['metric_registry_review_status'] == (
        'reviewed'
    )


def test_metrics_without_a_canonical_keep_a_namespaced_local_id():
    """Three of the four columns have no registry canonical yet.

    They are still published — dropping the length-controlled win rate would
    lose the metric AlpacaEval 2.0 is known for — but under an
    ``alpaca_eval.*`` id that cannot be mistaken for a registry slug, and
    labelled ``no_canonical`` so the gap is visible in the data.
    """
    log = _adapter(v2_rows=[_V2_ROW]).fetch_leaderboard('v2')[0]
    by_metric = _by_metric(log)

    for name, expected_id in (
        ('length_controlled_win_rate', 'alpaca_eval.length_controlled_win_rate'),
        ('discrete_win_rate', 'alpaca_eval.discrete_win_rate'),
        ('avg_length', 'alpaca_eval.avg_length'),
    ):
        config = by_metric[name].metric_config
        assert config.metric_id == expected_id
        assert 'metric_registry_id' not in config.additional_details
        assert config.additional_details['metric_registry_strategy'] == (
            'no_canonical'
        )


def test_evaluation_name_prefers_the_registrys_benchmark_id():
    """A canonical benchmark id is what makes these records joinable.

    AlpacaEval 2.0 has one, AlpacaEval 1.0 does not, so v1 keeps the local name
    and records why.
    """
    v1 = _adapter(v1_rows=[_V1_ROW]).fetch_leaderboard('v1')[0]
    v2 = _adapter(v2_rows=[_V2_ROW]).fetch_leaderboard('v2')[0]

    assert {r.evaluation_name for r in v2.evaluation_results} == {
        'alpacaeval-2-0'
    }
    assert {r.evaluation_name for r in v1.evaluation_results} == {
        'alpaca_eval.v1'
    }
    assert v1.evaluation_results[0].metric_config.additional_details[
        'benchmark_registry_strategy'
    ] == 'no_canonical'


def test_win_rates_are_published_on_the_registrys_scale():
    """The registry declares ``win-rate`` on [0, 100], which is the CSV's scale.

    So the score is the source value verbatim, and nothing is rescaled — see
    ``score_scale_divisor``.
    """
    log = _adapter(v2_rows=[_V2_ROW]).fetch_leaderboard('v2')[0]
    win_rate = _by_metric(log)['win_rate']
    lc = _by_metric(log)['length_controlled_win_rate']

    assert win_rate.metric_config.max_score == 100.0
    assert win_rate.metric_config.additional_details['score_scale_divisor'] == (
        '1.0'
    )
    assert lc.score_details.score == 55.12
    assert lc.metric_config.metric_unit == 'percent'
    assert lc.metric_config.min_score == 0.0
    # No canonical for the length-controlled variant, so its bounds are this
    # adapter's, chosen to match the canonical win rate it is a variant of.
    assert lc.metric_config.max_score == 100.0
    assert (
        lc.metric_config.min_score
        <= lc.score_details.score
        <= lc.metric_config.max_score
    )
    assert lc.score_details.details['source_length_controlled_winrate'] == (
        '55.12'
    )


def test_standard_error_is_analytic_and_on_the_score_scale():
    log = _adapter(v2_rows=[_V2_ROW]).fetch_leaderboard('v2')[0]
    uncertainty = _by_metric(log)['win_rate'].score_details.uncertainty

    assert uncertainty.num_samples == 805
    assert uncertainty.standard_error.value == 0.0
    # Upstream reports pandas' .sem() of the per-instruction preferences, not a
    # bootstrap estimate.
    assert uncertainty.standard_error.method == 'analytic'

    lc = _by_metric(log)['length_controlled_win_rate'].score_details.uncertainty
    assert lc.standard_error.value == 0.72
    assert lc.standard_error.method == 'analytic'


def test_no_standard_error_is_invented_where_the_column_is_absent():
    log = _adapter(v1_rows=[_V1_ROW]).fetch_leaderboard('v1')[0]
    by_metric = _by_metric(log)

    # v1 CSVs carry no lc_standard_error column at all.
    assert by_metric['discrete_win_rate'].score_details.uncertainty is not None
    assert (
        by_metric['discrete_win_rate'].score_details.uncertainty.standard_error
        is None
    )


def test_raw_comparison_counts_are_preserved():
    log = _adapter(v1_rows=[_V1_ROW]).fetch_leaderboard('v1')[0]
    details = _by_metric(log)['win_rate'].score_details.details

    assert details['n_wins'] == '767'
    assert details['n_wins_base'] == '38'
    assert details['n_draws'] == '0'
    assert details['n_total'] == '805'
    assert details['source_win_rate'] == '95.28'


def test_the_description_names_the_rows_own_denominator():
    """Not every row was judged on all 805 instructions.

    The win rate is a share, and a description that always claims 805 misstates
    what a lower-``n_total`` row's score is a share of.
    """
    rows = [
        dict(_V1_ROW, **{'': 'vicuna-7b'}, n_total='648'),
        _V1_ROW,
    ]
    logs = _adapter(v1_rows=rows).fetch_leaderboard('v1')

    assert '648 AlpacaEval instructions' in (
        _by_metric(logs[0])['win_rate'].metric_config.evaluation_description
    )
    assert '805 AlpacaEval instructions' in (
        _by_metric(logs[1])['win_rate'].metric_config.evaluation_description
    )


def test_missing_metric_columns_are_skipped_not_defaulted():
    log = _adapter(v1_rows=[_V1_ROW]).fetch_leaderboard('v1')[0]
    assert 'length_controlled_win_rate' not in _by_metric(log)
    assert 'discrete_win_rate' in _by_metric(log)


def test_avg_length_is_unscaled_characters_and_unbounded_above():
    log = _adapter(v2_rows=[_V2_ROW]).fetch_leaderboard('v2')[0]
    result = _by_metric(log)['avg_length']

    assert result.score_details.score == 2049.0
    assert result.metric_config.metric_unit == 'characters'
    assert result.metric_config.metric_kind == 'length'
    assert result.metric_config.max_score == float('inf')
    # Not a quality metric, and not judge-produced.
    assert result.metric_config.llm_scoring is None
    assert result.score_details.uncertainty is None
    assert 'character' in result.metric_config.evaluation_description.lower()


def test_metric_parameters_record_the_baseline_and_annotator():
    log = _adapter(v2_rows=[_V2_ROW]).fetch_leaderboard('v2')[0]
    params = _by_metric(log)['win_rate'].metric_config.metric_parameters

    assert params['baseline_model'] == 'gpt4_turbo'
    assert params['annotator'] == 'weighted_alpaca_eval_gpt4_turbo'


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def test_judge_is_recorded_with_its_verbatim_prompt():
    log = _adapter(v2_rows=[_V2_ROW]).fetch_leaderboard('v2')[0]
    scoring = _by_metric(log)['win_rate'].metric_config.llm_scoring

    assert scoring.input_prompt == _V2_JUDGE_PROMPT
    assert len(scoring.judges) == 1
    judge = scoring.judges[0]
    assert judge.model_info.id == 'openai/gpt-4-1106-preview'
    assert judge.model_info.developer == 'openai'
    assert judge.temperature == 1.0
    assert judge.additional_details['completion_parser'] == 'logprob_parser'
    assert judge.additional_details['top_logprobs'] == '5'
    assert scoring.additional_details['baseline_model'] == 'gpt4_turbo'


def test_v1_and_v2_use_different_judges_and_prompts():
    v1 = _by_metric(_adapter(v1_rows=[_V1_ROW]).fetch_leaderboard('v1')[0])
    v2 = _by_metric(_adapter(v2_rows=[_V2_ROW]).fetch_leaderboard('v2')[0])

    v1_judge = v1['win_rate'].metric_config.llm_scoring
    v2_judge = v2['win_rate'].metric_config.llm_scoring
    assert v1_judge.judges[0].model_info.id == 'openai/gpt-4'
    assert v1_judge.judges[0].temperature == 0.0
    assert v1_judge.input_prompt == _V1_JUDGE_PROMPT
    assert v2_judge.judges[0].model_info.id == 'openai/gpt-4-1106-preview'
    assert v2_judge.input_prompt == _V2_JUDGE_PROMPT


# ---------------------------------------------------------------------------
# Evaluated data and provenance
# ---------------------------------------------------------------------------


def test_source_data_points_at_the_evaluated_hf_dataset():
    log = _adapter(v2_rows=[_V2_ROW]).fetch_leaderboard('v2')[0]
    source_data = log.evaluation_results[0].source_data

    assert source_data.hf_repo == 'tatsu-lab/alpaca_eval'
    assert source_data.hf_split == 'eval'
    assert source_data.samples_number == 805
    assert source_data.additional_details['hf_config'] == (
        'alpaca_eval_gpt4_baseline'
    )
    assert LEADERBOARDS['v2']['csv_path'] in (
        source_data.additional_details['leaderboard_csv_url']
    )


def test_leaderboard_mode_is_kept_as_provenance():
    log = _adapter(v1_rows=[_V1_ROW]).fetch_leaderboard('v1')[0]
    assert log.source_metadata.additional_details['leaderboard_mode'] == (
        'minimal'
    )
    assert log.source_metadata.source_organization_name.startswith('Tatsu Lab')


def test_the_leaderboards_own_models_are_not_marked_independently_evaluated():
    """``alpaca-7b`` is Tatsu Lab's, and so is the leaderboard.

    Four upstream entries are the evaluator's own models. Marking them
    ``third_party`` asserts an independence that is not there; every other row
    stays ``third_party``.
    """
    snapshot = _snapshot(v1_rows=[dict(_V1_ROW, **{'': 'alpaca-7b'}), _V1_ROW])
    snapshot.model_configs['alpaca-7b'] = {
        'prompt_template': 'vicuna-7b/prompt.txt',
        'fn_completions': 'huggingface_local_completions',
        'completions_kwargs': {'model_name': 'tatsu-lab/alpaca-7b-wdiff'},
        'pretty_name': 'Alpaca 7B',
    }
    own, other = AlpacaEvalAdapter(snapshot=snapshot).fetch_leaderboard('v1')

    assert own.model_info.developer == 'tatsu-lab'
    assert own.source_metadata.evaluator_relationship.value == 'first_party'
    assert other.source_metadata.evaluator_relationship.value == 'third_party'


def test_evaluation_timestamp_is_not_guessed():
    log = _adapter(v1_rows=[_V1_ROW]).fetch_leaderboard('v1')[0]
    # The CSVs carry no per-row date; the fetch time would misdate 2023 rows.
    assert log.evaluation_timestamp is None
    assert log.retrieved_timestamp


def test_generation_config_comes_from_the_upstream_model_config():
    rows = [dict(_V1_ROW, **{'': 'vicuna-7b'})]
    log = _adapter(v1_rows=rows).fetch_leaderboard('v1')[0]
    generation = log.evaluation_results[0].generation_config

    assert generation.generation_args.temperature == 0.7
    assert generation.generation_args.max_tokens == 2048
    assert generation.additional_details['fn_completions'] == (
        'huggingface_local_completions'
    )


def test_prompt_template_carries_the_prompt_not_a_path():
    """A record has to show the prompt the model was actually given.

    The path and its pinned URL stay in ``additional_details`` — they say which
    same-model variant (``_concise``, ``_verbose``) this row is — but an offline
    reader cannot resolve either back into text.
    """
    rows = [dict(_V1_ROW, **{'': 'vicuna-7b'})]
    log = _adapter(v1_rows=rows).fetch_leaderboard('v1')[0]
    generation = log.evaluation_results[0].generation_config

    assert generation.generation_args.prompt_template == (
        'USER: {instruction}\nASSISTANT:'
    )
    assert generation.additional_details['prompt_template_path'] == (
        'vicuna-7b/prompt.txt'
    )
    assert generation.additional_details['prompt_template_url'].endswith(
        'src/alpaca_eval/models_configs/vicuna-7b/prompt.txt'
    )
    assert 'prompt_template_status' not in generation.additional_details


@pytest.mark.parametrize(
    'missing, expected',
    [
        ({}, 'not recorded in this snapshot'),
        ({'gpt4_turbo/chatml_prompt.txt': 'not fetchable: 404'}, '404'),
    ],
)
def test_absent_prompt_text_is_marked_rather_than_faked(missing, expected):
    """Two ways the text can be absent, and neither may look like a prompt.

    A snapshot saved before prompt text was recorded has no ``model_prompts`` at
    all; a template upstream stopped serving is in ``missing_model_prompts``.
    Either way the typed value stays unset and the reason is recorded.
    """
    snapshot = _snapshot(v1_rows=[dict(_V1_ROW, **{'': 'gpt4_turbo'})])
    snapshot.missing_model_prompts.update(missing)
    log = AlpacaEvalAdapter(snapshot=snapshot).fetch_leaderboard('v1')[0]
    generation = log.evaluation_results[0].generation_config

    assert generation.generation_args.prompt_template is None
    assert generation.additional_details['prompt_template_path'] == (
        'gpt4_turbo/chatml_prompt.txt'
    )
    assert expected in generation.additional_details['prompt_template_status']


# ---------------------------------------------------------------------------
# Rejected rows
# ---------------------------------------------------------------------------


def test_null_model_is_excluded_without_failing_the_conversion():
    rows = [_V1_ROW, dict(_V1_ROW, **{'': 'NullModel'})]
    result = _adapter(v1_rows=rows).fetch_leaderboard_result('v1')

    assert len(result.records) == 1
    assert result.failures == []
    assert len(result.exclusions) == 1
    assert result.exclusions[0].source_ref == "CSV row 3 ('NullModel')"
    result.raise_if_incomplete()  # exclusions are deliberate, not failures


def test_unresolvable_identity_is_reported_instead_of_silently_skipped():
    rows = [dict(_V1_ROW, **{'': 'mystery-model'})]
    with pytest.raises(ValueError, match='cannot determine model identity'):
        _adapter(v1_rows=rows).fetch_leaderboard('v1')


def test_partial_result_keeps_valid_rows_and_raw_failure_provenance():
    bad_row = dict(_V1_ROW, **{'': 'mystery-model'})
    result = _adapter(v1_rows=[_V1_ROW, bad_row]).fetch_leaderboard_result('v1')

    assert len(result.records) == 1
    assert len(result.failures) == 1
    assert result.failures[0].source_ref == "CSV row 3 ('mystery-model')"
    assert result.failures[0].source_record == bad_row
    with pytest.raises(SourceRecordsError):
        result.raise_if_incomplete()


def test_row_without_a_win_rate_is_a_failure():
    bad_row = dict(_V1_ROW, **{'': 'gpt4', 'win_rate': ''})
    result = _adapter(v1_rows=[bad_row]).fetch_leaderboard_result('v1')

    assert result.records == []
    assert result.failures[0].reason == 'missing win_rate'


@pytest.mark.parametrize('cell', ['nan', 'inf', '-inf', '1e999', 'n/a'])
def test_a_populated_but_unusable_win_rate_is_a_failure(cell):
    """``float`` accepts most of these, and none of them is a score.

    Letting them through publishes a record whose headline metric is missing, or
    a bare ``NaN`` token that is not valid JSON.
    """
    result = _adapter(
        v1_rows=[dict(_V1_ROW, win_rate=cell)]
    ).fetch_leaderboard_result('v1')

    assert result.records == []
    assert result.failures[0].reason == (
        f'win_rate is not a finite number: {cell!r}'
    )


def test_a_score_outside_the_registrys_bounds_is_a_failure():
    result = _adapter(
        v1_rows=[dict(_V1_ROW, win_rate='150.0')]
    ).fetch_leaderboard_result('v1')

    assert result.records == []
    assert 'outside the [0.0, 100.0]' in result.failures[0].reason


def test_an_absent_secondary_column_is_not_a_failure():
    """Only a populated cell can be invalid — v1 has no LC win rate at all."""
    result = _adapter(v1_rows=[_V1_ROW]).fetch_leaderboard_result('v1')

    assert result.failures == []
    assert len(result.records) == 1


@pytest.mark.parametrize('cell', ['n/a', 'nan', '-0.4'])
def test_an_unusable_standard_error_is_a_failure_not_a_missing_one(cell):
    """Parsing it leniently would publish the score with no uncertainty at all.

    A record that simply omits ``uncertainty`` is indistinguishable from one
    whose source never reported it.
    """
    result = _adapter(
        v2_rows=[dict(_V2_ROW, lc_standard_error=cell)]
    ).fetch_leaderboard_result('v2')

    assert result.records == []
    assert result.failures[0].reason == (
        f'lc_standard_error is not a usable standard error: {cell!r}'
    )


def test_a_standard_error_of_zero_is_kept():
    """The v2 baseline ties with itself on every instruction: SE is really 0."""
    result = _adapter(v2_rows=[_V2_ROW]).fetch_leaderboard_result('v2')
    uncertainty = _by_metric(result.records[0])['win_rate'].score_details

    assert result.failures == []
    assert uncertainty.uncertainty.standard_error.value == 0.0


def test_a_standard_error_without_its_score_does_not_drop_the_row():
    """It is never published, so it cannot be a reason to reject the row."""
    row = dict(_V1_ROW, discrete_win_rate='', standard_error='')
    result = _adapter(
        v1_rows=[dict(row, lc_standard_error='n/a')]
    ).fetch_leaderboard_result('v1')

    assert result.failures == []
    assert 'length_controlled_win_rate' not in _by_metric(result.records[0])


@pytest.mark.parametrize('cell', ['n/a', '0', '-805', '805.5'])
def test_an_unusable_n_total_is_a_failure_not_a_silent_805(cell):
    """It is the denominator the descriptions quote, not an optional extra."""
    result = _adapter(
        v1_rows=[dict(_V1_ROW, n_total=cell)]
    ).fetch_leaderboard_result('v1')

    assert result.records == []
    assert result.failures[0].reason == (
        f'n_total is not a positive count: {cell!r}'
    )


def test_805_is_only_used_where_a_leaderboard_has_no_n_total_column():
    rows = [{key: value for key, value in _V1_ROW.items() if key != 'n_total'}]
    result = _adapter(v1_rows=rows).fetch_leaderboard_result('v1')
    win_rate = _by_metric(result.records[0])['win_rate']

    assert result.failures == []
    assert '805 AlpacaEval instructions' in (
        win_rate.metric_config.evaluation_description
    )
    assert win_rate.score_details.uncertainty.num_samples is None


def test_a_count_that_cannot_be_parsed_keeps_its_source_text():
    """Counts are provenance: reproducing them beats reading them.

    ``_to_int`` would turn ``'767 (est.)'`` into ``None``, and a details mapping
    drops ``None``, so the value would disappear with nothing recorded.
    """
    result = _adapter(
        v1_rows=[dict(_V1_ROW, n_wins='767 (est.)')]
    ).fetch_leaderboard_result('v1')
    details = _by_metric(result.records[0])['win_rate'].score_details.details

    assert result.failures == []
    assert details['n_wins'] == '767 (est.)'


# ---------------------------------------------------------------------------
# Identity ladder
# ---------------------------------------------------------------------------


def test_identity_prefers_the_models_own_hf_repo_over_the_serving_host():
    # Served through Together's API under their own namespace, but the config
    # links the original weights; publishing the host's spelling would split
    # one model across two developers.
    config = {
        'fn_completions': 'openai_completions',
        'completions_kwargs': {
            'model_name': 'togethercomputer/alpaca-7b',
            'client_kwargs': {'base_url': 'https://api.together.xyz'},
        },
        'link': 'https://huggingface.co/tatsu-lab/alpaca-7b-wdiff',
    }
    resolved = identity_mod.resolve_identity('alpaca-7b_concise', config)

    assert resolved.model_id == 'tatsu-lab/alpaca-7b-wdiff'
    assert resolved.identity_source == 'hf_model_link'
    assert resolved.inference_platform == 'api.together.xyz'
    assert resolved.deployment_type == 'externally_managed'


def test_identity_rejects_a_serving_version_reference_as_a_repo_id():
    config = {
        'fn_completions': 'replicate_completions',
        'completions_kwargs': {'model_name': 'replicate/llama70b-v2-chat:e951'},
        'link': 'https://ai.meta.com/llama/',
    }
    resolved = identity_mod.resolve_identity('llama-2-70b-chat-hf', config)

    assert ':' not in resolved.model_id
    assert resolved.model_id == 'meta-llama/llama-2-70b-chat-hf'
    assert resolved.developer == 'meta-llama'
    assert resolved.identity_source == 'vendor_site'
    # The reference actually served is still recorded.
    assert resolved.upstream_model_name.endswith(':e951')


def test_identity_does_not_adopt_a_best_of_n_reward_models_repo():
    config = {
        'link': 'https://huggingface.co/openbmb/UltraRM-13b',
        'prompt_template': 'ultralm-13b-best-of-16/prompt.txt',
    }
    resolved = identity_mod.resolve_identity('ultralm-13b-best-of-16', config)

    assert resolved.model_id == 'openbmb/ultralm-13b-best-of-16'
    assert resolved.identity_source == 'hf_link_org'


def test_identity_reads_the_legacy_openai_api_base_as_local_serving():
    config = {
        'fn_completions': 'openai_completions',
        'completions_kwargs': {
            'openai_api_base': 'http://127.0.0.1:18888/v1',
            'model_name': 'openchat-13b',
        },
        'link': 'https://github.com/imoneoi/openchat',
    }
    resolved = identity_mod.resolve_identity('openchat-13b', config)

    assert resolved.developer == 'imoneoi'
    assert resolved.deployment_type == 'self_deployed'
    assert resolved.model_availability == 'open_weights'


@pytest.mark.parametrize(
    'kwargs,evidence',
    [
        # A torch dtype is an argument to from_pretrained; there is no remote
        # API to pass one to.
        (
            {'model_kwargs': {'torch_dtype': 'bfloat16'}, 'max_new_tokens': 4},
            'model_kwargs.torch_dtype',
        ),
        # transformers.generate() spellings. The API spelling is max_tokens.
        ({'max_new_tokens': 4096}, 'max_new_tokens'),
        ({'max_length': 2048}, 'max_length'),
    ],
)
def test_local_generate_kwargs_settle_a_missing_completions_fn(
    kwargs, evidence
):
    """28 upstream configs record no ``fn_completions``; the kwargs still say.

    ``deployment_type`` is what the model registry uses to tell one deployment
    of a model from another, so publishing ``unknown`` where the config does say
    is a gap rather than caution.
    """
    config = {'completions_kwargs': dict(kwargs, model_name='some/model')}
    resolved = identity_mod.resolve_identity('some-model', config)

    assert resolved.deployment_type == 'self_deployed'
    assert resolved.inference_platform == 'local'
    # The claim carries the kwarg it rests on, and does not invent an engine:
    # the kwargs say the weights were held locally, not what ran them.
    assert resolved.deployment_evidence == evidence
    assert resolved.inference_engine is None


def test_the_literal_string_null_is_read_as_no_completions_fn():
    """``Samba-CoE-v0.1`` spells an absent ``fn_completions`` as ``null``.

    Reading only a missing key and an empty string leaves that one looking like
    a completions function this converter has never heard of, so the entry keeps
    ``deployment_type: unknown`` even though its kwargs settle it.
    """
    config = {
        'fn_completions': 'null',
        'completions_kwargs': {
            'model_name': 'sambanovasystems/Samba-CoE-v0.1',
            'model_kwargs': {'torch_dtype': 'bfloat16'},
            'max_new_tokens': 4096,
        },
    }
    resolved = identity_mod.resolve_identity('Samba-CoE-v0.1', config)

    assert resolved.deployment_type == 'self_deployed'


def test_an_api_shaped_request_stays_unknown():
    """``max_tokens`` says a request was made, not to whom.

    The MoA entries (``Together-MoA``, ``TOA``, ``blendaxai-…``) are all this
    shape. Reading the API spelling as local serving would claim these ran on
    hardware the submitter controlled, which the config does not say.
    """
    config = {
        'completions_kwargs': {
            'model_name': 'Together-MoA',
            'max_tokens': 2048,
        },
        'link': 'https://github.com/togethercomputer/MoA',
    }
    resolved = identity_mod.resolve_identity('Together-MoA', config)

    assert identity_mod.local_generate_evidence(config) is None
    assert resolved.deployment_type == 'unknown'
    assert resolved.inference_platform is None
    assert resolved.deployment_evidence is None


def test_running_weights_locally_does_not_make_them_public():
    """The Humpback checkpoints were run locally and never released.

    ``deployment_type`` and ``model_availability`` are independent axes, so the
    kwargs-based rule sets the first and leaves the second to the link evidence.
    """
    config = {
        'completions_kwargs': {
            'model_name': 'humpback-llama-65b',
            'max_length': 2048,
        },
        'link': 'https://arxiv.org/abs/2308.06259',
    }
    resolved = identity_mod.resolve_identity('humpback-llama-65b', config)

    assert resolved.deployment_type == 'self_deployed'
    assert resolved.model_availability == 'unknown'


def test_identity_returns_none_without_evidence():
    assert identity_mod.resolve_identity('mystery-model', None) is None


def test_identity_takes_repo_casing_from_the_entrys_own_link():
    # Upstream types `baai/…` in the config but links the real `BAAI/…` repo;
    # keeping the config's spelling would publish a repo id that does not exist
    # and a second spelling of an organization already present as `BAAI`.
    config = {
        'completions_kwargs': {'model_name': 'baai/Infinity-Instruct-7M'},
        'link': 'https://huggingface.co/BAAI/Infinity-Instruct-7M',
    }
    resolved = identity_mod.resolve_identity('Infinity-Instruct-7M', config)

    assert resolved.model_id == 'BAAI/Infinity-Instruct-7M'
    assert resolved.developer == 'BAAI'
    assert resolved.upstream_model_name == 'baai/Infinity-Instruct-7M'


def test_identity_borrows_a_sibling_entrys_link_for_casing():
    # The best-of-16 row links its *reward* model, so its own config offers no
    # evidence for the served model's casing. A sibling entry's link does.
    served = {
        'completions_kwargs': {'model_name': '01-ai/Yi-34B-Chat'},
        'link': 'https://huggingface.co/01-ai/Yi-34B-Chat',
    }
    best_of = {
        'completions_kwargs': {'model_name': '01-ai/Yi-34b-Chat'},
        'link': 'https://huggingface.co/llm-blender/PairRM',
    }
    casing = identity_mod.canonical_repo_casing([served, best_of])

    assert (
        identity_mod.resolve_identity('pairrm-Yi-34B-Chat', best_of).model_id
        == '01-ai/Yi-34b-Chat'
    )
    assert (
        identity_mod.resolve_identity(
            'pairrm-Yi-34B-Chat', best_of, casing
        ).model_id
        == '01-ai/Yi-34B-Chat'
    )


def test_identity_casing_map_does_not_rename_a_different_repo():
    config = {
        'completions_kwargs': {'model_name': 'allenai/tulu-2-dpo-13b'},
        'link': 'https://huggingface.co/llm-blender/PairRM',
    }
    casing = {'allenai/tulu-2-dpo-70b': 'allenai/tulu-2-dpo-70b'}
    resolved = identity_mod.resolve_identity(
        'pairrm-tulu-2-13b', config, casing
    )

    assert resolved.model_id == 'allenai/tulu-2-dpo-13b'


# ---------------------------------------------------------------------------
# HuggingFace repo renames
# ---------------------------------------------------------------------------


def test_renamed_repo_is_published_under_the_id_hf_serves_today():
    """A stale id still redirects on HuggingFace but joins with nothing."""
    config = {
        'completions_kwargs': {'model_name': 'WizardLM/WizardLM-70B-V1.0'},
        'link': 'https://huggingface.co/WizardLM/WizardLM-70B-V1.0',
    }
    resolved = identity_mod.resolve_identity('wizardlm-70b', config)

    assert resolved.model_id == 'WizardLMTeam/WizardLM-70B-V1.0'
    # The organization the source names is the one that published the model; an
    # HTTP redirect cannot tell a rename from a transfer, so `developer` stays.
    assert resolved.developer == 'WizardLM'
    assert resolved.model_id_as_referenced == 'WizardLM/WizardLM-70B-V1.0'


def test_rename_lookup_is_case_insensitive_like_huggingface():
    config = {'link': 'https://huggingface.co/thudm/chatglm2-6b'}
    resolved = identity_mod.resolve_identity('chatglm2-6b', config)

    assert resolved.model_id == 'zai-org/chatglm2-6b'


def test_renames_can_be_switched_off_for_verbatim_source_ids():
    config = {
        'completions_kwargs': {'model_name': 'WizardLM/WizardLM-70B-V1.0'},
        'link': 'https://huggingface.co/WizardLM/WizardLM-70B-V1.0',
    }
    resolved = identity_mod.resolve_identity(
        'wizardlm-70b', config, hf_canonical={}
    )

    assert resolved.model_id == 'WizardLM/WizardLM-70B-V1.0'
    assert resolved.model_id_as_referenced is None


def test_a_vendor_site_publishes_the_registrys_huggingface_namespace():
    """``ai.meta.com`` names Meta, whose repos live under ``meta-llama``.

    ``meta`` hosts no Llama repo, so an id built out of the website's own name
    joins with nothing. The namespace the registry records for the organization
    replaces it, and the derived spelling stays in ``model_id_as_referenced``.
    """
    config = {'link': 'https://ai.meta.com/llama/'}
    resolved = identity_mod.resolve_identity('llama-2-70b-chat-hf', config)

    assert resolved.identity_source == 'vendor_site'
    assert resolved.model_id == 'meta-llama/llama-2-70b-chat-hf'
    assert resolved.developer == 'meta-llama'
    assert resolved.model_id_as_referenced == 'meta/llama-2-70b-chat-hf'


def test_the_namespace_lift_can_be_switched_off():
    config = {'link': 'https://ai.meta.com/llama/'}
    resolved = identity_mod.resolve_identity(
        'llama-2-70b-chat-hf', config, hf_namespaces={}
    )

    assert resolved.model_id == 'meta/llama-2-70b-chat-hf'
    assert resolved.developer == 'meta'
    assert resolved.model_id_as_referenced is None


def test_a_vendor_site_keeps_its_name_where_no_namespace_is_recorded():
    """An organization with no HuggingFace presence has nothing to lift to.

    The website's name is then the only spelling there is, so it is published as
    derived rather than replaced by a pattern guess.
    """
    config = {'link': 'https://ai.meta.com/llama/'}
    resolved = identity_mod.resolve_identity(
        'llama-2-70b-chat-hf', config, hf_namespaces={'alibaba': 'qwen'}
    )

    assert resolved.developer == 'meta'
    assert resolved.model_id_as_referenced is None


def test_the_namespace_lift_is_confined_to_the_vendor_site_rung():
    """A HuggingFace link already names a namespace that resolves.

    Rewriting it would trade a spelling the source read off HuggingFace for one
    derived from an organization id.
    """
    config = {'link': 'https://huggingface.co/meta/llama-2-70b-chat-hf'}
    resolved = identity_mod.resolve_identity(
        'llama-2-70b-chat-hf', config, hf_namespaces={'meta': 'meta-llama'}
    )

    assert resolved.identity_source == 'hf_model_link'
    assert resolved.model_id == 'meta/llama-2-70b-chat-hf'
    assert resolved.developer == 'meta'


def test_a_constructed_id_is_never_canonicalized_against_huggingface():
    """``cohere/command-nightly`` is a constructed id, not a repo id.

    HuggingFace redirects it to ``CohereLabs/Command-nightly``, but this row was
    served by Cohere's API under a rolling alias. Adopting the repo id would
    assert an equivalence nobody checked and contradict the record's own
    ``model_availability: closed_weights``.
    """
    config = {
        'fn_completions': 'cohere_completions',
        'completions_kwargs': {'model_name': 'command-nightly'},
    }
    renames = {'cohere/command-nightly': 'CohereLabs/Command-nightly'}
    resolved = identity_mod.resolve_identity(
        'cohere', config, hf_canonical=renames
    )

    assert resolved.identity_source == 'vendor_api'
    assert resolved.model_id == 'cohere/command-nightly'
    assert resolved.model_id_as_referenced is None


def test_rename_map_is_a_fixed_point():
    """Applying the map twice must change nothing.

    A value that is also a key would mean a chained rename, so which id gets
    published would depend on iteration order.
    """
    renames = identity_mod.hf_canonical_ids()

    assert renames
    assert all(key == key.lower() for key in renames)
    assert not set(renames.values()) & set(renames)


def test_rename_map_is_pinned_to_the_upstream_ref_it_was_built_from():
    """The map's keys are ids *this* upstream ref publishes.

    Moving the pinned ref changes the set of ids, so the map has to be refreshed
    with it — otherwise a new stale id is published with no signal that anything
    was missed.
    """
    payload = json.loads(
        identity_mod.HF_CANONICAL_PATH.read_text(encoding='utf-8')
    )

    assert payload['_meta']['upstream_ref'] == DEFAULT_UPSTREAM_REF
    assert payload['_meta']['retrieved_date']
    assert payload['_meta']['counts']['renamed'] == len(
        payload['renamed_repos']
    )


def test_published_record_reports_the_namespace_and_the_referenced_id():
    row = dict(_V2_ROW, **{'': 'wizardlm-13b-v1.2'})
    log = _adapter(v2_rows=[row]).fetch_leaderboard('v2')[0]
    details = log.model_info.additional_details

    assert log.model_info.id == 'WizardLMTeam/WizardLM-13B-V1.2'
    assert details['raw_model_namespace'] == 'WizardLMTeam'
    assert details['raw_model_id'] == 'WizardLMTeam/WizardLM-13B-V1.2'
    assert details['model_id_as_referenced'] == 'WizardLM/WizardLM-13B-V1.2'
    # The registry canonicalizes the organization the source named, not the
    # namespace the repo has since moved to.
    assert log.model_info.developer == 'wizardlm'


# ---------------------------------------------------------------------------
# Pinning the upstream ref
# ---------------------------------------------------------------------------


def test_an_immutable_ref_is_not_looked_up():
    def _no_network(*_args, **_kwargs):
        raise AssertionError('a 40-hex ref is already a commit')

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(upstream_mod, 'fetch_text', _no_network)

        assert resolve_ref(DEFAULT_UPSTREAM_REF) == DEFAULT_UPSTREAM_REF
        assert resolve_ref(f'  {DEFAULT_UPSTREAM_REF.upper()}  ') == (
            DEFAULT_UPSTREAM_REF
        )


def test_a_moving_ref_is_pinned_to_the_commit_it_names():
    """Two runs a week apart must not publish different input as one identity."""
    sha = 'a' * 40
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            upstream_mod,
            'fetch_text',
            lambda url, **_kwargs: json.dumps({'sha': sha, 'url': url}),
        )

        assert resolve_ref('main') == sha


@pytest.mark.parametrize('payload', ['not json', '{}', '{"sha": "abc"}'])
def test_a_ref_that_names_no_commit_fails_before_any_artefact_is_fetched(
    payload,
):
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            upstream_mod, 'fetch_text', lambda *_a, **_k: payload
        )

        with pytest.raises(FetchError):
            resolve_ref('no-such-branch')


def test_populate_fetches_each_prompt_template_once():
    """Distinct templates only — sibling variants share one file.

    A template upstream no longer serves is recorded rather than raised: a
    leaderboard is still convertible without one model's prompt.
    """
    snapshot = _snapshot(v1_rows=[_V1_ROW])
    snapshot.model_prompts.clear()
    snapshot.model_configs['gpt4_variant'] = dict(
        _MODEL_CONFIGS['gpt4'], pretty_name='GPT-4 (again)'
    )
    fetched = []

    def _fetch_text(url, **_kwargs):
        fetched.append(url)
        if 'gpt4/chatml_prompt.txt' in url:
            raise FetchError('404')
        return f'text of {url.rsplit("models_configs/", 1)[-1]}'

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(upstream_mod, 'fetch_text', _fetch_text)
        upstream_mod.populate_snapshot(
            snapshot, {'v1': LEADERBOARDS['v1']}, lambda rows: []
        )

    distinct = {
        config['prompt_template'] for config in snapshot.model_configs.values()
    }

    assert len(fetched) == len(set(fetched)) == len(distinct)
    assert snapshot.model_prompts['vicuna-7b/prompt.txt'] == (
        'text of vicuna-7b/prompt.txt'
    )
    assert '404' in snapshot.missing_model_prompts['gpt4/chatml_prompt.txt']


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_output_dir_defaults_to_a_throwaway_path():
    args = cli.build_parser().parse_args(['convert', 'alpaca_eval'])
    # A network-fetching source must not write a data/ tree into the cwd.
    assert args.output_dir != 'data'
    assert 'alpaca-eval-smoke' in args.output_dir


def test_two_default_runs_do_not_pool_their_output(snapshot_file, tmp_path):
    """Each record is named with a fresh UUID, so a fixed directory piles up.

    A reader of the throwaway output could not tell this run's records from last
    week's, and the second run's report would be counted against both.
    """
    snapshot_path = snapshot_file(v1_rows=[_V1_ROW])
    argv = [
        'convert', 'alpaca_eval',
        '--version', 'v1',
        '--input-json', str(snapshot_path),
        '--no-registry-resolve',
    ]

    made = []

    def _mkdtemp(**_kwargs):
        run = tmp_path / f'run{len(made)}'
        run.mkdir()
        made.append(run)
        return str(run)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(cli.tempfile, 'mkdtemp', _mkdtemp)
        assert cli.main(argv) == 0
        assert cli.main(argv) == 0

    assert len(made) == 2
    for run in made:
        assert len(list((run / 'data').rglob('*.json'))) == 1


def test_cli_converts_from_a_snapshot_without_network(snapshot_file, tmp_path):
    snapshot_path = snapshot_file(v1_rows=[_V1_ROW], v2_rows=[_V2_ROW])
    output_dir = tmp_path / 'data'

    exit_code = cli.main(
        [
            'convert',
            'alpaca_eval',
            '--input-json',
            str(snapshot_path),
            '--output-dir',
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert len(list(output_dir.rglob('*.json'))) == 2
    assert (output_dir / 'alpaca_eval_v1' / 'openai' / 'gpt-4').is_dir()


def test_cli_reports_what_the_live_registry_actually_did(
    snapshot_file, tmp_path, capsys
):
    """The status line printed before conversion can only ever report zeros.

    It announces the configuration before a multi-minute fetch, so the lookups it
    counts have not happened yet.
    """

    def _post(url, json=None, timeout=None):
        raise OSError('registry unreachable')

    requests = pytest.importorskip('requests')
    original = requests.post
    requests.post = _post
    try:
        cli.main(
            [
                'convert',
                'alpaca_eval',
                '--version',
                'v1',
                '--input-json',
                str(snapshot_file(v1_rows=[_V1_ROW])),
                '--output-dir',
                str(tmp_path / 'data'),
                '--registry-live',
            ]
        )
    finally:
        requests.post = original

    reported = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith('eval-card-registry live lookups')
    ]
    assert len(reported) == 1
    assert '0 queries' not in reported[0]
    assert '0 resolved' in reported[0]
    assert 'registry unreachable' in reported[0]


def test_the_live_lookup_summary_is_only_printed_for_a_live_run(
    snapshot_file, tmp_path, capsys
):
    """Off by default, so the line would be three zeros about nothing."""
    cli.main(
        [
            'convert',
            'alpaca_eval',
            '--version',
            'v1',
            '--input-json',
            str(snapshot_file(v1_rows=[_V1_ROW])),
            '--output-dir',
            str(tmp_path / 'data'),
        ]
    )

    assert 'live lookups' not in capsys.readouterr().out


def test_module_entry_point_accepts_every_option_the_handler_reads(
    snapshot_file, tmp_path
):
    """``python -m ...converters.alpaca_eval`` shares the top-level parser.

    It used to build its own namespace, which meant each option added to the
    handler — ``--ref``, ``--input-json``, the registry switches — raised
    ``AttributeError`` here before any conversion happened.
    """
    from every_eval_ever.converters.alpaca_eval.__main__ import (
        main as module_main,
    )

    output_dir = tmp_path / 'data'
    exit_code = module_main(
        [
            '--version',
            'v1',
            '--input-json',
            str(snapshot_file(v1_rows=[_V1_ROW])),
            '--output-dir',
            str(output_dir),
            '--no-registry-resolve',
        ]
    )

    assert exit_code == 0
    assert len(list(output_dir.rglob('*.json'))) == 1


def test_module_entry_point_reads_the_command_line():
    """``python -m`` passes nothing to ``main``; the options are in ``sys.argv``.

    Defaulting to an empty list instead silently converted both leaderboards to
    a throwaway directory, whatever was asked for.
    """
    from every_eval_ever.converters.alpaca_eval import __main__ as module

    seen = []
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(module.sys, 'argv', ['prog', '--version', 'v1'])
        patcher.setattr(cli, 'main', lambda argv: seen.append(argv) or 0)

        assert module.main() == 0

    assert seen == [['convert', 'alpaca_eval', '--version', 'v1']]


def test_cli_save_raw_json_writes_a_replayable_snapshot(
    snapshot_file, tmp_path
):
    snapshot_path = snapshot_file(v1_rows=[_V1_ROW])
    saved = tmp_path / 'raw' / 'upstream.json'

    cli.main(
        [
            'convert',
            'alpaca_eval',
            '--version',
            'v1',
            '--input-json',
            str(snapshot_path),
            '--save-raw-json',
            str(saved),
            '--output-dir',
            str(tmp_path / 'data'),
        ]
    )

    payload = json.loads(saved.read_text(encoding='utf-8'))
    assert payload['ref'] == DEFAULT_UPSTREAM_REF
    assert payload['leaderboards']['v1']['rows'][0]['win_rate'] == '95.28'
    assert UpstreamSnapshot.from_payload(payload).package_version == '0.6.6'


def test_cli_publishes_valid_rows_before_signaling_partial_failure(
    snapshot_file, tmp_path
):
    bad_row = dict(_V1_ROW, **{'': 'mystery-model'})
    snapshot_path = snapshot_file(v1_rows=[_V1_ROW, bad_row])
    output_dir = tmp_path / 'data'

    with pytest.raises(SourceRecordsError):
        cli.main(
            [
                'convert',
                'alpaca_eval',
                '--version',
                'v1',
                '--input-json',
                str(snapshot_path),
                '--output-dir',
                str(output_dir),
            ]
        )

    assert len(list(output_dir.rglob('*.json'))) == 1
    report_path = tmp_path / 'adapter_reports' / 'alpaca_eval_v1_failures.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['converted_records'] == 1
    assert report['failed_record_count'] == 1
    assert report['failed_records'][0]['source_record'] == bad_row


def test_cli_publishes_other_version_when_one_conversion_fails(
    snapshot_file, tmp_path
):
    snapshot_path = snapshot_file(v1_rows=[_V1_ROW], v2_rows=[_V2_ROW])
    successful = _adapter(v2_rows=[_V2_ROW]).fetch_leaderboard_result('v2')

    def fetch_version(_self, version):
        if version == 'v1':
            raise RuntimeError('upstream unavailable')
        return successful

    output_dir = tmp_path / 'data'
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            AlpacaEvalAdapter, 'fetch_leaderboard_result', fetch_version
        )
        with pytest.raises(SourceRecordsError, match='upstream unavailable'):
            cli.main(
                [
                    'convert',
                    'alpaca_eval',
                    '--input-json',
                    str(snapshot_path),
                    '--output-dir',
                    str(output_dir),
                ]
            )

    assert len(list(output_dir.rglob('*.json'))) == len(successful.records)
    report = json.loads(
        (
            tmp_path / 'adapter_reports' / 'alpaca_eval_v1_failures.json'
        ).read_text(encoding='utf-8')
    )
    assert report['failed_record_count'] == 1
    assert report['failed_records'][0]['source_record']['version'] == 'v1'


# ---------------------------------------------------------------------------
# eval-card-registry resolution
# ---------------------------------------------------------------------------


def test_developer_is_the_registrys_canonical_organization():
    """``meta-llama`` publishes the repo; ``meta`` is the organization.

    The registry records both as identities for Meta, so the id keeps the
    namespace that resolves on HuggingFace and the developer names the company.
    """
    row = dict(_V2_ROW, **{'': 'Meta-Llama-3-70B-Instruct'})
    log = _adapter(
        v2_rows=[row],
    ).fetch_leaderboard('v2')[0]

    assert log.model_info.id.startswith('meta-llama/')
    assert log.model_info.developer == 'meta'
    details = log.model_info.additional_details
    assert details['raw_model_namespace'] == 'meta-llama'
    assert details['developer_registry_id'] == 'meta'
    assert details['developer_registry_review_status'] == 'reviewed'


def test_registry_can_be_switched_off_without_losing_provenance():
    """``--no-registry-resolve`` must be visible in the record, not silent."""
    disabled = registry_mod.Registry(enabled=False)
    log = _adapter(v2_rows=[_V2_ROW], registry=disabled).fetch_leaderboard(
        'v2'
    )[0]
    win_rate = _by_metric(log)['win_rate'].metric_config

    assert win_rate.metric_id == 'alpaca_eval.win_rate'
    assert win_rate.additional_details['metric_registry_strategy'] == (
        'registry_disabled'
    )
    assert log.model_info.additional_details[
        'developer_registry_strategy'
    ] == 'registry_disabled'
    # Falls back to the source-derived spelling rather than to nothing.
    assert log.model_info.developer == log.model_info.id.split('/')[0]


def test_disabled_registry_still_declares_usable_bounds():
    """Bounds cannot depend on the registry being reachable."""
    log = _adapter(
        v2_rows=[_V2_ROW], registry=registry_mod.Registry(enabled=False)
    ).fetch_leaderboard('v2')[0]

    for result in log.evaluation_results:
        config = result.metric_config
        assert config.min_score == 0.0
        assert config.min_score <= result.score_details.score
        assert result.score_details.score <= config.max_score


# ---------------------------------------------------------------------------
# Canonical-id refresh
# ---------------------------------------------------------------------------

_WIZARDLM_REPO = 'WizardLM/WizardLM-13B-V1.2'
_VICUNA_REPO = 'lmsys/vicuna-7b-delta-v1.1'
_RENAMED_MAP = {_WIZARDLM_REPO: 'WizardLMTeam/WizardLM-13B-V1.2'}


def _refresh_run(snapshot_file, tmp_path, answers, argv=()):
    """Run the refresh against a canned ``{repo_id: (status, id)}`` sweep."""
    snapshot_path = snapshot_file(
        v1_rows=[
            dict(_V1_ROW, **{'': 'wizardlm-13b-v1.2'}),
            dict(_V1_ROW, **{'': 'vicuna-7b'}),
        ]
    )
    output = tmp_path / identity_mod.HF_CANONICAL_NAME
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(
            refresh_mod, 'hf_canonical_id', lambda repo_id: answers[repo_id]
        )
        exit_code = refresh_mod.main(
            [
                '--upstream-snapshot',
                str(snapshot_path),
                '--output',
                str(output),
                *argv,
            ]
        )
    return exit_code, output


def test_a_transient_failure_leaves_the_committed_map_untouched(
    snapshot_file, tmp_path, capsys
):
    """The map is rebuilt from scratch, so writing an incomplete sweep deletes.

    A 429 or a 5xx for the WizardLM id would drop its confirmed rename, and the
    adapter would go back to publishing the stale id with nothing to show why.
    """
    exit_code, output = _refresh_run(
        snapshot_file,
        tmp_path,
        {_WIZARDLM_REPO: (503, None), _VICUNA_REPO: (200, _VICUNA_REPO)},
    )

    assert exit_code == 1
    assert not output.exists()
    assert f'{_WIZARDLM_REPO} (503)' in capsys.readouterr().err


def test_check_does_not_report_a_stale_map_from_an_incomplete_sweep(
    snapshot_file, tmp_path, capsys
):
    """Otherwise ``--check`` advises the refresh that does the deleting."""
    committed = json.dumps({'renamed_repos': _RENAMED_MAP}) + '\n'
    (tmp_path / identity_mod.HF_CANONICAL_NAME).write_text(
        committed, encoding='utf-8'
    )
    exit_code, output = _refresh_run(
        snapshot_file,
        tmp_path,
        {_WIZARDLM_REPO: (429, None), _VICUNA_REPO: (200, _VICUNA_REPO)},
        argv=('--check',),
    )

    assert exit_code == 1
    assert output.read_text(encoding='utf-8') == committed
    stderr = capsys.readouterr().err
    assert 'did not answer' in stderr
    assert 'differs from HuggingFace' not in stderr


def test_a_gated_repo_is_still_a_complete_sweep(snapshot_file, tmp_path):
    """401 is HuggingFace's answer, not a missing one — it must not block."""
    exit_code, output = _refresh_run(
        snapshot_file,
        tmp_path,
        {
            _WIZARDLM_REPO: (200, _RENAMED_MAP[_WIZARDLM_REPO]),
            _VICUNA_REPO: (401, None),
        },
    )
    written = json.loads(output.read_text(encoding='utf-8'))

    assert exit_code == 0
    assert written['renamed_repos'] == _RENAMED_MAP
    assert written['_meta']['unverifiable_repo_ids'] == [_VICUNA_REPO]


def test_a_later_401_does_not_drop_a_previously_confirmed_rename(
    snapshot_file, tmp_path
):
    """A gate today does not undo a rename HuggingFace already confirmed.

    The map is rebuilt from scratch every run, so without carrying the prior
    entry forward, a repo that answered 200 last time and 401 this time would
    silently lose its rename and the adapter would go back to the stale id.
    """
    (tmp_path / identity_mod.HF_CANONICAL_NAME).write_text(
        json.dumps({'renamed_repos': _RENAMED_MAP}) + '\n', encoding='utf-8'
    )

    exit_code, output = _refresh_run(
        snapshot_file,
        tmp_path,
        {_WIZARDLM_REPO: (401, None), _VICUNA_REPO: (200, _VICUNA_REPO)},
    )
    written = json.loads(output.read_text(encoding='utf-8'))

    assert exit_code == 0
    assert written['renamed_repos'] == _RENAMED_MAP
    # Retained via the prior map, not freshly confirmed - not unverifiable.
    assert _WIZARDLM_REPO not in written['_meta']['unverifiable_repo_ids']


def test_a_200_overrides_a_stale_prior_mapping(snapshot_file, tmp_path):
    """A fresh 200 is authoritative even over a previously confirmed rename."""
    (tmp_path / identity_mod.HF_CANONICAL_NAME).write_text(
        json.dumps({'renamed_repos': _RENAMED_MAP}) + '\n', encoding='utf-8'
    )
    second_rename = 'WizardLMTeam/WizardLM-13B-V1.2-renamed-again'

    exit_code, output = _refresh_run(
        snapshot_file,
        tmp_path,
        {
            _WIZARDLM_REPO: (200, second_rename),
            _VICUNA_REPO: (200, _VICUNA_REPO),
        },
    )
    written = json.loads(output.read_text(encoding='utf-8'))

    assert exit_code == 0
    assert written['renamed_repos'][_WIZARDLM_REPO] == second_rename
