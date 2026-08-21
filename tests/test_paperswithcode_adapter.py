"""Offline, fixture-based tests for the Papers with Code adapter.

No network and no PostgreSQL: the pure builders take plain dicts (the same shape
``pgdumplib`` yields), so the ``core`` CI matrix runs these without ``pgdumplib``.
"""

from __future__ import annotations

import json

import pytest

from every_eval_ever.adapters.paperswithcode import adapter
from every_eval_ever.validate import validate_file

RETRIEVED_TS = '1700000000.0'
DUMP_VERSION = '20260716'


def _datasets():
    return {
        # hf_dataset source
        '218': {
            'id': '218',
            'name': 'ETH3D (relative)',
            'slug': 'eth3d-relative',
            'hf_url': 'https://huggingface.co/datasets/ritianyu/eth3d_tar',
            'url': 'https://www.eth3d.net/schoeps2017cvpr.pdf',
            'paper_url': 'https://arxiv.org/abs/1704.00001',
            'introduced_year': '2017',
        },
        # url source (no hf_url)
        '906': {
            'id': '906',
            'name': 'RealEstate10K (2-view)',
            'slug': 're10k-2-view',
            'hf_url': None,
            'url': 'https://google.github.io/realestate10k/',
            'paper_url': None,
        },
        # private/other source (no urls and no slug -> unreachable in real PwC
        # data, but the code path must still produce a valid record)
        '999': {
            'id': '999',
            'name': 'Secret Bench',
            'slug': None,
            'hf_url': None,
            'url': None,
            'paper_url': None,
        },
    }


def _tasks():
    return {
        '10': {'id': '10', 'slug': 'depth-estimation'},
        '20': {'id': '20', 'slug': 'novel-view-synthesis'},
        '30': {'id': '30', 'slug': 'secret-task'},
    }


def _metric_dir():
    return {
        'AbsRel': 'lower_is_better',
        'delta1': 'higher_is_better',
        'PSNR': 'higher_is_better',
        'SSIM': 'higher_is_better',
        'Accuracy': 'higher_is_better',
        'CustomZMetric': 'higher_is_better',  # not in the registry snapshot
    }


def _metric_meta():
    return {
        'AbsRel': {'full_name': 'Absolute Relative Error', 'scale': '0-1'},
        'delta1': {'full_name': 'Delta < 1.25', 'scale': '0-1'},
        'PSNR': {
            'full_name': 'Peak Signal-to-Noise Ratio',
            'scale': 'unbounded',
        },
        'Accuracy': {'full_name': 'Accuracy', 'scale': None},
    }


def _metric_ranges():
    return {
        'AbsRel': (0.008, 0.794),
        'delta1': (0.629, 0.993),
        'PSNR': (13.5, 41.22),
        'SSIM': (0.39, 0.9831),
        'Accuracy': (0.0, 100.0),
        'CustomZMetric': (0.0, 1.0),
    }


def _evaluations():
    return [
        # open model (hf url), multi-metric row: delta1 + SSIM (both bounded) for
        # fan-out, plus AbsRel (unbounded -> emitted with inf)
        {
            'id': '11533',
            'paper_id': '900',
            'task_id': '10',
            'dataset_id': '218',
            'model_name': 'MoGe-2',
            'metrics': json.dumps(
                {'delta1': '0.991', 'SSIM': '0.85', 'AbsRel': '0.028'}
            ),
            'evaluated_on': '2026-07-06',
            'created_at': '2026-06-18 13:13:06+00',
            'hf_model_url': 'https://huggingface.co/Ruicheng/moge-2-vitl',
            'num_parameters': '326000000',
            'is_open': 't',
            'external': 'f',
            'harness': None,
            'best_rank': '3',
        },
        # url dataset, PSNR (unbounded -> inf) + a made-up metric that is not in
        # the registry (unresolved path).
        {
            'id': '13524',
            'paper_id': None,
            'task_id': '20',
            'dataset_id': '906',
            'model_name': 'StructSplat',
            'metrics': json.dumps({'PSNR': '22.240', 'CustomZMetric': '0.5'}),
            'evaluated_on': None,
            'created_at': '2026-07-14 00:00:00+00',
            'hf_model_url': 'https://huggingface.co/nyu-vision/structsplat',
            'is_open': 't',
            'external': 'f',
            'harness': 'Not reported',
        },
        # a research method with no HF url whose developer the shared helper does
        # not know: there is no organization to name, so the row is a failure
        # rather than a record filed under a placeholder developer.
        {
            'id': '13525',
            'paper_id': None,
            'task_id': '20',
            'dataset_id': '906',
            'model_name': 'AnonSplat-XL',
            'metrics': json.dumps({'PSNR': '19.8'}),
            'evaluated_on': None,
            'created_at': '2026-07-14 00:00:00+00',
            'hf_model_url': None,
            'is_open': 't',
            'external': 'f',
            'harness': None,
        },
        # European decimal comma + a known LLM name -> developer from helper
        {
            'id': '20001',
            'paper_id': None,
            'task_id': '10',
            'dataset_id': '218',
            'model_name': 'GPT-5.5 Pro (xhigh)',
            'metrics': json.dumps({'Accuracy': '97,3'}),
            'evaluated_on': None,
            'created_at': '2026-07-01 00:00:00+00',
            'hf_model_url': None,
            'is_open': 'f',
            'external': 't',
            'harness': 'SWE-agent',
        },
    ]


def _papers():
    return {
        '900': {
            'arxiv_id': '2507.02546',
            'title': 'MoGe-2',
            'source_url': None,
        }
    }


def _make_resolver():
    return adapter.MetricResolver(pwc_directions=_metric_dir())


def _resolved_metric(**over):
    fields = {
        'metric_id': 'accuracy',
        'metric_kind': 'accuracy',
        'lower_is_better': False,
        'score_type': 'continuous',
        'min_score': 0.0,
        'max_score': 1.0,
        'resolved': True,
        'detail': {},
    }
    return adapter.ResolvedMetric(**{**fields, **over})


def _convert(resolver=None, evaluations=None):
    resolver = resolver or _make_resolver()
    return adapter.build_logs(
        _evaluations() if evaluations is None else evaluations,
        _datasets(),
        _tasks(),
        resolver,
        _metric_ranges(),
        _metric_meta(),
        _papers(),
        DUMP_VERSION,
        RETRIEVED_TS,
    )


def _build(resolver=None, evaluations=None):
    return _convert(resolver, evaluations).records


def test_parse_metric_value_edge_cases():
    assert adapter.parse_metric_value('95.2') == (95.2, None)
    assert adapter.parse_metric_value('30%') == (30.0, None)
    # European decimal comma, not thousands separator
    assert adapter.parse_metric_value('97,3') == (97.3, None)
    # >2 digits after the comma are still a decimal comma (regression: the old
    # rule only accepted 1-2 digits and mangled these into integers)
    assert adapter.parse_metric_value('0,991') == (0.991, None)
    assert adapter.parse_metric_value('97,345') == (97.345, None)
    # thousands separator: comma alongside a '.', or several commas
    assert adapter.parse_metric_value('1,234.5') == (1234.5, None)
    assert adapter.parse_metric_value('1,234,567') == (1234567.0, None)
    # non-finite values are rejected -- a score is a finite real, not a bound
    assert adapter.parse_metric_value('NaN') == (None, None)
    assert adapter.parse_metric_value('Infinity') == (None, None)
    assert adapter.parse_metric_value('-inf') == (None, None)
    # '±' spread is kept as raw TEXT (its type -- SE/SD/CI -- is unknown), not a float
    score, unc = adapter.parse_metric_value('33.7 ± 0.82')
    assert score == 33.7 and unc == '0.82'
    assert adapter.parse_metric_value('n/a') == (None, None)


def test_model_identity_prefers_hf_url_casing():
    mid, dev, slug, name = adapter.model_identity(
        'MoGe-2', 'https://huggingface.co/Ruicheng/moge-2-vitl'
    )
    assert mid == 'Ruicheng/moge-2-vitl'  # HF-true casing preserved
    assert dev == 'Ruicheng'


def test_model_identity_guesses_developer_from_name():
    mid, dev, slug, name = adapter.model_identity('GPT-5.5 Pro (xhigh)', None)
    assert dev == 'openai'
    assert name == 'GPT-5.5 Pro (xhigh)'  # raw display name preserved


def test_collection_url_is_not_read_as_a_model_identity():
    # huggingface.co/collections/<owner>/<slug> is a collection page, not a
    # model repo. It must not resolve to the `collections/<owner>` identity;
    # the URL is ignored and the developer comes from the display name.
    mid, dev, slug, name = adapter.model_identity(
        'GPT-5.5 Pro (xhigh)',
        'https://huggingface.co/collections/openai/reasoning-o1',
    )
    assert dev == 'openai'
    assert not mid.startswith('collections/')


def test_unestablished_developer_is_reported_not_dropped(tmp_path):
    conversion = _convert()
    assert conversion.total_records == len(_evaluations())
    assert [f.source_ref for f in conversion.failures] == [
        'evaluations.id=13525'
    ]
    failure = conversion.failures[0]
    assert 'AnonSplat-XL' in failure.reason
    assert failure.source_record['model_name'] == 'AnonSplat-XL'
    assert all(
        b.log.model_info.name != 'AnonSplat-XL' for b in conversion.records
    )

    report = json.loads(
        adapter.save_failure_report(
            conversion, tmp_path / 'paperswithcode_failures.json'
        ).read_text()
    )
    assert report['source_name'] == 'Papers with Code'
    assert report['failed_record_count'] == 1
    assert report['failed_records'][0]['source_ref'] == 'evaluations.id=13525'

    # the run still has to signal that the conversion was partial
    with pytest.raises(ValueError, match='Papers with Code'):
        conversion.raise_if_incomplete()


def _unplaceable_evaluations():
    base = {
        'paper_id': None,
        'task_id': '10',
        'dataset_id': '218',
        'model_name': 'MoGe-2',
        'evaluated_on': None,
        'created_at': '2026-07-01 00:00:00+00',
        'hf_model_url': 'https://huggingface.co/Ruicheng/moge-2-vitl',
        'is_open': 't',
        'external': 'f',
        'harness': None,
    }
    return [
        # a boundary overrun small enough for the scale-classification tolerance
        {**base, 'id': '30001', 'metrics': json.dumps({'delta1': '1.0005'})},
        # impossible under [0,1] and no single power of 100 lands it in range,
        # so it stays a raw scale_anomaly -- alongside a sound metric
        {
            **base,
            'id': '30002',
            'metrics': json.dumps({'delta1': '5000', 'SSIM': '0.85'}),
        },
    ]


def test_scores_outside_canonical_bounds_are_reported_not_published():
    conversion = adapter.build_logs(
        _unplaceable_evaluations(),
        _datasets(),
        _tasks(),
        _make_resolver(),
        _metric_ranges(),
        _metric_meta(),
        _papers(),
        DUMP_VERSION,
        RETRIEVED_TS,
    )
    # the rejection is per metric cell: the sound SSIM result on the second row
    # still publishes, so one bad number does not cost the whole row
    results = [r for b in conversion.records for r in b.log.evaluation_results]
    assert [r.metric_config.metric_name for r in results] == ['SSIM']

    assert [f.source_ref for f in conversion.failures] == [
        'evaluations.id=30001 metric=delta1',
        'evaluations.id=30002 metric=delta1',
    ]
    assert all(
        'outside the canonical range [0.0, 1.0]' in f.reason
        for f in conversion.failures
    )
    with pytest.raises(ValueError, match='Papers with Code'):
        conversion.raise_if_incomplete()


def test_source_data_variants():
    ds = _datasets()
    hf = adapter.build_source_data(ds['218'])
    assert hf.source_type == 'hf_dataset' and hf.hf_repo == 'ritianyu/eth3d_tar'
    url = adapter.build_source_data(ds['906'])
    assert url.source_type == 'url' and url.url
    other = adapter.build_source_data(ds['999'])
    assert other.source_type == 'other'


def test_evaluation_id_is_stable_not_now():
    bundles = _build()
    for b in bundles:
        assert b.log.evaluation_id.endswith(f'/{DUMP_VERSION}')
        assert RETRIEVED_TS not in b.log.evaluation_id


def test_directions_and_bounds():
    bundles = _build()
    by_metric = {}
    for b in bundles:
        for r in b.log.evaluation_results:
            by_metric[r.metric_config.metric_name] = r.metric_config
    # bounded canonical metrics are emitted with finite [0,1] bounds + direction
    assert by_metric['delta1'].lower_is_better is False
    assert by_metric['delta1'].max_score == 1.0
    assert by_metric['SSIM'].max_score == 1.0
    # unbounded metrics (PSNR, AbsRel) are emitted with inf (serialized "Infinity")
    assert by_metric['PSNR'].max_score == float('inf')
    assert by_metric['AbsRel'].max_score == float('inf')
    assert by_metric['AbsRel'].lower_is_better is True


def test_multi_metric_row_fans_out_with_distinct_result_ids():
    bundles = _build()
    moge = next(
        b for b in bundles if b.log.model_info.id == 'Ruicheng/moge-2-vitl'
    )
    ids = [r.evaluation_result_id for r in moge.log.evaluation_results]
    # delta1 + SSIM + AbsRel all fan out (AbsRel emitted with an inf bound)
    assert set(ids) == {
        'paperswithcode.11533.delta1',
        'paperswithcode.11533.ssim',
        'paperswithcode.11533.absrel',
    }


def test_unbounded_metrics_emitted_with_inf_and_reported():
    resolver = _make_resolver()
    _build(resolver)
    # PSNR (row 13524) and AbsRel (row 11533) are unbounded in the registry
    assert set(resolver.unbounded_emitted) == {'PSNR', 'AbsRel'}


def _rows(*per_row, model_name='MoGe-2'):
    """One model, one leaderboard row per ``(is_open, hf_model_url)`` pair."""
    return [
        {
            'id': f'7770{i}',
            'paper_id': '900',
            'task_id': '10',
            'dataset_id': '218',
            'model_name': model_name,
            'metrics': json.dumps({'delta1': '0.991'}),
            'evaluated_on': '2026-07-06',
            'created_at': '2026-06-18 13:13:06+00',
            'hf_model_url': url,
            'is_open': flag,
            'external': 'f',
            'harness': None,
        }
        for i, (flag, url) in enumerate(per_row)
    ]


def test_rows_spelling_one_model_two_ways_make_one_log():
    """PwC spells a model differently per row; only some rows carry the HF url.

    Left as distinct ids the results split across two records whose paths differ
    only in case -- and on a case-insensitive filesystem, silently collide.
    """
    # the url row is spelled moonshotai/Kimi-K2.5; the row without one falls back
    # to the helper's developer and a lowercased slug, moonshotai/kimi-k2.5
    url = 'https://huggingface.co/moonshotai/Kimi-K2.5'
    bundles = _build(
        evaluations=_rows(
            ('t', url), ('t', None), ('t', url), model_name='Kimi-K2.5'
        )
    )
    assert len(bundles) == 1
    log = bundles[0].log
    assert log.model_info.id == 'moonshotai/Kimi-K2.5'
    assert (bundles[0].developer, bundles[0].model) == (
        'moonshotai',
        'Kimi-K2.5',
    )
    # every row's results reach the one log rather than being split in two
    assert len(log.evaluation_results) == 3


def test_canonical_spelling_prefers_the_hf_url_over_the_slug():
    hf = ('moonshotai/Kimi-K2.5', 'moonshotai', 'Kimi-K2.5')
    slug = ('moonshotai/kimi-k2.5', 'moonshotai', 'kimi-k2.5')
    assert adapter.canonical_spelling({hf, slug}) == hf
    assert adapter.canonical_spelling({slug, hf}) == hf
    # a lone spelling is kept as-is, whichever kind it is
    assert adapter.canonical_spelling({slug}) == slug
    # urls that disagree in case still resolve to one, and to the same one
    other = ('moonshotai/KIMI-K2.5', 'moonshotai', 'KIMI-K2.5')
    assert adapter.canonical_spelling({hf, other}) == adapter.canonical_spelling(
        {other, hf}
    )


def _availability(*flags, url=None):
    """One model, one row per ``is_open`` flag, each row linking ``url`` or not.

    The model name is one `helpers.developer` resolves, so a row carrying no url
    still names a publisher and reaches a log rather than becoming a failure.
    """
    bundles = _build(
        evaluations=_rows(
            *((flag, url) for flag in flags), model_name='Kimi-K2.5'
        )
    )
    assert len(bundles) == 1
    return bundles[0].log.model_info.additional_details


def test_model_availability_reads_every_row_of_a_model_not_just_the_first():
    assert _availability('t')['model_availability'] == 'open_weights'
    assert _availability('f')['model_availability'] == 'closed_weights'
    assert _availability(None)['model_availability'] == 'unknown'
    assert _availability('t', 't')['model_availability'] == 'open_weights'
    # rows that contradict each other name nothing; taking the first row's flag
    # would make the answer depend on dump order
    assert _availability('t', 'f')['model_availability'] == 'unknown'
    assert _availability('f', 't')['model_availability'] == 'unknown'
    # a stated flag still counts when another row simply omits it
    assert _availability('t', None)['model_availability'] == 'open_weights'
    # PwC records what the model is, never how the reporting party reached it
    assert _availability('t')['deployment_type'] == 'unknown'


def test_a_linked_weight_repo_contradicts_a_not_open_flag():
    """PwC ships this: `osunlp/UGround-V1-7B` is flagged not-open on a row that
    links the repo its weights are in. Both fields reach the record, so reading
    the flag alone files a model as closed beside a link to its own weights.
    """
    url = 'https://huggingface.co/osunlp/UGround-V1-7B'
    assert _availability('f', url=url)['model_availability'] == 'unknown'
    assert _availability('f', 'f', url=url)['model_availability'] == 'unknown'
    # only a not-open flag is put in doubt; a linked repo agreeing with an open
    # one is the ordinary case, 738 of the dump's 1,455 models
    assert _availability('t', url=url)['model_availability'] == 'open_weights'
    # and a flag with no repo to contradict it still names the model
    assert _availability('f')['model_availability'] == 'closed_weights'


def test_additional_details_are_all_strings():
    bundles = _build()
    for b in bundles:
        for d in (b.log.model_info.additional_details or {}).values():
            assert isinstance(d, str)
        for r in b.log.evaluation_results:
            for d in (r.score_details.details or {}).values():
                assert isinstance(d, str)


def test_built_logs_validate(tmp_path):
    """Prove the skeleton: construct, save, and run the real validator."""
    bundles = _build()
    assert bundles
    paths = adapter.save_evaluation_logs(
        [
            adapter.EvaluationLogOutput(
                eval_log=b.log,
                base_dir=tmp_path,
                developer=b.developer,
                model_name=b.model,
            )
            for b in bundles
        ]
    )
    assert len(paths) == len(bundles)
    for path in paths:
        report = validate_file(path)
        assert report.valid, report.errors


# --- registry resolver / three tiers -------------------------------------------


def test_resolver_registry_hit_uses_canonical():
    r = _make_resolver()
    m = r.resolve('Accuracy', (0.0, 1.0))  # source already on canonical scale
    assert m.resolved is True
    assert m.metric_id == 'accuracy'
    assert (m.min_score, m.max_score) == (
        0.0,
        1.0,
    )  # from the registry, not 0.39
    assert m.lower_is_better is False
    assert not r.unresolved


def test_resolver_keeps_canonical_bounds():
    r = _make_resolver()
    # canonical accuracy is [0,1]; the resolver keeps that regardless of obs range
    m = r.resolve('Accuracy', (0.0, 100.0))
    assert m.resolved is True and m.metric_id == 'accuracy'
    assert (m.min_score, m.max_score) == (0.0, 1.0)


def test_reconcile_scale_no_group_fixes_impossible_value():
    # No group context (a singleton board): a value impossible under the declared
    # range, uniquely resolved by /100, IS fixed (kept raw upstream, flagged).
    score, detail = adapter.reconcile_scale(97.3, 0.0, 1.0, resolved=True)
    assert score == 0.973
    assert detail['canonical_rescale_factor'] == 0.01
    assert detail['rescale_basis'] == 'per_row'
    # already on the canonical scale -> untouched, no flag
    assert adapter.reconcile_scale(0.87, 0.0, 1.0, resolved=True) == (0.87, {})
    # unresolved metrics are never rescaled (bounds are observed, same scale)
    assert adapter.reconcile_scale(97.3, 0.0, 100.0, resolved=False) == (
        97.3,
        {},
    )
    # no unique factor (500 fits neither /100 nor x100 into [0,1]) -> flag, keep raw
    score, detail = adapter.reconcile_scale(500.0, 0.0, 1.0, resolved=True)
    assert score == 500.0
    assert detail['scale_anomaly'] == 'score_outside_canonical_range'


def test_analyze_group_clean_and_uniform_percent():
    # a clean proportion board -> no rescale, no flag
    gs = adapter.analyze_group([0.8, 0.85, 0.9, 0.95], 0.0, 1.0)
    assert gs.mode == 'uniform' and gs.factor == 1.0
    # a whole percent board for a [0,1] metric -> uniform /100 (technical rescale)
    gs = adapter.analyze_group([80.0, 85.0, 90.0, 48.0], 0.0, 1.0)
    assert gs.mode == 'uniform' and gs.factor == 0.01
    # a lone in-range 1.0 follows the board's known scale (1% -> 0.01)
    score, detail = adapter.reconcile_scale(1.0, 0.0, 1.0, True, gs)
    assert score == 0.01 and detail['rescale_basis'] == 'group_uniform'


def test_analyze_group_lone_stray_is_fixed_per_row():
    # 11 fractions + one percent (80.39): the lone stray is below the mass floor,
    # so the board stays as-is and the stray is fixed per-row (kept raw, flagged).
    vals = [0.83, 0.85, 0.87, 0.87, 0.88, 0.89, 0.91, 0.91, 0.92, 0.94, 80.39]
    gs = adapter.analyze_group(vals, 0.0, 1.0)
    assert gs.mode == 'uniform' and gs.factor == 1.0
    assert adapter.reconcile_scale(0.89, 0.0, 1.0, True, gs) == (0.89, {})
    s_fix, d_fix = adapter.reconcile_scale(80.39, 0.0, 1.0, True, gs)
    assert round(s_fix, 4) == 0.8039 and d_fix['rescale_basis'] == 'per_row'


def test_analyze_group_systematic_mismatch_is_flagged_not_fixed():
    # A [0,1]-registered metric whose board smoothly straddles 1.0 (bad-pixel %):
    # a substantial minority is out of range with no clean valley -> the SCALE is
    # wrong, not the rows. Flag the whole group; never squash the values.
    vals = [
        0.22,
        0.26,
        0.28,
        0.35,
        0.7,
        0.72,
        0.98,
        0.99,
        1.02,
        1.14,
        1.19,
        1.83,
        2.44,
        2.79,
    ]
    gs = adapter.analyze_group(vals, 0.0, 1.0)
    assert gs.mode == 'anomaly' and gs.reason == 'group_scale_mismatch'
    s, d = adapter.reconcile_scale(2.79, 0.0, 1.0, True, gs)
    assert s == 2.79 and d['scale_anomaly'] == 'group_scale_mismatch'
    # once re-registered on its natural scale ([0,100]), the same board is clean
    gs2 = adapter.analyze_group(vals, 0.0, 100.0)
    assert gs2.mode == 'uniform' and gs2.factor == 1.0
    assert adapter.reconcile_scale(2.79, 0.0, 100.0, True, gs2) == (2.79, {})


def test_analyze_group_true_mixture_rescales_per_cluster():
    # Two real clusters (proportions ~0.9 and percents ~90), each above the mass
    # floor, separated by a ~2-decade valley -> per-cluster rescale.
    props = [0.80, 0.82, 0.85, 0.88, 0.90]
    pcts = [78.0, 84.0, 88.0, 90.0, 93.0]
    gs = adapter.analyze_group(props + pcts, 0.0, 1.0)
    assert gs.mode == 'mixed'
    # low cluster is already canonical -> kept as-is; high cluster -> /100
    assert adapter.reconcile_scale(0.85, 0.0, 1.0, True, gs) == (0.85, {})
    hi_s, hi_d = adapter.reconcile_scale(90.0, 0.0, 1.0, True, gs)
    assert hi_s == 0.9 and hi_d['rescale_basis'] == 'group_mixed'


def test_group_scale_applied_consistently_in_build():
    evals = [
        {
            'id': 'a',
            'task_id': '10',
            'dataset_id': '218',
            'model_name': 'ModelA',
            'metrics': json.dumps({'Accuracy': '95'}),
            'created_at': '2026-07-01 00:00:00+00',
            'hf_model_url': 'https://huggingface.co/acme/model-a',
            'is_open': 't',
        },
        {
            'id': 'b',
            'task_id': '10',
            'dataset_id': '218',
            'model_name': 'ModelB',
            'metrics': json.dumps({'Accuracy': '1.0'}),
            'created_at': '2026-07-01 00:00:00+00',
            'hf_model_url': 'https://huggingface.co/acme/model-b',
            'is_open': 't',
        },
    ]
    # This (dataset 218, Accuracy) leaderboard is percent (centre 48) -> the WHOLE
    # group is divided by 100, so ModelB's '1.0' becomes 0.01. Per-score inference
    # would have left 1.0 in place, silently inconsistent with the rest of the board.
    resolver = _make_resolver()
    group_scales = adapter.build_group_scales(
        {('218', 'Accuracy'): [95.0, 1.0]}, resolver
    )
    bundles = adapter.build_logs(
        evals,
        _datasets(),
        _tasks(),
        resolver,
        _metric_ranges(),
        _metric_meta(),
        _papers(),
        DUMP_VERSION,
        RETRIEVED_TS,
        group_scales=group_scales,
    ).records
    by_model = {b.log.model_info.name: b for b in bundles}
    a = by_model['ModelA'].log.evaluation_results[0].score_details
    b = by_model['ModelB'].log.evaluation_results[0].score_details
    assert round(a.score, 10) == 0.95 and round(b.score, 10) == 0.01
    assert b.details['rescale_basis'] == 'group_uniform'


def test_resolver_unbounded_canonical_emits_inf():
    r = _make_resolver()
    # 'elo' is unbounded in the registry (max_score: null)
    m = r.resolve('ELO', (900.0, 2100.0))
    assert m.resolved is True and m.metric_id == 'elo'
    # null bound -> inf (serialized as the JSON string "Infinity")
    assert m.max_score == float('inf')
    assert m.detail['canonical_max'] == 'unbounded'


def test_accuracy_rescaled_to_canonical_in_build():
    accs = [
        r
        for b in _build()
        for r in b.log.evaluation_results
        if r.metric_config.metric_id == 'accuracy'
    ]
    assert accs
    for r in accs:
        assert r.metric_config.max_score == 1.0
        assert 0.0 <= r.score_details.score <= 1.0
        # singleton board (one Accuracy row) with a value impossible under [0,1]:
        # fixed per-row by /100 (multiplier 0.01), raw kept in raw_value.
        assert r.score_details.details.get('canonical_rescale_factor') == '0.01'
        assert r.score_details.details.get('rescale_basis') == 'per_row'


def test_resolver_unresolved_is_recorded_and_falls_back():
    r = _make_resolver()
    m = r.resolve('MadeUpMetricXYZ', (0.0, 5.0), dataset_slug='some-bench')
    assert m.resolved is False
    assert m.metric_id == 'paperswithcode.madeupmetricxyz'
    assert 'some-bench' in r.unresolved['MadeUpMetricXYZ']


def test_fail_closed_report_names_metrics_and_next_step():
    resolver = _make_resolver()
    _build(resolver)  # CustomZMetric is not in the registry snapshot
    assert resolver.unresolved  # would trigger the fail-closed gate in run()
    msg = adapter._report_unresolved(resolver.unresolved)
    assert 'CustomZMetric' in msg and 'registry-entity-aliases' in msg
    assert '--allow-unresolved' in msg


# --- exact-first matching / name collisions -----------------------------------


def test_similar_names_resolve_to_their_own_id_exact_first():
    r = _make_resolver()
    # 'CLIP-IQA' and 'CLIPIQA+' both normalize to 'clipiqa', so a normalized-only
    # index would let whichever was seen first win. Exact (case-insensitive) match
    # is tried first, so each spelling resolves to ITS canonical id.
    clip = r.resolve('CLIP-IQA', (0.0, 1.0))
    assert clip.resolved and clip.metric_id == 'clip-iqa'
    assert clip.detail['match_tier'] == 'exact'
    plus = r.resolve('CLIPIQA+', (0.0, 1.0))
    assert plus.resolved and plus.metric_id == 'clipiqa-plus'
    assert plus.detail['match_tier'] == 'exact'


def test_ambiguous_normalized_name_fails_closed():
    r = _make_resolver()
    # A bare 'clipiqa' matches neither spelling exactly and collides on the
    # normalized key -> unresolved (fail closed), never a silent guess.
    m = r.resolve('clipiqa', (0.1, 0.9), dataset_slug='some-iqa-bench')
    assert m.resolved is False
    assert m.detail['match_tier'] == 'ambiguous_normalized'
    assert set(m.detail['collision_candidates']) == {'clip-iqa', 'clipiqa-plus'}
    assert r.unresolved_reason['clipiqa'][0] == 'ambiguous_normalized'
    # the fail-closed report calls the collision out (distinct from 'unknown')
    msg = adapter._report_unresolved(r.unresolved, r.unresolved_reason)
    assert 'AMBIGUOUS' in msg and 'clipiqa' in msg


def test_chamfer_distance_resolves_after_alias_fix():
    r = _make_resolver()
    # Registry #50 moved the 'Chamfer Distance' alias off 'overall-chamfer' (its
    # bad owner) onto 'chamfer-distance'. It now resolves there, unambiguously.
    m = r.resolve('Chamfer Distance', (0.0, 5.0))
    assert m.resolved and m.metric_id == 'chamfer-distance'
    assert m.detail['match_tier'] == 'exact'


def test_registry_hit_surfaces_review_status_and_revision():
    r = _make_resolver()
    m = r.resolve('CLIP-IQA', (0.0, 1.0))  # a draft metric in the snapshot
    assert m.detail['canonical_review_status'] == 'draft'
    assert m.detail['canonical_confidence'] == 'high'
    assert m.detail['canonical_metric_kind_flag'] == 'real'
    # the exact registry commit the bound came from travels with every hit
    assert m.detail['bound_registry_revision'] == r.registry_revision
    assert r.registry_revision  # snapshot records a revision


def test_snapshot_has_no_exact_spelling_collisions():
    """Invariant the exact-first matcher relies on: no casefolded spelling
    (id/display_name/alias) maps to more than one canonical id in the snapshot."""
    data = json.loads(adapter.SNAPSHOT_PATH.read_text(encoding='utf-8'))
    seen: dict[str, set[str]] = {}
    for m in data['metrics']:
        for sp in (m['id'], m.get('display_name'), *(m.get('aliases') or [])):
            if sp:
                seen.setdefault(str(sp).strip().casefold(), set()).add(m['id'])
    collisions = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    assert not collisions, (
        f'exact-spelling collisions in snapshot: {collisions}'
    )


# --- idempotency: pinned timestamp + folder replace (Erotemic #3) --------------


def test_retrieved_ts_from_dump_is_deterministic():
    ts = adapter.retrieved_ts_from_dump('20260716')
    # same dump date -> same epoch, every run (not time.time())
    assert ts == adapter.retrieved_ts_from_dump('20260716_031511')
    assert float(ts) > 0
    # an un-parseable version falls back to the raw string rather than raising
    assert adapter.retrieved_ts_from_dump('weird-version') == 'weird-version'


def test_dump_version_retains_subday_suffix_for_distinct_same_day_dumps():
    # A date-only filename yields just the date (the one-dump-per-day case).
    assert adapter.dump_version_from_path('pwc_20260716.dump') == '20260716'
    # Two dumps on the same date keep their _HHMMSS suffix, so they get distinct
    # versions -> distinct evaluation_ids rather than colliding on the date.
    v1 = adapter.dump_version_from_path('paperswithcode_hf_20260716_031511.dump')
    v2 = adapter.dump_version_from_path('paperswithcode_hf_20260716_144207.dump')
    assert v1 == '20260716_031511'
    assert v2 == '20260716_144207'
    assert v1 != v2
    # A name with no date still falls back to the stem rather than raising.
    assert adapter.dump_version_from_path('weird.dump') == 'weird'


def test_superseded_records_are_removed_and_kept_ones_survive(tmp_path):
    out = tmp_path / 'out'
    (out / 'dev' / 'model').mkdir(parents=True)
    old_a = out / 'dev' / 'model' / 'a.json'
    old_b = out / 'dev' / 'model' / 'b.json'
    fresh = out / 'dev' / 'model' / 'c.json'
    for path in (old_a, old_b, fresh):
        path.write_text('{}')

    stale = adapter.existing_records(out)
    assert stale == [old_a, old_b, fresh]

    removed = adapter.remove_superseded_records(stale, {fresh}, out)
    assert removed == 2
    assert fresh.exists() and not old_a.exists() and not old_b.exists()
    assert out.exists()
    assert adapter.existing_records(tmp_path / 'missing') == []


def test_emptied_directories_are_pruned_no_higher_than_the_output_root(
    tmp_path,
):
    out = tmp_path / 'out'
    (out / 'dev' / 'model').mkdir(parents=True)
    record = out / 'dev' / 'model' / 'a.json'
    record.write_text('{}')

    adapter.remove_superseded_records([record], set(), out)

    assert not (out / 'dev').exists()
    assert out.exists() and tmp_path.exists()


# --- metric_unit, uncertainty, provenance, bucket listing ----------------------


def test_metric_unit_comes_from_the_canonical_bounds_not_the_source_scale():
    def unit(lo, hi, resolved=True):
        return adapter._metric_unit_from_bounds(
            _resolved_metric(min_score=lo, max_score=hi, resolved=resolved)
        )

    assert unit(0.0, 1.0) == 'proportion'
    assert unit(0.0, 100.0) == 'percent'
    # any other canonical shape has no unit name
    assert unit(0.0, float('inf')) is None
    assert unit(-1.0, 1.0) is None
    assert unit(1.0, 5.0) is None
    # an unresolved metric has no canonical contract to take a unit from, even
    # though its observed-range fallback happens to look like a proportion
    assert unit(0.0, 1.0, resolved=False) is None


def test_metric_unit_and_raw_scale_in_build():
    cfgs = {
        r.metric_config.metric_name: r.metric_config
        for b in _build()
        for r in b.log.evaluation_results
    }
    # canonical [0,1] -> proportion; unbounded -> no unit, raw scale preserved
    assert cfgs['delta1'].metric_unit == 'proportion'
    assert cfgs['PSNR'].metric_unit is None
    assert cfgs['PSNR'].additional_details.get('pwc_scale') == 'unbounded'


def test_reported_uncertainty_kept_as_text_not_typed_se():
    sd = adapter.score_details(
        {'id': '1'}, '33.7 ± 0.82', 33.7, '0.82', {}, None
    )
    # the '±' spread is untyped in the source, so no typed Uncertainty is asserted
    assert sd.uncertainty is None
    assert sd.details['reported_uncertainty'] == '0.82'
    # no spread -> the key is simply absent (not an empty/None value)
    sd2 = adapter.score_details({'id': '2'}, '0.5', 0.5, None, {}, None)
    assert sd2.uncertainty is None
    assert 'reported_uncertainty' not in sd2.details


def test_source_metadata_provenance_reflects_source():
    # local --dump: no bucket provenance claim, records the dump file name
    local = adapter.build_source_metadata(
        '20260716', source_bucket=None, dump_file='pwc_20260716.dump'
    )
    assert 'source_bucket' not in local.additional_details
    assert local.additional_details['source_dump_file'] == 'pwc_20260716.dump'
    # downloaded from a bucket: records the exact bucket it came from
    remote = adapter.build_source_metadata(
        '20260716', source_bucket='some/other-bucket', dump_file='x.dump'
    )
    assert remote.additional_details['source_bucket'] == 'some/other-bucket'


def test_latest_dump_remote_path_lists_postgres_recursively(monkeypatch):
    seen = {}

    class _Entry:
        def __init__(self, path):
            self.path = path

    class _FakeApi:
        def list_bucket_tree(self, bucket, prefix=None, recursive=False):
            seen['prefix'] = prefix
            seen['recursive'] = recursive
            return [
                _Entry(
                    'postgres'
                ),  # the dir entry itself -> ignored (no .dump)
                _Entry('postgres/paperswithcode_hf_20260715_010101.dump'),
                _Entry('postgres/paperswithcode_hf_20260716_031511.dump'),
                _Entry('README.md'),
            ]

    monkeypatch.setattr('huggingface_hub.HfApi', _FakeApi)
    got = adapter.latest_dump_remote_path('huggingface/paperswithcode-backups')
    # nested files are found (recursive) and the newest is chosen
    assert got == 'postgres/paperswithcode_hf_20260716_031511.dump'
    assert seen == {'prefix': 'postgres', 'recursive': True}


def test_emit_source_version_names_a_local_dump_without_network(capsys):
    # A --dump path is named from its own stamp, so the probe touches no bucket.
    exit_code = adapter.run(
        adapter.parse_args(
            [
                '--emit-source-version',
                '--dump',
                '/tmp/pwc-raw/paperswithcode_hf_20260716_031511.dump',
            ]
        )
    )

    assert exit_code == 0
    # One line, the dump version, and nothing converted -- this is what the
    # scheduler compares to decide whether a run can be skipped.
    assert capsys.readouterr().out.strip() == '20260716_031511'


def test_emit_source_version_resolves_the_latest_dump_by_listing(
    monkeypatch, capsys
):
    class _Entry:
        def __init__(self, path):
            self.path = path

    downloaded = []

    class _FakeApi:
        def list_bucket_tree(self, bucket, prefix=None, recursive=False):
            return [
                _Entry('postgres/paperswithcode_hf_20260715_010101.dump'),
                _Entry('postgres/paperswithcode_hf_20260716_031511.dump'),
            ]

        def download_bucket_files(self, *args, **kwargs):
            downloaded.append(args)  # a probe must never download the dump

    monkeypatch.setattr('huggingface_hub.HfApi', _FakeApi)

    exit_code = adapter.run(adapter.parse_args(['--emit-source-version']))

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == '20260716_031511'
    # The probe lists the bucket; it must not download the multi-GB dump.
    assert downloaded == []
