"""Registry-contract regressions for the PwC DrugBank adapter."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
import yaml

from every_eval_ever.adapters.paperswithcode import adapter as pwc_adapter
from every_eval_ever.adapters.paperswithcode_drugbank import adapter

DUMP_SHA = 'a' * 64
OVERLAY_SHA = 'b' * 64
RETRIEVED_TS = '1784160000.0'
RESOLVER = pwc_adapter.MetricResolver()
REGISTRY_REVISION = RESOLVER.registry_revision
assert REGISTRY_REVISION is not None


def _payload(*, source_scale: str = 'identity') -> dict[str, object]:
    return {
        'schema_version': 2,
        'dump_sha256': DUMP_SHA,
        'registry_revision': REGISTRY_REVISION,
        'retrieved_timestamp': RETRIEVED_TS,
        'entries': [
            {
                'pwc_evaluation_id': 'eval-1',
                'anchors': {
                    'paper_id': 'paper-1',
                    'dataset_id': 'drugbank-id',
                    'task_id': 'ddi-task',
                    'model_name': 'Method Alpha',
                    'model_id': 'example-org/method-alpha',
                    'developer': 'example-org',
                },
                'source_metrics_sha256': adapter.source_metrics_sha256(
                    {'AUROC': '99.49' if source_scale == 'percent' else '0.9949'}
                ),
                'qualification': {
                    'benchmark_id': (
                        'paperswithcode-drugbank.alpha-study.'
                        'ddi-event-multiclass.inductive.one-unseen-drug'
                    ),
                    'split_id': 'inductive-s2',
                    'study_id': 'alpha-study',
                    'protocol_id': 'one-unseen-drug',
                    'task_id': 'ddi-event-multiclass',
                    'task_type': 'ddi-event-multiclass',
                    'candidate_label_space': 'reported-relation-labels',
                    'drug_entity_overlap': 'one-unseen',
                    'pair_overlap': 'none',
                    'relation_class_overlap': 'shared',
                    'temporal_ordering': 'not-reported',
                    'negative_sampling': 'paper-defined',
                    'split_preprocessing': 'paper-defined',
                },
                'evidence': {
                    'source_url': 'https://example.org/paper',
                    'source_locator': 'Table 1',
                    'review_note': 'Synthetic registry-contract fixture.',
                },
                'metrics': [
                    {
                        'source_name': 'AUROC',
                        'metric_id': 'auroc',
                        'metric_name': 'AUROC',
                        'metric_kind': 'auroc',
                        'metric_unit': 'proportion',
                        'lower_is_better': False,
                        'min_score': 0.0,
                        'max_score': 1.0,
                        'source_scale': source_scale,
                    }
                ],
            }
        ],
    }


def _overlay(*, source_scale: str = 'identity') -> adapter.ProtocolOverlay:
    return adapter.ProtocolOverlay.model_validate(
        _payload(source_scale=source_scale)
    )


def _row(*, percent: bool = False) -> dict[str, object]:
    metrics = {'AUROC': '99.49' if percent else '0.9949'}
    return {
        'id': 'eval-1',
        'paper_id': 'paper-1',
        'dataset_id': 'drugbank-id',
        'task_id': 'ddi-task',
        'model_name': 'Method Alpha',
        'evaluated_on': '2024-03-25',
        'metrics': json.dumps(metrics),
    }


def _datasets() -> dict[str, dict[str, object]]:
    return {
        'drugbank-id': {
            'id': 'drugbank-id',
            'name': 'DrugBank',
            'slug': 'drugbank',
            'url': 'https://paperswithcode.com/dataset/drugbank',
            'homepage': 'https://go.drugbank.com',
        }
    }


def test_manifest_revision_must_equal_vendored_snapshot() -> None:
    payload = _payload()
    payload['registry_revision'] = 'd' * 40
    overlay = adapter.ProtocolOverlay.model_validate(payload)

    with pytest.raises(ValueError, match='registry_revision does not match'):
        adapter.validate_registry_contract(overlay)

    with pytest.raises(ValueError, match='registry_revision does not match'):
        adapter.build_logs([_row()], _datasets(), overlay, OVERLAY_SHA)


@pytest.mark.parametrize(
    ('field_name', 'replacement'),
    [
        ('metric_id', 'accuracy'),
        ('metric_name', 'Area Under ROC'),
        ('metric_kind', 'roc_auc'),
        ('metric_unit', 'percent'),
        ('lower_is_better', True),
        ('min_score', -1.0),
        ('max_score', 2.0),
    ],
)
def test_manifest_canonical_fields_are_registry_assertions(
    field_name: str, replacement: object
) -> None:
    payload = _payload()
    payload['entries'][0]['metrics'][0][field_name] = replacement
    overlay = adapter.ProtocolOverlay.model_validate(payload)

    with pytest.raises(ValueError, match=rf'mismatch for {field_name}'):
        adapter.validate_registry_contract(overlay)


def test_unknown_source_metric_fails_closed() -> None:
    payload = _payload()
    metric = payload['entries'][0]['metrics'][0]
    metric.update(
        {
            'source_name': 'Definitely Not A Registry Metric',
            'metric_id': 'unknown-metric',
            'metric_name': 'Definitely Not A Registry Metric',
            'metric_kind': 'unknown-metric',
        }
    )
    overlay = adapter.ProtocolOverlay.model_validate(payload)

    with pytest.raises(ValueError, match='not resolvable'):
        adapter.validate_registry_contract(overlay)


def test_ambiguous_normalized_source_metric_fails_closed() -> None:
    payload = _payload()
    metric = payload['entries'][0]['metrics'][0]
    metric.update(
        {
            'source_name': 'clipiqa',
            'metric_id': 'clip-iqa',
            'metric_name': 'clipiqa',
            'metric_kind': 'clip-iqa',
        }
    )
    overlay = adapter.ProtocolOverlay.model_validate(payload)

    with pytest.raises(ValueError, match='ambiguous_normalized'):
        adapter.validate_registry_contract(overlay)


def test_auroc_output_matches_generic_pwc_canonical_contract() -> None:
    overlay = _overlay(source_scale='percent')
    [log] = adapter.build_logs(
        [_row(percent=True)], _datasets(), overlay, OVERLAY_SHA
    )
    [result] = log.evaluation_results

    observed = (99.49, 99.49)
    resolved = RESOLVER.resolve('AUROC', observed, 'drugbank')
    generic = pwc_adapter.build_metric_config('AUROC', resolved, observed, None)

    assert result.metric_config.metric_id == generic.metric_id
    assert result.metric_config.metric_name == generic.metric_name
    assert result.metric_config.metric_kind == generic.metric_kind
    assert result.metric_config.metric_unit == generic.metric_unit
    assert result.metric_config.lower_is_better == generic.lower_is_better
    assert result.metric_config.score_type == generic.score_type
    assert result.metric_config.min_score == generic.min_score
    assert result.metric_config.max_score == generic.max_score
    assert result.metric_config.additional_details['bound_registry_revision'] == (
        REGISTRY_REVISION
    )
    assert result.metric_config.additional_details['observed_min'] == '99.49'
    assert result.metric_config.additional_details['observed_max'] == '99.49'
    assert result.score_details.score == pytest.approx(0.9949)
    assert result.score_details.details['reviewed_source_scale'] == 'percent'
    assert result.score_details.details['applied_scale_factor'] == '0.01'


def test_observed_range_is_derived_from_source_rows() -> None:
    overlay = _overlay(source_scale='percent')
    selected = _row(percent=True)
    other = dict(selected)
    other['id'] = 'eval-2'
    other['metrics'] = json.dumps({'AUROC': '75.0'})

    [log] = adapter.build_logs(
        [selected, other], _datasets(), overlay, OVERLAY_SHA
    )
    [result] = log.evaluation_results

    assert result.metric_config.additional_details['observed_min'] == '75.0'
    assert result.metric_config.additional_details['observed_max'] == '99.49'
    assert result.metric_config.min_score == 0.0
    assert result.metric_config.max_score == 1.0


def test_cli_contract_gate_runs_before_dump_access(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = deepcopy(_payload())
    payload['entries'][0]['metrics'][0]['metric_kind'] = 'roc_auc'
    overlay_path = tmp_path / 'overlay.yaml'
    overlay_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding='utf-8'
    )
    dump_path = tmp_path / 'paperswithcode.dump'
    output_dir = tmp_path / 'data' / adapter.COLLECTION_NAME

    def unexpected_dump_access(_path):
        raise AssertionError('registry contract must fail before dump access')

    monkeypatch.setattr(adapter, 'file_sha256', unexpected_dump_access)

    with pytest.raises(ValueError, match='mismatch for metric_kind'):
        adapter.run(
            adapter.parse_args(
                [
                    '--dump',
                    str(dump_path),
                    '--overlay',
                    str(overlay_path),
                    '--output-dir',
                    str(output_dir),
                ]
            )
        )
