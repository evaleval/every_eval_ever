"""Offline tests for the standalone Papers with Code DrugBank adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from every_eval_ever.adapters.paperswithcode_drugbank import adapter

DUMP_SHA = 'a' * 64
OVERLAY_SHA = 'b' * 64
RETRIEVED_TS = '1784160000.0'


def _metrics_hash(metrics: dict[str, object]) -> str:
    return adapter.source_metrics_sha256(metrics)


def _overlay_payload(
    metrics: dict[str, object],
    *,
    overlap: str = 'one-unseen',
    source_scale: str = 'percent',
) -> dict[str, object]:
    protocol_id = {
        'both-seen': 'pair-held-out',
        'one-unseen': 'one-unseen-drug',
        'both-unseen': 'two-unseen-drugs',
    }[overlap]
    return {
        'schema_version': 1,
        'dump_sha256': DUMP_SHA,
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
                'source_metrics_sha256': _metrics_hash(metrics),
                'qualification': {
                    'study_id': 'alpha-study',
                    'protocol_id': protocol_id,
                    'task_id': 'ddi-event-multiclass',
                    'task_type': 'ddi-event-multiclass',
                    'candidate_label_space': 'reported-relation-labels',
                    'drug_entity_overlap': overlap,
                    'pair_overlap': 'none',
                    'relation_class_overlap': 'shared',
                    'temporal_ordering': 'not-reported',
                    'negative_sampling': 'paper-defined',
                    'split_preprocessing': 'paper-defined',
                },
                'evidence': {
                    'source_url': 'https://example.org/paper',
                    'source_locator': 'Table 1 / split definition',
                    'review_note': 'Synthetic fixture with explicit split semantics.',
                },
                'metrics': [
                    {
                        'source_name': 'AUROC',
                        'metric_id': 'auroc',
                        'metric_name': 'AUROC',
                        'metric_kind': 'roc_auc',
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


def _overlay(
    metrics: dict[str, object],
    *,
    overlap: str = 'one-unseen',
    source_scale: str = 'percent',
) -> adapter.ProtocolOverlay:
    return adapter.ProtocolOverlay.model_validate(
        _overlay_payload(metrics, overlap=overlap, source_scale=source_scale)
    )


def _source_rows(metrics: dict[str, object]) -> list[dict[str, object]]:
    return [
        {
            'id': 'eval-1',
            'paper_id': 'paper-1',
            'dataset_id': 'drugbank-id',
            'task_id': 'ddi-task',
            'model_name': 'Method Alpha',
            'evaluated_on': '2024-03-25',
            'metrics': json.dumps(metrics),
        }
    ]


def _datasets() -> dict[str, dict[str, object]]:
    return {
        'drugbank-id': {
            'id': 'drugbank-id',
            'name': 'DrugBank',
            'slug': 'drugbank',
            'url': 'https://go.drugbank.com',
            'homepage': 'https://go.drugbank.com',
        }
    }


def test_generalization_regime_is_derived_from_explicit_entity_overlap() -> None:
    metrics = {'AUROC': '99.49'}

    transductive = _overlay(metrics, overlap='both-seen')
    assert transductive.entries[0].qualification.generalization_regime == (
        'transductive'
    )
    assert '.transductive.' in (
        transductive.entries[0].qualification.evaluation_name()
    )

    for overlap in ('one-unseen', 'both-unseen'):
        inductive = _overlay(metrics, overlap=overlap)
        assert inductive.entries[0].qualification.generalization_regime == (
            'inductive'
        )
        assert '.inductive.' in inductive.entries[0].qualification.evaluation_name()


def test_explicit_percent_scale_converts_without_distribution_inference() -> None:
    metrics = {'AUROC': '99.49'}
    result = adapter.build_logs(
        _source_rows(metrics),
        _datasets(),
        _overlay(metrics),
        OVERLAY_SHA,
        dump_file='paperswithcode.dump',
    )

    assert result.total_records == 1
    assert result.failures == []
    assert result.exclusions == []
    assert len(result.records) == 1

    bundle = result.records[0]
    log = bundle.log
    assert log.model_info.id == 'example-org/method-alpha'
    assert log.model_info.developer == 'example-org'
    assert log.source_metadata.source_type.value == 'documentation'
    assert log.source_metadata.source_name == adapter.SOURCE_NAME

    score = log.evaluation_results[0]
    assert score.score_details.score == pytest.approx(0.9949)
    assert score.metric_config.metric_id == 'auroc'
    assert score.metric_config.metric_unit == 'proportion'
    assert score.score_details.details['raw_value'] == '99.49'
    assert score.score_details.details['reviewed_source_scale'] == 'percent'
    assert score.score_details.details['applied_scale_factor'] == '0.01'
    assert score.score_details.details['generalization_regime'] == 'inductive'
    assert score.source_data.dataset_name == 'DrugBank'


def test_wrong_explicit_scale_fails_closed_instead_of_guessing() -> None:
    metrics = {'AUROC': '99.49'}
    with pytest.raises(ValueError, match='outside reviewed canonical range'):
        adapter.build_logs(
            _source_rows(metrics),
            _datasets(),
            _overlay(metrics, source_scale='identity'),
            OVERLAY_SHA,
        )


def test_percent_marker_must_agree_with_manifest_scale() -> None:
    metrics = {'AUROC': '99.49%'}
    with pytest.raises(ValueError, match='percent marker'):
        adapter.build_logs(
            _source_rows(metrics),
            _datasets(),
            _overlay(metrics, source_scale='identity'),
            OVERLAY_SHA,
        )


def test_metrics_payload_and_source_anchors_are_drift_guards() -> None:
    metrics = {'AUROC': '99.49'}
    overlay = _overlay(metrics)

    with pytest.raises(ValueError, match='metrics payload drift'):
        adapter.build_logs(
            _source_rows({'AUROC': '99.50'}),
            _datasets(),
            overlay,
            OVERLAY_SHA,
        )

    wrong_anchor = _source_rows(metrics)
    wrong_anchor[0]['paper_id'] = 'other-paper'
    with pytest.raises(ValueError, match='anchor drift'):
        adapter.build_logs(wrong_anchor, _datasets(), overlay, OVERLAY_SHA)


def test_non_drugbank_dataset_is_rejected() -> None:
    metrics = {'AUROC': '99.49'}
    bad_datasets = _datasets()
    bad_datasets['drugbank-id'] = {
        'id': 'drugbank-id',
        'name': 'Other',
        'slug': 'other',
    }
    with pytest.raises(ValueError, match='does not target DrugBank'):
        adapter.build_logs(
            _source_rows(metrics),
            bad_datasets,
            _overlay(metrics),
            OVERLAY_SHA,
        )


def test_duplicate_source_cell_and_inconsistent_model_identity_are_rejected() -> None:
    metrics = {'AUROC': '99.49'}
    payload = _overlay_payload(metrics)
    duplicate = json.loads(json.dumps(payload['entries'][0]))
    payload['entries'].append(duplicate)
    with pytest.raises(ValueError, match='selected more than once'):
        adapter.ProtocolOverlay.model_validate(payload)

    payload = _overlay_payload(metrics)
    payload['entries'][0]['anchors']['developer'] = 'wrong-org'
    with pytest.raises(ValueError, match='developer component'):
        adapter.ProtocolOverlay.model_validate(payload)


def test_stable_evaluation_id_binds_dump_and_manifest_hashes() -> None:
    metrics = {'AUROC': '99.49'}
    result = adapter.build_logs(
        _source_rows(metrics),
        _datasets(),
        _overlay(metrics),
        OVERLAY_SHA,
    )
    assert result.records[0].log.evaluation_id.endswith(
        'aaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb'
    )


def test_overlay_requires_entries_and_unix_retrieval_time() -> None:
    with pytest.raises(ValueError):
        adapter.ProtocolOverlay.model_validate(
            {
                'schema_version': 1,
                'dump_sha256': DUMP_SHA,
                'retrieved_timestamp': RETRIEVED_TS,
                'entries': [],
            }
        )

    payload = _overlay_payload({'AUROC': '99.49'})
    payload['retrieved_timestamp'] = 'not-an-epoch'
    with pytest.raises(ValueError, match='Unix-epoch'):
        adapter.ProtocolOverlay.model_validate(payload)


def test_output_directory_must_be_empty(tmp_path) -> None:
    output = tmp_path / 'out'
    output.mkdir()
    (output / 'old.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='must be empty'):
        adapter.require_empty_output_dir(output)


@dataclass
class _DumpEntry:
    desc: str
    tag: str
    defn: str


class _FakeDump:
    def __init__(self, evaluation: dict[str, object], dataset: dict[str, object]):
        self.entries = [
            _DumpEntry(
                'TABLE',
                'evaluations',
                """CREATE TABLE evaluations (
                id text,
                paper_id text,
                dataset_id text,
                task_id text,
                model_name text,
                evaluated_on text,
                metrics jsonb
                );""",
            ),
            _DumpEntry(
                'TABLE',
                'datasets',
                """CREATE TABLE datasets (
                id text,
                name text,
                slug text,
                url text,
                homepage text
                );""",
            ),
        ]
        self._rows = {
            'evaluations': [
                tuple(
                    evaluation[key]
                    for key in (
                        'id',
                        'paper_id',
                        'dataset_id',
                        'task_id',
                        'model_name',
                        'evaluated_on',
                        'metrics',
                    )
                )
            ],
            'datasets': [
                tuple(
                    dataset.get(key)
                    for key in ('id', 'name', 'slug', 'url', 'homepage')
                )
            ],
        }

    def table_data(self, schema: str, table: str):
        assert schema == 'public'
        return iter(self._rows[table])


def test_source_context_reads_only_manifest_selected_rows() -> None:
    metrics = {'AUROC': '99.49'}
    overlay = _overlay(metrics)
    evaluation = _source_rows(metrics)[0]
    dataset = _datasets()['drugbank-id']
    dump = _FakeDump(evaluation, dataset)

    evaluations, datasets = adapter.load_source_context(dump, overlay)
    assert [row['id'] for row in evaluations] == ['eval-1']
    assert set(datasets) == {'drugbank-id'}


def test_cli_requires_local_dump_and_external_manifest(tmp_path) -> None:
    dump = tmp_path / 'pwc.dump'
    manifest = tmp_path / 'overlay.yaml'
    args = adapter.parse_args(
        [
            '--dump',
            str(dump),
            '--overlay',
            str(manifest),
            '--output-dir',
            str(tmp_path / 'out'),
        ]
    )
    assert args.dump == dump
    assert args.overlay == manifest
    assert args.output_dir == tmp_path / 'out'
