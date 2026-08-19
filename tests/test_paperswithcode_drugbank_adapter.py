"""Offline tests for the standalone Papers with Code DrugBank adapter."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from every_eval_ever.adapters.paperswithcode_drugbank import adapter

DUMP_SHA = 'a' * 64
OVERLAY_SHA = 'b' * 64
REGISTRY_REVISION = 'c' * 40
RETRIEVED_TS = '1784160000.0'


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
    regime = 'transductive' if overlap == 'both-seen' else 'inductive'
    return {
        'schema_version': 1,
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
                'source_metrics_sha256': adapter.source_metrics_sha256(metrics),
                'qualification': {
                    'benchmark_id': (
                        'paperswithcode-drugbank.alpha-study.'
                        f'ddi-event-multiclass.{regime}.{protocol_id}'
                    ),
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
            'url': 'https://paperswithcode.com/dataset/drugbank',
            'homepage': 'https://go.drugbank.com',
        }
    }


def test_generalization_regime_is_derived_from_explicit_entity_overlap() -> (
    None
):
    metrics = {'AUROC': '99.49'}

    transductive = _overlay(metrics, overlap='both-seen')
    assert transductive.entries[0].qualification.generalization_regime == (
        'transductive'
    )
    assert '.transductive.' in (
        transductive.entries[0].qualification.benchmark_id
    )

    for overlap in ('one-unseen', 'both-unseen'):
        inductive = _overlay(metrics, overlap=overlap)
        assert inductive.entries[0].qualification.generalization_regime == (
            'inductive'
        )
        assert '.inductive.' in inductive.entries[0].qualification.benchmark_id


def test_explicit_percent_scale_converts_without_distribution_inference() -> (
    None
):
    metrics = {'AUROC': '99.49'}
    overlay = _overlay(metrics)
    logs = adapter.build_logs(
        _source_rows(metrics),
        _datasets(),
        overlay,
        OVERLAY_SHA,
        dump_file='paperswithcode.dump',
    )

    [log] = logs
    assert log.model_info.id == 'example-org/method-alpha'
    assert log.model_info.developer == 'example-org'
    assert log.source_metadata.source_type.value == 'documentation'
    assert log.source_metadata.source_name == adapter.SOURCE_NAME
    assert log.source_metadata.additional_details['registry_revision'] == (
        REGISTRY_REVISION
    )

    score = log.evaluation_results[0]
    qualification = overlay.entries[0].qualification
    assert score.evaluation_name == qualification.benchmark_id
    assert score.score_details.details['protocol_semantic_sha256'] == (
        qualification.semantic_sha256()
    )
    assert score.score_details.score == pytest.approx(0.9949)
    assert score.metric_config.metric_id == 'auroc'
    assert score.metric_config.metric_unit == 'proportion'
    assert score.score_details.details['raw_value'] == '99.49'
    assert score.score_details.details['reviewed_source_scale'] == 'percent'
    assert score.score_details.details['applied_scale_factor'] == '0.01'
    assert score.score_details.details['generalization_regime'] == 'inductive'
    assert score.source_data.dataset_name == 'DrugBank'
    assert (
        'https://paperswithcode.com/dataset/drugbank'
        not in score.source_data.url
    )
    assert score.source_data.url == [adapter.DRUGBANK_URL]
    assert score.source_data.additional_details['raw_dataset_url'] == (
        'https://paperswithcode.com/dataset/drugbank'
    )
    assert log.source_metadata.additional_details['pwc_data_archive_url'] == (
        adapter.PWC_DATASET_ARCHIVE_URL
    )


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


@pytest.mark.parametrize('raw_value', ['1,000', '0,5'])
def test_metric_parser_rejects_ambiguous_comma_formatting(
    raw_value: str,
) -> None:
    with pytest.raises(ValueError, match='ambiguous comma formatting'):
        adapter.parse_metric_value(raw_value)


def test_source_data_uses_only_trusted_drugbank_url() -> None:
    source = adapter.build_source_data(
        {
            'id': 'drugbank-id',
            'name': 'DrugBank',
            'slug': 'drugbank',
            'url': 'https://go.drugbank.com@evil.example/',
            'homepage': 'http://localhost/private',
            'paper_url': 'https://example.org/paper-not-dataset',
        }
    )

    assert source.url == [adapter.DRUGBANK_URL]
    assert source.additional_details['raw_dataset_url'] == (
        'https://go.drugbank.com@evil.example/'
    )
    assert source.additional_details['raw_dataset_homepage'] == (
        'http://localhost/private'
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


@pytest.mark.parametrize(
    ('name', 'slug'),
    [('Other', 'other'), ('Other', 'drugbank'), ('DrugBank', 'other')],
)
def test_non_drugbank_or_conflicting_dataset_is_rejected(
    name: str, slug: str
) -> None:
    metrics = {'AUROC': '99.49'}
    bad_datasets = _datasets()
    bad_datasets['drugbank-id'] = {
        'id': 'drugbank-id',
        'name': name,
        'slug': slug,
    }
    with pytest.raises(ValueError, match='does not target DrugBank'):
        adapter.build_logs(
            _source_rows(metrics),
            bad_datasets,
            _overlay(metrics),
            OVERLAY_SHA,
        )


def test_overlay_rejects_duplicate_cells_and_invalid_model_mappings() -> None:
    metrics = {'AUROC': '99.49'}
    payload = _overlay_payload(metrics)
    duplicate = deepcopy(payload['entries'][0])
    payload['entries'].append(duplicate)
    with pytest.raises(ValueError, match='selected more than once'):
        adapter.ProtocolOverlay.model_validate(payload)

    payload = _overlay_payload(metrics)
    payload['entries'][0]['anchors']['model_id'] = 'missing-model-namespace'
    with pytest.raises(ValueError, match='at least two'):
        adapter.ProtocolOverlay.model_validate(payload)

    payload = _overlay_payload(metrics)
    conflicting_model = deepcopy(payload['entries'][0])
    conflicting_model['anchors']['model_id'] = 'other-org/method-alpha'
    conflicting_model['anchors']['developer'] = 'other-company'
    payload['entries'].append(conflicting_model)
    with pytest.raises(ValueError, match='multiple canonical model ids'):
        adapter.ProtocolOverlay.model_validate(payload)


def test_model_ids_allow_nested_namespaces_and_independent_developers(
    tmp_path,
) -> None:
    metrics = {'AUROC': '99.49'}
    payload = _overlay_payload(metrics)
    payload['entries'][0]['anchors']['model_id'] = (
        'example-org/azure/method-alpha'
    )
    payload['entries'][0]['anchors']['developer'] = 'example-company'
    logs = adapter.build_logs(
        _source_rows(metrics),
        _datasets(),
        adapter.ProtocolOverlay.model_validate(payload),
        OVERLAY_SHA,
    )

    [log] = logs
    assert log.model_info.id == 'example-org/azure/method-alpha'
    assert log.model_info.developer == 'example-company'

    output_dir = tmp_path / 'data' / 'paperswithcode-drugbank'
    [path] = adapter.export(logs, output_dir)
    assert path.parent == output_dir / 'example-org' / 'azure_method-alpha'


def test_aliases_with_conflicting_developers_are_rejected() -> None:
    metrics = {'AUROC': '99.49'}
    payload = _overlay_payload(metrics)
    second_entry = deepcopy(payload['entries'][0])
    second_entry['pwc_evaluation_id'] = 'eval-2'
    second_entry['anchors']['paper_id'] = 'paper-2'
    second_entry['anchors']['model_name'] = 'Method Alpha v1'
    second_entry['anchors']['developer'] = 'other-company'
    payload['entries'].append(second_entry)

    rows = _source_rows(metrics)
    second_row = dict(rows[0])
    second_row.update(
        id='eval-2', paper_id='paper-2', model_name='Method Alpha v1'
    )
    rows.append(second_row)

    with pytest.raises(ValueError, match='inconsistent reviewed developer'):
        adapter.build_logs(
            rows,
            _datasets(),
            adapter.ProtocolOverlay.model_validate(payload),
            OVERLAY_SHA,
        )


def test_stable_evaluation_id_binds_dump_and_raw_source_bundle_only() -> None:
    metrics = {'AUROC': '99.49'}
    first = adapter.build_logs(
        _source_rows(metrics),
        _datasets(),
        _overlay(metrics),
        OVERLAY_SHA,
    )
    after_review_note_edit = adapter.build_logs(
        _source_rows(metrics),
        _datasets(),
        _overlay(metrics),
        'c' * 64,
    )

    evaluation_id = first[0].evaluation_id
    assert evaluation_id == after_review_note_edit[0].evaluation_id
    assert evaluation_id == adapter._bundle_evaluation_id(DUMP_SHA, ['eval-1'])


def test_model_ids_that_collided_under_underscore_flattening_remain_distinct() -> (
    None
):
    metrics = {'AUROC': '99.49'}
    payload = _overlay_payload(metrics)
    first_entry = payload['entries'][0]
    first_entry['anchors']['model_id'] = 'foo_bar/baz'
    first_entry['anchors']['developer'] = 'foo_bar'

    second_entry = deepcopy(first_entry)
    second_entry['pwc_evaluation_id'] = 'eval-2'
    second_entry['anchors']['paper_id'] = 'paper-2'
    second_entry['anchors']['model_name'] = 'Method Beta'
    second_entry['anchors']['model_id'] = 'foo/bar_baz'
    second_entry['anchors']['developer'] = 'foo'
    payload['entries'].append(second_entry)

    rows = _source_rows(metrics)
    second_row = dict(rows[0])
    second_row.update(id='eval-2', paper_id='paper-2', model_name='Method Beta')
    rows.append(second_row)

    logs = adapter.build_logs(
        rows,
        _datasets(),
        adapter.ProtocolOverlay.model_validate(payload),
        OVERLAY_SHA,
    )

    assert {log.model_info.id for log in logs} == {
        'foo_bar/baz',
        'foo/bar_baz',
    }
    assert len({log.evaluation_id for log in logs}) == 2


def test_source_model_aliases_consolidate_under_canonical_model_id() -> None:
    metrics = {'AUROC': '99.49'}
    payload = _overlay_payload(metrics)
    second_entry = deepcopy(payload['entries'][0])
    second_entry['pwc_evaluation_id'] = 'eval-2'
    second_entry['anchors']['paper_id'] = 'paper-2'
    second_entry['anchors']['model_name'] = 'Method Alpha v1'
    payload['entries'].append(second_entry)

    rows = _source_rows(metrics)
    second_row = dict(rows[0])
    second_row.update(
        id='eval-2', paper_id='paper-2', model_name='Method Alpha v1'
    )
    rows.append(second_row)

    logs = adapter.build_logs(
        rows,
        _datasets(),
        adapter.ProtocolOverlay.model_validate(payload),
        OVERLAY_SHA,
    )

    [log] = logs
    assert log.model_info.id == 'example-org/method-alpha'
    assert json.loads(log.model_info.additional_details['raw_model_names']) == [
        'Method Alpha',
        'Method Alpha v1',
    ]
    names_by_source_id = {
        result.score_details.details[
            'pwc_evaluation_id'
        ]: result.score_details.details['pwc_model_name']
        for result in log.evaluation_results
    }
    assert names_by_source_id == {
        'eval-1': 'Method Alpha',
        'eval-2': 'Method Alpha v1',
    }
    assert len(log.evaluation_results) == 2


@pytest.mark.parametrize(
    ('field', 'replacement'),
    [
        ('study_id', 'beta-study'),
        ('protocol_id', 'alternate-protocol'),
        ('task_id', 'ddi-binary'),
        ('task_type', 'ddi-binary'),
        ('candidate_label_space', 'binary-interaction'),
        ('drug_entity_overlap', 'both-unseen'),
        ('pair_overlap', 'partial'),
        ('relation_class_overlap', 'disjoint'),
        ('temporal_ordering', 'chronological'),
        ('negative_sampling', 'uniform'),
        ('split_preprocessing', 'deduplicated-before-split'),
    ],
)
def test_protocol_digest_binds_every_semantic_field_without_rekeying_benchmark(
    field: str, replacement: str
) -> None:
    payload = _overlay_payload({'AUROC': '99.49'})
    qualification = payload['entries'][0]['qualification']
    baseline = adapter.ProtocolQualification.model_validate(qualification)
    changed = adapter.ProtocolQualification.model_validate(
        {**qualification, field: replacement}
    )

    assert changed.semantic_sha256() != baseline.semantic_sha256()
    assert changed.benchmark_id == baseline.benchmark_id


@pytest.mark.parametrize(
    ('updates', 'message'),
    [
        (
            {'metric_unit': 'percent', 'min_score': 0.0, 'max_score': 100.0},
            'canonical metric_unit',
        ),
        (
            {'metric_unit': 'proportion', 'min_score': 0.0, 'max_score': 100.0},
            'canonical bounds',
        ),
    ],
)
def test_percent_conversion_rejects_contradictory_output_metadata(
    updates: dict[str, object], message: str
) -> None:
    payload = _overlay_payload({'AUROC': '99.49'})
    payload['entries'][0]['metrics'][0].update(updates)

    with pytest.raises(ValueError, match=message):
        adapter.ProtocolOverlay.model_validate(payload)


def test_overlay_requires_entries_and_unix_retrieval_time() -> None:
    with pytest.raises(ValueError):
        adapter.ProtocolOverlay.model_validate(
            {
                'schema_version': 1,
                'dump_sha256': DUMP_SHA,
                'registry_revision': REGISTRY_REVISION,
                'retrieved_timestamp': RETRIEVED_TS,
                'entries': [],
            }
        )

    payload = _overlay_payload({'AUROC': '99.49'})
    payload['retrieved_timestamp'] = 'not-an-epoch'
    with pytest.raises(ValueError, match='Unix-epoch'):
        adapter.ProtocolOverlay.model_validate(payload)

    payload = _overlay_payload({'AUROC': '99.49'})
    payload['registry_revision'] = 'main'
    with pytest.raises(ValueError, match='commit SHA'):
        adapter.ProtocolOverlay.model_validate(payload)


def test_output_directory_must_be_fixed_collection_and_empty(tmp_path) -> None:
    with pytest.raises(ValueError, match='must end with'):
        adapter.validate_output_dir(tmp_path / 'out')

    output = tmp_path / 'data' / adapter.COLLECTION_NAME
    output.mkdir(parents=True)
    (output / 'old.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='must be empty'):
        adapter.validate_output_dir(output)

    output_file = tmp_path / 'other' / 'data' / adapter.COLLECTION_NAME
    output_file.parent.mkdir(parents=True)
    output_file.write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='must be a directory'):
        adapter.validate_output_dir(output_file)


@dataclass
class _DumpEntry:
    desc: str
    tag: str
    namespace: str
    copy_stmt: str


class _FakeDump:
    def __init__(
        self, evaluation: dict[str, object], dataset: dict[str, object]
    ):
        self.entries = [
            _DumpEntry(
                'TABLE DATA',
                'evaluations',
                'public',
                'COPY public.evaluations '
                '(id, paper_id, dataset_id, task_id, model_name, '
                'evaluated_on, metrics) FROM stdin;',
            ),
            _DumpEntry(
                'TABLE DATA',
                'datasets',
                'public',
                'COPY public.datasets '
                '(id, name, slug, url, homepage) FROM stdin;',
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


def test_dump_row_width_must_match_copy_columns() -> None:
    metrics = {'AUROC': '99.49'}
    dump = _FakeDump(_source_rows(metrics)[0], _datasets()['drugbank-id'])
    dump._rows['evaluations'][0] = dump._rows['evaluations'][0][:-1]

    with pytest.raises(ValueError):
        list(adapter.table_rows(dump, 'evaluations'))


def test_dump_copy_target_must_match_selected_table() -> None:
    metrics = {'AUROC': '99.49'}
    dump = _FakeDump(_source_rows(metrics)[0], _datasets()['drugbank-id'])
    dump.entries[0].copy_stmt = dump.entries[0].copy_stmt.replace(
        'public.evaluations', 'public.datasets'
    )

    with pytest.raises(ValueError, match='cannot read column order'):
        list(adapter.table_rows(dump, 'evaluations'))


def test_real_pgdumplib_dump_runs_through_cli_path(tmp_path) -> None:
    pgdumplib = pytest.importorskip(
        'pgdumplib',
        reason='manual adapter dependency is not in the core environment',
    )
    metrics = {'AUROC': '99.49'}
    dump_path = tmp_path / 'paperswithcode.dump'
    dump = pgdumplib.new('paperswithcode', 'UTF8', appear_as='15.0')

    evaluations = dump.add_entry(
        'TABLE',
        namespace='public',
        tag='evaluations',
        owner='postgres',
        defn="""CREATE TABLE public.evaluations (
        id text,
        paper_id text,
        dataset_id text,
        task_id text,
        model_name text,
        evaluated_on text,
        metrics jsonb
        );""",
    )
    with dump.table_data_writer(
        evaluations,
        (
            'id',
            'paper_id',
            'dataset_id',
            'task_id',
            'model_name',
            'evaluated_on',
            'metrics',
        ),
    ) as writer:
        writer.append(
            'eval-1',
            'paper-1',
            'drugbank-id',
            'ddi-task',
            'Method Alpha',
            '2024-03-25',
            json.dumps(metrics, separators=(',', ':')),
        )

    datasets = dump.add_entry(
        'TABLE',
        namespace='public',
        tag='datasets',
        owner='postgres',
        defn="""CREATE TABLE public.datasets (
        id text,
        name text,
        slug text,
        url text,
        homepage text
        );""",
    )
    with dump.table_data_writer(
        datasets, ('id', 'name', 'slug', 'url', 'homepage')
    ) as writer:
        writer.append(
            'drugbank-id',
            'DrugBank',
            'drugbank',
            'https://paperswithcode.com/dataset/drugbank',
            'https://go.drugbank.com',
        )
    dump.save(dump_path)

    overlay_payload = _overlay_payload(metrics)
    overlay_payload['dump_sha256'] = adapter.file_sha256(dump_path)
    overlay_path = tmp_path / 'overlay.yaml'
    overlay_path.write_text(
        yaml.safe_dump(overlay_payload, sort_keys=False), encoding='utf-8'
    )
    output_dir = tmp_path / 'data' / 'paperswithcode-drugbank'

    written = adapter.run(
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

    output_files = list(output_dir.glob('example-org/method-alpha/*.json'))
    assert written == 1
    assert len(output_files) == 1
    record = json.loads(output_files[0].read_text(encoding='utf-8'))
    assert record['evaluation_results'][0]['score_details'][
        'score'
    ] == pytest.approx(0.9949)
    assert record['source_metadata']['source_organization_url'] == (
        'https://github.com/paperswithcode'
    )

    validation = subprocess.run(
        [
            sys.executable,
            '-m',
            'every_eval_ever',
            'validate',
            str(output_files[0]),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '0 warnings' in validation.stdout


def test_cli_requires_local_dump_and_external_manifest(tmp_path) -> None:
    dump = tmp_path / 'pwc.dump'
    manifest = tmp_path / 'overlay.yaml'
    output_dir = tmp_path / 'data' / adapter.COLLECTION_NAME
    args = adapter.parse_args(
        [
            '--dump',
            str(dump),
            '--overlay',
            str(manifest),
            '--output-dir',
            str(output_dir),
        ]
    )
    assert args.dump == dump
    assert args.overlay == manifest
    assert args.output_dir == output_dir

    default_args = adapter.parse_args(
        ['--dump', str(dump), '--overlay', str(manifest)]
    )
    assert default_args.output_dir == Path(adapter.DEFAULT_OUTPUT_DIR)
