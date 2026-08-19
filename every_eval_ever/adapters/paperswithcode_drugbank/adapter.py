#!/usr/bin/env python3
"""Convert reviewed Papers with Code DrugBank result cells to EEE.

The adapter consumes a local PwC PostgreSQL dump and a YAML manifest that binds
each selected score cell to reviewed model, metric, split, protocol, and
provenance metadata. It checks the declared split against drug-entity overlap
and does not infer semantics from labels or score distributions.

Run with the adapter extra installed::

    uv run --extra paperswithcode-drugbank python -m \
      every_eval_ever.adapters.paperswithcode_drugbank.adapter \
      --dump /path/to/paperswithcode.dump \
      --overlay /path/to/reviewed-drugbank.yaml \
      --output-dir /tmp/paperswithcode-drugbank/data/paperswithcode-drugbank
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal
from urllib.parse import urlparse

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    save_evaluation_logs,
)

SOURCE_NAME = 'Papers with Code DrugBank'
SOURCE_ORGANIZATION = 'Papers with Code'
SOURCE_ORGANIZATION_URL = 'https://github.com/paperswithcode'
PWC_DATASET_ARCHIVE_URL = 'https://huggingface.co/datasets/pwc-archive/datasets'
DRUGBANK_URL = 'https://go.drugbank.com'
COLLECTION_NAME = 'paperswithcode-drugbank'
DEFAULT_OUTPUT_DIR = f'/tmp/paperswithcode-drugbank/data/{COLLECTION_NAME}'

_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_SAFE_SLUG_RE = re.compile(r'^[a-z0-9][a-z0-9._-]*$')
_MODEL_COMPONENT_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
_REGISTRY_REVISION_RE = re.compile(r'^(?:[0-9a-f]{40}|[0-9a-f]{64})$')


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f'overlay YAML contains duplicate key: {key!r}')
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _sha256(value: str, field: str) -> str:
    value = value.strip()
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f'{field} must be a lowercase 64-character SHA-256')
    return value


def _slug(value: str, field: str) -> str:
    value = value.strip()
    if not _SAFE_SLUG_RE.fullmatch(value):
        raise ValueError(f'{field} must be a lowercase semantic slug')
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


class OverlayAnchors(_StrictModel):
    paper_id: StrictStr | StrictInt | None
    dataset_id: StrictStr | StrictInt
    task_id: StrictStr | StrictInt
    model_name: str
    model_id: str
    developer: str

    @field_validator('model_name', 'model_id', 'developer')
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('model identity anchors must be non-empty')
        return value

    @model_validator(mode='after')
    def valid_model_id(self):
        parts = self.model_id.split('/')
        if len(parts) < 2 or any(
            not _MODEL_COMPONENT_RE.fullmatch(p) for p in parts
        ):
            raise ValueError(
                'model_id must contain at least two safe namespace/model components'
            )
        return self


class ProtocolQualification(_StrictModel):
    benchmark_id: str
    split_id: Literal['transductive', 'inductive-s1', 'inductive-s2']
    study_id: str
    protocol_id: str
    task_id: str
    task_type: str
    candidate_label_space: str
    drug_entity_overlap: Literal['both-seen', 'one-unseen', 'both-unseen']
    pair_overlap: str
    relation_class_overlap: str
    temporal_ordering: str
    negative_sampling: str
    split_preprocessing: str

    @field_validator('benchmark_id', 'study_id', 'protocol_id', 'task_id')
    @classmethod
    def semantic_slug(cls, value: str, info) -> str:
        return _slug(value, info.field_name)

    @field_validator(
        'task_type',
        'candidate_label_space',
        'pair_overlap',
        'relation_class_overlap',
        'temporal_ordering',
        'negative_sampling',
        'split_preprocessing',
    )
    @classmethod
    def semantic_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('protocol semantic fields must be non-empty')
        return value

    @model_validator(mode='after')
    def split_matches_entity_overlap(self):
        expected_overlap = {
            'transductive': 'both-seen',
            'inductive-s1': 'both-unseen',
            'inductive-s2': 'one-unseen',
        }[self.split_id]
        if self.drug_entity_overlap != expected_overlap:
            raise ValueError(
                f'split_id {self.split_id!r} requires '
                f'drug_entity_overlap {expected_overlap!r}'
            )
        return self

    @property
    def generalization_regime(self) -> Literal['transductive', 'inductive']:
        return (
            'transductive' if self.split_id == 'transductive' else 'inductive'
        )

    def semantic_sha256(self) -> str:
        return _canonical_sha256(self.model_dump(mode='json'))


class ProtocolEvidence(_StrictModel):
    source_url: str
    source_locator: str
    review_note: str

    @field_validator('source_url')
    @classmethod
    def absolute_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlparse(value)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
            raise ValueError('source_url must be an absolute HTTP(S) URL')
        return value

    @field_validator('source_locator', 'review_note')
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError(
                'evidence locator and review note must be non-empty'
            )
        return value


class OverlayMetric(_StrictModel):
    source_name: str
    metric_id: str
    metric_name: str
    metric_kind: str
    metric_unit: str | None = None
    lower_is_better: bool
    min_score: float
    max_score: float
    source_scale: Literal['identity', 'percent'] = 'identity'

    @field_validator('source_name', 'metric_name', 'metric_kind')
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('metric text fields must be non-empty')
        return value

    @field_validator('metric_id')
    @classmethod
    def metric_slug(cls, value: str) -> str:
        return _slug(value, 'metric_id')

    @model_validator(mode='after')
    def valid_bounds(self):
        if not math.isfinite(self.min_score) or not math.isfinite(
            self.max_score
        ):
            raise ValueError('metric bounds must be finite')
        if self.max_score <= self.min_score:
            raise ValueError('metric max_score must be greater than min_score')
        if self.source_scale == 'percent':
            if self.metric_unit != 'proportion':
                raise ValueError(
                    "percent source_scale requires canonical metric_unit 'proportion'"
                )
            if self.min_score != 0.0 or self.max_score != 1.0:
                raise ValueError(
                    'percent source_scale requires canonical bounds [0.0, 1.0]'
                )
        return self

    @property
    def scale_factor(self) -> float:
        return 0.01 if self.source_scale == 'percent' else 1.0


class ProtocolOverlayEntry(_StrictModel):
    pwc_evaluation_id: StrictStr | StrictInt
    anchors: OverlayAnchors
    source_metrics_sha256: str
    qualification: ProtocolQualification
    evidence: ProtocolEvidence
    metrics: list[OverlayMetric] = Field(min_length=1)

    @field_validator('pwc_evaluation_id')
    @classmethod
    def evaluation_id_present(cls, value):
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError('pwc_evaluation_id must be non-empty')
        return value

    @field_validator('source_metrics_sha256')
    @classmethod
    def metrics_hash(cls, value: str) -> str:
        return _sha256(value, 'source_metrics_sha256')

    @model_validator(mode='after')
    def unique_metric_names(self):
        source_names = [metric.source_name for metric in self.metrics]
        if len(source_names) != len(set(source_names)):
            raise ValueError(
                'metric source_name selectors must be unique within an entry'
            )
        metric_ids = [metric.metric_id for metric in self.metrics]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError(
                'canonical metric_id selectors must be unique within an entry'
            )
        return self


class ProtocolOverlay(_StrictModel):
    schema_version: Literal[2]
    dump_sha256: str
    registry_revision: str
    retrieved_timestamp: str
    entries: list[ProtocolOverlayEntry] = Field(min_length=1)

    @field_validator('dump_sha256')
    @classmethod
    def dump_hash(cls, value: str) -> str:
        return _sha256(value, 'dump_sha256')

    @field_validator('registry_revision')
    @classmethod
    def registry_commit(cls, value: str) -> str:
        value = value.strip()
        if not _REGISTRY_REVISION_RE.fullmatch(value):
            raise ValueError(
                'registry_revision must be a 40- or 64-character commit SHA'
            )
        return value

    @field_validator('retrieved_timestamp')
    @classmethod
    def unix_epoch(cls, value: str) -> str:
        value = value.strip()
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(
                'retrieved_timestamp must be a Unix-epoch string'
            ) from exc
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError(
                'retrieved_timestamp must be a finite non-negative epoch'
            )
        return value

    @model_validator(mode='after')
    def consistent_entries(self):
        seen: set[tuple[str, str]] = set()
        seen_canonical_metrics: set[tuple[str, str, str]] = set()
        model_by_evaluation: dict[str, str] = {}
        developer_by_model: dict[str, str] = {}
        protocol_by_benchmark: dict[str, str] = {}
        for entry in self.entries:
            evaluation_id = str(entry.pwc_evaluation_id)
            model_id = entry.anchors.model_id
            prior_model = model_by_evaluation.setdefault(
                evaluation_id, model_id
            )
            if prior_model != model_id:
                raise ValueError(
                    f'PwC evaluation {evaluation_id} is assigned to multiple '
                    f'canonical model ids: {prior_model!r} and {model_id!r}'
                )
            prior_developer = developer_by_model.setdefault(
                model_id, entry.anchors.developer
            )
            if prior_developer != entry.anchors.developer:
                raise ValueError(
                    f'inconsistent reviewed developer for {model_id}'
                )
            benchmark_id = entry.qualification.benchmark_id
            protocol_digest = entry.qualification.semantic_sha256()
            prior_digest = protocol_by_benchmark.setdefault(
                benchmark_id, protocol_digest
            )
            if prior_digest != protocol_digest:
                raise ValueError(
                    'canonical benchmark_id is reused for conflicting '
                    f'protocol semantics: {benchmark_id!r}'
                )
            for metric in entry.metrics:
                key = (evaluation_id, metric.source_name)
                if key in seen:
                    raise ValueError(
                        f'PwC source score cell is selected more than once: {key}'
                    )
                seen.add(key)
                metric_key = (
                    evaluation_id,
                    protocol_digest,
                    metric.metric_id,
                )
                if metric_key in seen_canonical_metrics:
                    raise ValueError(
                        'canonical metric_id is selected more than once for '
                        f'one source protocol: {metric.metric_id!r}'
                    )
                seen_canonical_metrics.add(metric_key)
        return self


@dataclass
class _ModelGroup:
    developer: str
    result_ids: set[str] = field(default_factory=set)
    model_names: set[str] = field(default_factory=set)
    results: list[EvaluationResult] = field(default_factory=list)


def stringify(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'null'
    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
            allow_nan=False,
        )
    return str(value)


def stringify_details(details: dict[str, Any]) -> dict[str, str]:
    return {
        key: stringify(value)
        for key, value in details.items()
        if value is not None
    }


def source_metrics_sha256(metrics: dict[str, Any]) -> str:
    return _canonical_sha256(metrics)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_overlay(path: str | Path) -> tuple[ProtocolOverlay, str]:
    raw = Path(path).read_bytes()
    payload = yaml.load(raw, Loader=_UniqueKeySafeLoader)
    if not isinstance(payload, dict):
        raise ValueError('overlay YAML must decode to an object')
    return ProtocolOverlay.model_validate(payload), hashlib.sha256(
        raw
    ).hexdigest()


def load_dump(dump_path: str | Path):
    import pgdumplib

    return pgdumplib.load(str(dump_path))


def _columns_for(dump, table: str) -> list[str]:
    for entry in dump.entries:
        if (
            entry.desc != 'TABLE DATA'
            or entry.namespace != 'public'
            or entry.tag != table
        ):
            continue
        copy_stmt = (entry.copy_stmt or '').strip()
        prefix = f'COPY public.{table} ('
        suffix = ') FROM stdin;'
        if not copy_stmt.startswith(prefix) or not copy_stmt.endswith(suffix):
            raise ValueError(f'cannot read column order for public.{table}')
        column_list = copy_stmt[len(prefix) : -len(suffix)]
        columns = [
            column.strip().strip('"') for column in column_list.split(',')
        ]
        if not columns or any(not column for column in columns):
            raise ValueError(f'public.{table} has an invalid COPY column list')
        if len(columns) != len(set(columns)):
            raise ValueError(
                f'public.{table} COPY column list contains duplicates'
            )
        return columns
    raise KeyError(f'table public.{table} not found in dump')


def table_rows(dump, table: str) -> Iterator[dict[str, Any]]:
    columns = _columns_for(dump, table)
    for row in dump.table_data('public', table):
        yield dict(zip(columns, row, strict=True))


def load_source_context(
    dump, overlay: ProtocolOverlay
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    wanted_evals = {str(entry.pwc_evaluation_id) for entry in overlay.entries}
    evaluations: dict[str, dict[str, Any]] = {}
    for row in table_rows(dump, 'evaluations'):
        row_id = str(row.get('id'))
        if row_id in wanted_evals:
            if row_id in evaluations:
                raise ValueError(
                    f'duplicate PwC evaluation id in dump: {row_id}'
                )
            evaluations[row_id] = row
    missing = sorted(wanted_evals - set(evaluations))
    if missing:
        raise ValueError(
            f'overlay references missing PwC evaluations: {missing}'
        )

    wanted_datasets = {
        str(entry.anchors.dataset_id) for entry in overlay.entries
    }
    datasets: dict[str, dict[str, Any]] = {}
    for row in table_rows(dump, 'datasets'):
        row_id = str(row.get('id'))
        if row_id in wanted_datasets:
            if row_id in datasets:
                raise ValueError(f'duplicate PwC dataset id in dump: {row_id}')
            datasets[row_id] = row
    missing = sorted(wanted_datasets - set(datasets))
    if missing:
        raise ValueError(f'overlay references missing PwC datasets: {missing}')
    return list(evaluations.values()), datasets


def _metrics_object(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get('metrics')
    try:
        if isinstance(raw, dict):
            metrics = raw
        elif raw:
            metrics = json.loads(raw)
        else:
            metrics = {}
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'PwC evaluation {row.get("id")} has invalid metrics JSON'
        ) from exc
    if not isinstance(metrics, dict) or any(
        not isinstance(name, str) for name in metrics
    ):
        raise ValueError(
            f'PwC evaluation {row.get("id")} metrics must be a string-keyed object'
        )
    return metrics


def parse_metric_value(raw: Any) -> tuple[float, str | None, bool]:
    if raw is None or isinstance(raw, bool):
        raise ValueError('metric value must be a finite number')
    if isinstance(raw, (int, float)):
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError('metric value must be finite')
        return value, None, False
    text = str(raw).strip()
    if not text:
        raise ValueError('metric value is empty')
    uncertainty = None
    for separator in ('±', '+/-', '+-'):
        if separator in text:
            text, _, uncertainty = text.partition(separator)
            text, uncertainty = text.strip(), uncertainty.strip() or None
            break
    has_percent_marker = text.endswith('%')
    text = text.removesuffix('%').strip()
    if ',' in text:
        raise ValueError(
            f'metric value uses ambiguous comma formatting: {raw!r}'
        )
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f'metric value is not numeric: {raw!r}') from exc
    if not math.isfinite(value):
        raise ValueError('metric value must be finite')
    return value, uncertainty, has_percent_marker


def _assert_anchor(
    evaluation_id: str, field: str, expected: Any, actual: Any
) -> None:
    expected = None if expected is None else str(expected)
    actual = None if actual is None else str(actual)
    if expected != actual:
        raise ValueError(
            f'overlay anchor drift for PwC evaluation {evaluation_id}: '
            f'{field} expected {expected!r}, got {actual!r}'
        )


def build_source_data(dataset: dict[str, Any]) -> SourceDataUrl:
    return SourceDataUrl(
        dataset_name='DrugBank',
        source_type='url',
        url=[DRUGBANK_URL],
        additional_details=stringify_details(
            {
                'raw_dataset_id': dataset.get('id'),
                'raw_dataset_url': dataset.get('url'),
                'raw_dataset_homepage': dataset.get('homepage'),
                'pwc_dataset_slug': dataset.get('slug'),
            }
        ),
    )


def build_source_metadata(
    overlay: ProtocolOverlay, overlay_sha256: str, dump_file: str | None
) -> SourceMetadata:
    return SourceMetadata(
        source_name=SOURCE_NAME,
        source_type='documentation',
        source_organization_name=SOURCE_ORGANIZATION,
        source_organization_url=SOURCE_ORGANIZATION_URL,
        evaluator_relationship=EvaluatorRelationship.third_party,
        additional_details=stringify_details(
            {
                'source_role': 'aggregator',
                'dump_sha256': overlay.dump_sha256,
                'registry_revision': overlay.registry_revision,
                'overlay_sha256': overlay_sha256,
                'source_dump_file': dump_file,
                'pwc_data_archive_url': PWC_DATASET_ARCHIVE_URL,
                'qualification_policy': (
                    'explicit source-cell manifest with no protocol inference '
                    'from the DrugBank label or score values'
                ),
            }
        ),
    )


def _metric_result_suffix(metric: OverlayMetric) -> str:
    slug = (
        re.sub(r'[^a-z0-9]+', '-', metric.source_name.lower()).strip('-')
        or 'metric'
    )
    digest = _canonical_sha256(metric.model_dump(mode='json'))
    return f'{slug}-{metric.metric_id}-{digest}'


def _bundle_evaluation_id(dump_sha256: str, result_ids: Iterable[str]) -> str:
    selected_result_ids = sorted(set(result_ids))
    if not selected_result_ids:
        raise ValueError(
            'cannot build an evaluation id without selected metric results'
        )
    source_bundle_sha256 = _canonical_sha256(selected_result_ids)
    return f'paperswithcode-drugbank/{dump_sha256}/{source_bundle_sha256}'


def _build_result(
    entry: ProtocolOverlayEntry,
    row: dict[str, Any],
    dataset: dict[str, Any],
    metric: OverlayMetric,
    raw_value: Any,
) -> EvaluationResult:
    source_value, uncertainty, percent = parse_metric_value(raw_value)
    if percent and metric.source_scale != 'percent':
        raise ValueError(
            f'PwC evaluation {entry.pwc_evaluation_id} metric {metric.source_name!r} '
            'has a percent marker but the reviewed source_scale is not percent'
        )
    score = source_value * metric.scale_factor
    if not metric.min_score <= score <= metric.max_score:
        raise ValueError(
            f'PwC evaluation {entry.pwc_evaluation_id} metric {metric.source_name!r} '
            f'converts to {score}, outside reviewed canonical range '
            f'[{metric.min_score}, {metric.max_score}]'
        )

    q = entry.qualification
    protocol_digest = q.semantic_sha256()
    details = stringify_details(
        {
            'pwc_evaluation_id': entry.pwc_evaluation_id,
            'pwc_paper_id': entry.anchors.paper_id,
            'pwc_dataset_id': entry.anchors.dataset_id,
            'pwc_task_id': entry.anchors.task_id,
            'pwc_model_name': entry.anchors.model_name,
            'source_metrics_sha256': entry.source_metrics_sha256,
            'protocol_semantic_sha256': protocol_digest,
            'protocol_study_id': q.study_id,
            'protocol_id': q.protocol_id,
            'split_id': q.split_id,
            'generalization_regime': q.generalization_regime,
            'drug_entity_overlap': q.drug_entity_overlap,
            'pair_overlap': q.pair_overlap,
            'relation_class_overlap': q.relation_class_overlap,
            'temporal_ordering': q.temporal_ordering,
            'negative_sampling': q.negative_sampling,
            'split_preprocessing': q.split_preprocessing,
            'task_id': q.task_id,
            'task_type': q.task_type,
            'candidate_label_space': q.candidate_label_space,
            'protocol_evidence_url': entry.evidence.source_url,
            'protocol_evidence_locator': entry.evidence.source_locator,
            'protocol_review_note': entry.evidence.review_note,
            'raw_value': raw_value,
            'reported_uncertainty': uncertainty,
            'reviewed_source_scale': metric.source_scale,
            'applied_scale_factor': metric.scale_factor,
        }
    )
    return EvaluationResult(
        evaluation_result_id=(
            f'paperswithcode-drugbank.{entry.pwc_evaluation_id}.'
            f'{_metric_result_suffix(metric)}.{protocol_digest}'
        ),
        evaluation_name=q.benchmark_id,
        source_data=build_source_data(dataset),
        evaluation_timestamp=str(row.get('evaluated_on'))
        if row.get('evaluated_on')
        else None,
        metric_config=MetricConfig(
            evaluation_description=(
                f'{metric.metric_name} for DrugBank protocol {q.protocol_id} '
                f'({q.split_id}).'
            ),
            metric_id=metric.metric_id,
            metric_name=metric.metric_name,
            metric_kind=metric.metric_kind,
            metric_unit=metric.metric_unit,
            lower_is_better=metric.lower_is_better,
            score_type=ScoreType.continuous,
            min_score=metric.min_score,
            max_score=metric.max_score,
            additional_details={
                'source_metric_name': metric.source_name,
                'reviewed_source_scale': metric.source_scale,
            },
        ),
        score_details=ScoreDetails(
            score=score, uncertainty=None, details=details
        ),
    )


def build_logs(
    evaluations: Iterable[dict[str, Any]],
    datasets_by_id: dict[str, dict[str, Any]],
    overlay: ProtocolOverlay,
    overlay_sha256: str,
    *,
    dump_file: str | None = None,
) -> list[EvaluationLog]:
    rows: dict[str, dict[str, Any]] = {}
    for row in evaluations:
        evaluation_id = str(row.get('id'))
        if evaluation_id in rows:
            raise ValueError(
                f'duplicate PwC evaluation id in source: {evaluation_id}'
            )
        rows[evaluation_id] = row

    groups: dict[str, _ModelGroup] = {}
    for entry in overlay.entries:
        evaluation_id = str(entry.pwc_evaluation_id)
        row = rows.get(evaluation_id)
        if row is None:
            raise ValueError(
                f'overlay references missing PwC evaluation {evaluation_id}'
            )
        for anchor_field in ('paper_id', 'dataset_id', 'task_id', 'model_name'):
            _assert_anchor(
                evaluation_id,
                anchor_field,
                getattr(entry.anchors, anchor_field),
                row.get(anchor_field),
            )

        dataset = datasets_by_id.get(str(entry.anchors.dataset_id))
        if dataset is None:
            raise ValueError(
                f'overlay references missing PwC dataset {entry.anchors.dataset_id!r}'
            )
        labels = {
            str(dataset[field]).strip().casefold()
            for field in ('name', 'slug')
            if dataset.get(field)
        }
        if labels != {'drugbank'}:
            raise ValueError(
                f'overlay entry {evaluation_id} does not target DrugBank'
            )

        metrics = _metrics_object(row)
        actual_hash = source_metrics_sha256(metrics)
        if actual_hash != entry.source_metrics_sha256:
            raise ValueError(
                f'PwC evaluation {evaluation_id} metrics payload drift: expected '
                f'{entry.source_metrics_sha256}, got {actual_hash}'
            )

        model_id = entry.anchors.model_id
        group = groups.setdefault(
            model_id, _ModelGroup(entry.anchors.developer)
        )
        group.model_names.add(entry.anchors.model_name)
        for metric in entry.metrics:
            if metric.source_name not in metrics:
                raise ValueError(
                    f'PwC evaluation {evaluation_id} is missing selected metric '
                    f'{metric.source_name!r}'
                )
            result = _build_result(
                entry,
                row,
                dataset,
                metric,
                metrics[metric.source_name],
            )
            if result.evaluation_result_id is None:
                raise ValueError('DrugBank result is missing its stable id')
            if result.evaluation_result_id in group.result_ids:
                raise ValueError(
                    'duplicate DrugBank evaluation_result_id: '
                    f'{result.evaluation_result_id}'
                )
            group.result_ids.add(result.evaluation_result_id)
            group.results.append(result)

    logs: list[EvaluationLog] = []
    for model_id, group in sorted(groups.items()):
        raw_model_names = sorted(group.model_names)
        log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=_bundle_evaluation_id(
                overlay.dump_sha256,
                group.result_ids,
            ),
            retrieved_timestamp=overlay.retrieved_timestamp,
            source_metadata=build_source_metadata(
                overlay, overlay_sha256, dump_file
            ),
            eval_library=EvalLibrary(name='unknown', version='unknown'),
            model_info=ModelInfo(
                name=raw_model_names[0],
                id=model_id,
                developer=group.developer,
                additional_details={
                    'raw_model_name': raw_model_names[0],
                    'raw_model_names': json.dumps(
                        raw_model_names,
                        ensure_ascii=False,
                        separators=(',', ':'),
                    ),
                    'identity_source': 'protocol_qualification_manifest',
                },
            ),
            evaluation_results=sorted(
                group.results,
                key=lambda result: result.evaluation_result_id or '',
            ),
        )
        logs.append(log)
    if not logs:
        raise ValueError(
            'DrugBank qualification manifest selected zero source score cells'
        )
    return logs


def validate_output_dir(output_dir: Path) -> None:
    if output_dir.name != COLLECTION_NAME or output_dir.parent.name != 'data':
        raise ValueError(
            f'--output-dir must end with data/{COLLECTION_NAME}: {output_dir}'
        )
    if not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise ValueError(f'output path must be a directory: {output_dir}')
    if any(output_dir.iterdir()):
        raise ValueError(f'output directory must be empty: {output_dir}')


def export(logs: Iterable[EvaluationLog], output_dir: Path) -> list[Path]:
    outputs: list[EvaluationLogOutput] = []
    for log in logs:
        model_developer, model_name = log.model_info.id.split('/', 1)
        outputs.append(
            EvaluationLogOutput(
                eval_log=log,
                base_dir=output_dir,
                developer=model_developer,
                model_name=model_name,
            )
        )
    return save_evaluation_logs(outputs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Convert explicitly qualified PwC DrugBank transductive, '
            'inductive-S1, and inductive-S2 results.'
        )
    )
    parser.add_argument(
        '--dump',
        type=Path,
        required=True,
        help='Local PwC PostgreSQL custom-format dump.',
    )
    parser.add_argument(
        '--overlay',
        type=Path,
        required=True,
        help='External YAML manifest binding exact PwC cells to DrugBank protocol semantics.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output collection directory (default: {DEFAULT_OUTPUT_DIR}).',
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    overlay, overlay_sha256 = load_overlay(args.overlay)
    actual_dump_sha256 = file_sha256(args.dump)
    if actual_dump_sha256 != overlay.dump_sha256:
        raise ValueError(
            'PwC dump SHA-256 does not match the qualification manifest: '
            f'expected {overlay.dump_sha256}, got {actual_dump_sha256}'
        )
    evaluations, datasets = load_source_context(load_dump(args.dump), overlay)
    logs = build_logs(
        evaluations, datasets, overlay, overlay_sha256, dump_file=args.dump.name
    )
    validate_output_dir(args.output_dir)
    paths = export(logs, args.output_dir)
    for path in paths:
        print(path)
    return len(paths)


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} Papers with Code DrugBank model log(s).')
