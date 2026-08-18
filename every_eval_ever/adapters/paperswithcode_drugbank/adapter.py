#!/usr/bin/env python3
"""Convert explicitly qualified Papers with Code DrugBank result cells to EEE.

The adapter is deliberately DrugBank-only and independent of the generic PwC
adapter. It consumes a local PwC PostgreSQL dump plus an external YAML manifest
that binds exact source cells to model identity, metric scale, source evidence,
and drug-entity overlap. ``transductive`` versus ``inductive`` is derived only
from that explicit overlap; the adapter never infers protocol from the DrugBank
label or score distributions.

Run manually with ``pgdumplib`` available::

    uv run --with 'pgdumplib>=4.0.0' python -m \
      every_eval_ever.adapters.paperswithcode_drugbank.adapter \
      --dump /path/to/paperswithcode.dump \
      --overlay /path/to/reviewed-drugbank.yaml \
      --output-dir /tmp/paperswithcode-drugbank
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    SourceConversionResult,
    sanitize_filename,
    save_evaluation_logs,
)

SOURCE_NAME = 'Papers with Code DrugBank'
SOURCE_ORGANIZATION = 'Papers with Code'
SOURCE_ORGANIZATION_URL = 'https://paperswithcode.com'
PWC_DATASET_URL = 'https://paperswithcode.com/dataset/drugbank'
DRUGBANK_URL = 'https://go.drugbank.com'
DEFAULT_OUTPUT_DIR = 'data/paperswithcode-drugbank'

_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_SAFE_SLUG_RE = re.compile(r'^[a-z0-9][a-z0-9._-]*$')
_MODEL_COMPONENT_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


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


class OverlayAnchors(_StrictModel):
    paper_id: str | int | None
    dataset_id: str | int
    task_id: str | int
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
        if len(parts) != 2 or any(not _MODEL_COMPONENT_RE.fullmatch(p) for p in parts):
            raise ValueError('model_id must be a two-component developer/model id')
        if parts[0] != self.developer:
            raise ValueError('model_id developer component must equal developer')
        return self


class ProtocolQualification(_StrictModel):
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

    @field_validator('study_id', 'protocol_id', 'task_id')
    @classmethod
    def semantic_slug(cls, value: str, info) -> str:
        return _slug(value, info.field_name)

    @field_validator(
        'task_type', 'candidate_label_space', 'pair_overlap',
        'relation_class_overlap', 'temporal_ordering', 'negative_sampling',
        'split_preprocessing',
    )
    @classmethod
    def semantic_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('protocol semantic fields must be non-empty')
        return value

    @property
    def generalization_regime(self) -> Literal['transductive', 'inductive']:
        return 'transductive' if self.drug_entity_overlap == 'both-seen' else 'inductive'

    def evaluation_name(self) -> str:
        return (
            f'paperswithcode-drugbank.{self.study_id}.'
            f'{self.generalization_regime}.{self.protocol_id}'
        )


class ProtocolEvidence(_StrictModel):
    source_url: str
    source_locator: str
    review_note: str

    @field_validator('source_url')
    @classmethod
    def absolute_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlparse(value)
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
            raise ValueError('source_url must be an absolute HTTP(S) URL')
        return value

    @field_validator('source_locator', 'review_note')
    @classmethod
    def nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('evidence locator and review note must be non-empty')
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
        if not math.isfinite(self.min_score) or not math.isfinite(self.max_score):
            raise ValueError('metric bounds must be finite')
        if self.max_score <= self.min_score:
            raise ValueError('metric max_score must be greater than min_score')
        return self

    @property
    def scale_factor(self) -> float:
        return 0.01 if self.source_scale == 'percent' else 1.0


class ProtocolOverlayEntry(_StrictModel):
    pwc_evaluation_id: str | int
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
        names = [metric.source_name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError('metric source_name selectors must be unique within an entry')
        return self


class ProtocolOverlay(_StrictModel):
    schema_version: Literal[1]
    dump_sha256: str
    retrieved_timestamp: str
    entries: list[ProtocolOverlayEntry] = Field(min_length=1)

    @field_validator('dump_sha256')
    @classmethod
    def dump_hash(cls, value: str) -> str:
        return _sha256(value, 'dump_sha256')

    @field_validator('retrieved_timestamp')
    @classmethod
    def unix_epoch(cls, value: str) -> str:
        value = value.strip()
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError('retrieved_timestamp must be a Unix-epoch string') from exc
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError('retrieved_timestamp must be a finite non-negative epoch')
        return value

    @model_validator(mode='after')
    def unique_cells(self):
        seen: set[tuple[str, str]] = set()
        for entry in self.entries:
            for metric in entry.metrics:
                key = (str(entry.pwc_evaluation_id), metric.source_name)
                if key in seen:
                    raise ValueError(f'PwC source score cell is selected more than once: {key}')
                seen.add(key)
        return self


@dataclass(frozen=True)
class LogBundle:
    log: EvaluationLog
    developer: str
    model: str


def stringify(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if value is None:
        return 'null'
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(',', ':'))
    return str(value)


def stringify_details(details: dict[str, Any]) -> dict[str, str]:
    return {key: stringify(value) for key, value in details.items() if value is not None}


def slugify(value: Any, fallback: str = 'unknown') -> str:
    raw = str(value if value not in (None, '') else fallback).strip().lower()
    raw = sanitize_filename(raw).replace('&', 'and')
    raw = re.sub(r'[\s_]+', '-', raw)
    raw = re.sub(r'[^a-z0-9.\-]+', '-', raw)
    return re.sub(r'-{2,}', '-', raw).strip('-') or fallback


def source_metrics_sha256(metrics: dict[str, Any]) -> str:
    payload = json.dumps(
        metrics, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_overlay(path: str | Path) -> tuple[ProtocolOverlay, str]:
    raw = Path(path).read_bytes()
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError('overlay YAML must decode to an object')
    return ProtocolOverlay.model_validate(payload), hashlib.sha256(raw).hexdigest()


def _parse_columns(create_defn: str) -> list[str]:
    body = create_defn.split('(', 1)[1]
    return [
        line.strip().rstrip(',').split()[0]
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith(('CONSTRAINT', ')'))
    ]


def load_dump(dump_path: str | Path):
    import pgdumplib

    return pgdumplib.load(str(dump_path))


def _columns_for(dump, table: str) -> list[str]:
    for entry in dump.entries:
        if entry.desc == 'TABLE' and entry.tag == table:
            return _parse_columns(entry.defn)
    raise KeyError(f'table public.{table} not found in dump')


def table_rows(dump, table: str) -> Iterator[dict[str, Any]]:
    columns = _columns_for(dump, table)
    for row in dump.table_data('public', table):
        yield dict(zip(columns, row))


def load_source_context(
    dump, overlay: ProtocolOverlay
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    wanted_evals = {str(entry.pwc_evaluation_id) for entry in overlay.entries}
    evaluations: dict[str, dict[str, Any]] = {}
    for row in table_rows(dump, 'evaluations'):
        row_id = str(row.get('id'))
        if row_id in wanted_evals:
            if row_id in evaluations:
                raise ValueError(f'duplicate PwC evaluation id in dump: {row_id}')
            evaluations[row_id] = row
    missing = sorted(wanted_evals - set(evaluations))
    if missing:
        raise ValueError(f'overlay references missing PwC evaluations: {missing}')

    wanted_datasets = {str(entry.anchors.dataset_id) for entry in overlay.entries}
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
    for dataset_id, dataset in datasets.items():
        name = str(dataset.get('name') or '').strip().lower()
        slug = str(dataset.get('slug') or '').strip().lower()
        if name != 'drugbank' and slug != 'drugbank':
            raise ValueError(f'PwC dataset {dataset_id} is not DrugBank')
    return list(evaluations.values()), datasets


def _metrics_object(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get('metrics')
    try:
        metrics = raw if isinstance(raw, dict) else json.loads(raw) if raw else {}
    except (TypeError, ValueError) as exc:
        raise ValueError(f'PwC evaluation {row.get("id")} has invalid metrics JSON') from exc
    if not isinstance(metrics, dict) or any(not isinstance(name, str) for name in metrics):
        raise ValueError(f'PwC evaluation {row.get("id")} metrics must be a string-keyed object')
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
    percent = text.endswith('%')
    text = text.rstrip('%').strip()
    text = text.replace(',', '') if ('.' in text or text.count(',') > 1) else text.replace(',', '.')
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f'metric value is not numeric: {raw!r}') from exc
    if not math.isfinite(value):
        raise ValueError('metric value must be finite')
    return value, uncertainty, percent


def _assert_anchor(evaluation_id: str, field: str, expected: Any, actual: Any) -> None:
    expected = None if expected is None else str(expected)
    actual = None if actual is None else str(actual)
    if expected != actual:
        raise ValueError(
            f'overlay anchor drift for PwC evaluation {evaluation_id}: '
            f'{field} expected {expected!r}, got {actual!r}'
        )


def build_source_data(dataset: dict[str, Any]) -> SourceDataUrl:
    urls = []
    for candidate in (
        dataset.get('url'), dataset.get('homepage'), dataset.get('paper_url'),
        DRUGBANK_URL, PWC_DATASET_URL,
    ):
        if candidate and str(candidate) not in urls:
            urls.append(str(candidate))
    return SourceDataUrl(
        dataset_name=dataset.get('name') or 'DrugBank',
        source_type='url',
        url=urls,
        additional_details=stringify_details(
            {
                'raw_dataset_id': dataset.get('id'),
                'pwc_dataset_slug': dataset.get('slug'),
                'pwc_dataset_url': PWC_DATASET_URL,
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
                'overlay_sha256': overlay_sha256,
                'source_dump_file': dump_file,
                'qualification_policy': (
                    'explicit source-cell manifest; no protocol inference from '
                    'DrugBank label or score values'
                ),
            }
        ),
    )


def _metric_result_suffix(source_name: str) -> str:
    digest = hashlib.sha256(source_name.encode()).hexdigest()[:8]
    return f'{slugify(source_name)}-{digest}'


def _build_result(
    entry: ProtocolOverlayEntry,
    row: dict[str, Any],
    dataset: dict[str, Any],
    metric: OverlayMetric,
    raw_value: Any,
    overlay: ProtocolOverlay,
    overlay_sha256: str,
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
    details = stringify_details(
        {
            'pwc_evaluation_id': entry.pwc_evaluation_id,
            'pwc_paper_id': entry.anchors.paper_id,
            'pwc_dataset_id': entry.anchors.dataset_id,
            'pwc_task_id': entry.anchors.task_id,
            'source_metrics_sha256': entry.source_metrics_sha256,
            'dump_sha256': overlay.dump_sha256,
            'overlay_sha256': overlay_sha256,
            'protocol_study_id': q.study_id,
            'protocol_id': q.protocol_id,
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
            f'{_metric_result_suffix(metric.source_name)}'
        ),
        evaluation_name=q.evaluation_name(),
        source_data=build_source_data(dataset),
        evaluation_timestamp=str(row.get('evaluated_on')) if row.get('evaluated_on') else None,
        metric_config=MetricConfig(
            evaluation_description=(
                f'{metric.metric_name} for DrugBank protocol {q.protocol_id} '
                f'({q.generalization_regime}).'
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
                'protocol_id': q.protocol_id,
                'generalization_regime': q.generalization_regime,
            },
        ),
        score_details=ScoreDetails(score=score, uncertainty=None, details=details),
    )


def build_logs(
    evaluations: Iterable[dict[str, Any]],
    datasets_by_id: dict[str, dict[str, Any]],
    overlay: ProtocolOverlay,
    overlay_sha256: str,
    *,
    dump_file: str | None = None,
) -> SourceConversionResult[LogBundle]:
    rows: dict[str, dict[str, Any]] = {}
    for row in evaluations:
        evaluation_id = str(row.get('id'))
        if evaluation_id in rows:
            raise ValueError(f'duplicate PwC evaluation id in source: {evaluation_id}')
        rows[evaluation_id] = row

    grouped: dict[str, list[EvaluationResult]] = defaultdict(list)
    anchors_by_model: dict[str, OverlayAnchors] = {}
    selected_cells = 0
    for entry in overlay.entries:
        evaluation_id = str(entry.pwc_evaluation_id)
        row = rows.get(evaluation_id)
        if row is None:
            raise ValueError(f'overlay references missing PwC evaluation {evaluation_id}')
        for field in ('paper_id', 'dataset_id', 'task_id', 'model_name'):
            _assert_anchor(evaluation_id, field, getattr(entry.anchors, field), row.get(field))

        dataset = datasets_by_id.get(str(entry.anchors.dataset_id))
        if dataset is None:
            raise ValueError(f'overlay references missing PwC dataset {entry.anchors.dataset_id!r}')
        name = str(dataset.get('name') or '').strip().lower()
        slug = str(dataset.get('slug') or '').strip().lower()
        if name != 'drugbank' and slug != 'drugbank':
            raise ValueError(f'overlay entry {evaluation_id} does not target DrugBank')

        metrics = _metrics_object(row)
        actual_hash = source_metrics_sha256(metrics)
        if actual_hash != entry.source_metrics_sha256:
            raise ValueError(
                f'PwC evaluation {evaluation_id} metrics payload drift: expected '
                f'{entry.source_metrics_sha256}, got {actual_hash}'
            )

        model_id = entry.anchors.model_id
        prior = anchors_by_model.setdefault(model_id, entry.anchors)
        if prior.developer != entry.anchors.developer or prior.model_name != entry.anchors.model_name:
            raise ValueError(f'inconsistent reviewed model identity for {model_id}')
        for metric in entry.metrics:
            if metric.source_name not in metrics:
                raise ValueError(
                    f'PwC evaluation {evaluation_id} is missing selected metric '
                    f'{metric.source_name!r}'
                )
            grouped[model_id].append(
                _build_result(
                    entry, row, dataset, metric, metrics[metric.source_name],
                    overlay, overlay_sha256,
                )
            )
            selected_cells += 1

    bundles = []
    for model_id, results in sorted(grouped.items()):
        anchors = anchors_by_model[model_id]
        log = EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=(
                f'paperswithcode-drugbank/{model_id.replace("/", "_")}/'
                f'{overlay.dump_sha256[:16]}-{overlay_sha256[:16]}'
            ),
            retrieved_timestamp=overlay.retrieved_timestamp,
            source_metadata=build_source_metadata(overlay, overlay_sha256, dump_file),
            eval_library=EvalLibrary(name='unknown', version='unknown'),
            model_info=ModelInfo(
                name=anchors.model_name,
                id=model_id,
                developer=anchors.developer,
                additional_details={
                    'raw_model_name': anchors.model_name,
                    'identity_source': 'protocol_qualification_manifest',
                },
            ),
            evaluation_results=sorted(results, key=lambda r: r.evaluation_result_id or ''),
        )
        bundles.append(
            LogBundle(log=log, developer=anchors.developer, model=model_id.split('/', 1)[1])
        )
    if selected_cells == 0 or not bundles:
        raise ValueError('DrugBank qualification manifest selected zero source score cells')
    return SourceConversionResult(
        source_name=SOURCE_NAME,
        total_records=selected_cells,
        records=bundles,
        failures=[],
        exclusions=[],
    )


def require_empty_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f'output directory must be empty for a fail-closed conversion: {output_dir}')


def export(bundles: Iterable[LogBundle], output_dir: Path) -> list[Path]:
    return save_evaluation_logs(
        EvaluationLogOutput(
            eval_log=bundle.log,
            base_dir=output_dir,
            developer=bundle.developer,
            model_name=bundle.model,
        )
        for bundle in bundles
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert explicitly qualified PwC DrugBank transductive/inductive results.'
    )
    parser.add_argument('--dump', type=Path, required=True, help='Local PwC PostgreSQL custom-format dump.')
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
    result = build_logs(
        evaluations, datasets, overlay, overlay_sha256, dump_file=args.dump.name
    )
    require_empty_output_dir(args.output_dir)
    paths = export(result.records, args.output_dir)
    for path in paths:
        print(path)
    return len(paths)


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} Papers with Code DrugBank model log(s).')
