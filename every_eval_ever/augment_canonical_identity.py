"""Backfill canonical metric/eval identity into existing datastore files."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from every_eval_ever.validate import expand_paths

WORDLE_METRICS: dict[str, dict[str, Any]] = {
    'win_rate': {
        'metric_id': 'win_rate',
        'metric_name': 'Win Rate',
        'metric_kind': 'win_rate',
        'metric_unit': 'proportion',
    },
    'avg_attempts': {
        'metric_id': 'mean_attempts',
        'metric_name': 'Average Attempts',
        'metric_kind': 'count',
        'metric_unit': 'attempts',
    },
    'avg_latency_ms': {
        'metric_id': 'latency_mean',
        'metric_name': 'Average Latency',
        'metric_kind': 'latency',
        'metric_unit': 'ms',
    },
}

HFOPENLLM_METRICS: dict[str, dict[str, Any]] = {
    'IFEval': {
        'metric_id': 'accuracy',
        'metric_name': 'Accuracy',
        'metric_kind': 'accuracy',
        'metric_unit': 'proportion',
    },
    'BBH': {
        'metric_id': 'accuracy',
        'metric_name': 'Accuracy',
        'metric_kind': 'accuracy',
        'metric_unit': 'proportion',
    },
    'MATH Level 5': {
        'metric_id': 'exact_match',
        'metric_name': 'Exact Match',
        'metric_kind': 'exact_match',
        'metric_unit': 'proportion',
    },
    'GPQA': {
        'metric_id': 'accuracy',
        'metric_name': 'Accuracy',
        'metric_kind': 'accuracy',
        'metric_unit': 'proportion',
    },
    'MUSR': {
        'metric_id': 'accuracy',
        'metric_name': 'Accuracy',
        'metric_kind': 'accuracy',
        'metric_unit': 'proportion',
    },
    'MMLU-PRO': {
        'metric_id': 'accuracy',
        'metric_name': 'Accuracy',
        'metric_kind': 'accuracy',
        'metric_unit': 'proportion',
    },
}


@dataclass
class CanonicalPatch:
    evaluation_name: str | None = None
    metric_id: str | None = None
    metric_name: str | None = None
    metric_kind: str | None = None
    metric_unit: str | None = None
    metric_parameters: dict[str, str | float | bool | None] | None = None


@dataclass
class AugmentationReport:
    file_path: Path
    benchmark_family: str
    aggregate_changed: bool
    sample_file_path: Path | None
    sample_changed: bool
    changed_results: int
    sample_rows: int

    @property
    def changed(self) -> bool:
        return self.aggregate_changed or self.sample_changed


def _slug(value: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_')
    return slug or 'unknown'


def _store_raw_evaluation_name(
    metric_config: dict[str, Any], raw_evaluation_name: str
) -> None:
    additional = metric_config.get('additional_details')
    if not isinstance(additional, dict):
        additional = {}
        metric_config['additional_details'] = additional
    additional.setdefault('raw_evaluation_name', raw_evaluation_name)


def _set_metric_field(
    metric_config: dict[str, Any], key: str, value: Any
) -> bool:
    if value is None:
        return False
    if metric_config.get(key) in (None, ''):
        metric_config[key] = value
        return True
    return False


def _metric_from_phrase(phrase: str) -> CanonicalPatch | None:
    normalized = phrase.strip().lower()

    if normalized in {'accuracy', 'cot correct', 'correct'}:
        return CanonicalPatch(
            metric_id='accuracy',
            metric_name='Accuracy',
            metric_kind='accuracy',
            metric_unit='proportion',
        )
    if 'strict acc' in normalized:
        return CanonicalPatch(
            metric_id='strict_accuracy',
            metric_name='Strict Accuracy',
            metric_kind='accuracy',
            metric_unit='proportion',
        )
    if normalized == 'em':
        return CanonicalPatch(
            metric_id='exact_match',
            metric_name='Exact Match',
            metric_kind='exact_match',
            metric_unit='proportion',
        )
    if normalized == 'f1':
        return CanonicalPatch(
            metric_id='f1',
            metric_name='F1',
            metric_kind='f1',
            metric_unit='proportion',
        )
    if normalized in {'mean win rate', 'win rate'}:
        return CanonicalPatch(
            metric_id='win_rate',
            metric_name='Win Rate',
            metric_kind='win_rate',
            metric_unit='proportion',
        )
    if normalized in {'mean score', 'score', 'wb score'}:
        return CanonicalPatch(
            metric_id='score',
            metric_name='Score',
            metric_kind='score',
            metric_unit='proportion',
        )
    if normalized == 'harmlessness':
        return CanonicalPatch(
            metric_id='harmlessness',
            metric_name='Harmlessness',
            metric_kind='score',
            metric_unit='points',
        )
    return None


def _wordle_patch(raw_evaluation_name: str) -> CanonicalPatch | None:
    prefix = 'wordle_arena_'
    if not raw_evaluation_name.startswith(prefix):
        return None

    metric_suffix = raw_evaluation_name[len(prefix) :]
    metric = WORDLE_METRICS.get(metric_suffix)
    if metric is None:
        return None

    return CanonicalPatch(
        evaluation_name='wordle_arena',
        metric_id=metric['metric_id'],
        metric_name=metric['metric_name'],
        metric_kind=metric['metric_kind'],
        metric_unit=metric['metric_unit'],
    )


def _fibble_patch(raw_evaluation_name: str) -> CanonicalPatch | None:
    match = re.fullmatch(
        r'fibble_arena_(?P<variant>1lie|5lies)_(?P<metric>.+)',
        raw_evaluation_name,
    )
    if match is None:
        return None

    metric = WORDLE_METRICS.get(match.group('metric'))
    if metric is None:
        return None

    return CanonicalPatch(
        evaluation_name=f'fibble_arena_{match.group("variant")}',
        metric_id=metric['metric_id'],
        metric_name=metric['metric_name'],
        metric_kind=metric['metric_kind'],
        metric_unit=metric['metric_unit'],
    )


def _apex_agents_patch(raw_evaluation_name: str) -> CanonicalPatch | None:
    pass_match = re.fullmatch(
        r'(?P<slice>.+) Pass@(?P<k>\d+)', raw_evaluation_name
    )
    if pass_match is not None:
        k = int(pass_match.group('k'))
        return CanonicalPatch(
            evaluation_name=pass_match.group('slice'),
            metric_id='pass_at_k',
            metric_name=f'Pass@{k}',
            metric_kind='pass_rate',
            metric_unit='proportion',
            metric_parameters={'k': k},
        )

    score_match = re.fullmatch(r'(?P<slice>.+) Mean Score', raw_evaluation_name)
    if score_match is not None:
        return CanonicalPatch(
            evaluation_name=score_match.group('slice'),
            metric_id='mean_score',
            metric_name='Mean Score',
            metric_kind='score',
            metric_unit='proportion',
        )

    return None


def _bfcl_patch(
    raw_evaluation_name: str, metric_config: dict[str, Any]
) -> CanonicalPatch | None:
    metric_id = metric_config.get('metric_id')
    if not isinstance(metric_id, str) or not metric_id.startswith('bfcl.'):
        return None

    parts = metric_id.split('.')
    if len(parts) < 3:
        return None

    eval_slice = '.'.join(parts[1:-1])
    if not eval_slice:
        return None

    if raw_evaluation_name != metric_id and raw_evaluation_name == eval_slice:
        return None

    return CanonicalPatch(evaluation_name=eval_slice)


def _rewardbench_patch(
    raw_evaluation_name: str, metric_config: dict[str, Any]
) -> CanonicalPatch | None:
    description = str(metric_config.get('evaluation_description') or '')
    lowered = description.lower()

    if raw_evaluation_name == 'Score':
        return CanonicalPatch(
            evaluation_name='RewardBench',
            metric_id='rewardbench.score',
            metric_name='Score',
            metric_kind='score',
            metric_unit='proportion',
        )

    if 'accuracy' in lowered:
        return CanonicalPatch(
            metric_id='accuracy',
            metric_name='Accuracy',
            metric_kind='accuracy',
            metric_unit='proportion',
        )

    if 'score' in lowered:
        return CanonicalPatch(
            metric_id='score',
            metric_name='Score',
            metric_kind='score',
            metric_unit='proportion',
        )

    return None


def _helm_patch(
    benchmark_family: str,
    raw_evaluation_name: str,
    metric_config: dict[str, Any],
) -> CanonicalPatch | None:
    description = str(metric_config.get('evaluation_description') or '')

    if raw_evaluation_name.lower().startswith('mean '):
        metric = _metric_from_phrase(raw_evaluation_name)
        if metric is None:
            return None
        metric.evaluation_name = benchmark_family
        return metric

    if ' on ' in description:
        metric_phrase = description.split(' on ', 1)[0]
        return _metric_from_phrase(metric_phrase)

    return None


def _canonical_patch_for_result(
    benchmark_family: str,
    result: dict[str, Any],
) -> CanonicalPatch | None:
    raw_evaluation_name = str(result.get('evaluation_name') or '')
    metric_config = result.setdefault('metric_config', {})

    if benchmark_family == 'global-mmlu-lite':
        return CanonicalPatch(
            metric_id='accuracy',
            metric_name='Accuracy',
            metric_kind='accuracy',
            metric_unit='proportion',
        )

    if benchmark_family == 'hfopenllm_v2':
        metric = HFOPENLLM_METRICS.get(raw_evaluation_name)
        if metric is None:
            return None
        return CanonicalPatch(
            metric_id=metric['metric_id'],
            metric_name=metric['metric_name'],
            metric_kind=metric['metric_kind'],
            metric_unit=metric['metric_unit'],
        )

    if benchmark_family == 'reward-bench':
        return _rewardbench_patch(raw_evaluation_name, metric_config)

    if benchmark_family == 'terminal-bench-2.0':
        return CanonicalPatch(
            metric_id='accuracy',
            metric_name='Task Resolution Accuracy',
            metric_kind='accuracy',
            metric_unit='percentage',
        )

    if benchmark_family == 'wordle_arena':
        return _wordle_patch(raw_evaluation_name)

    if benchmark_family == 'fibble_arena':
        return _fibble_patch(raw_evaluation_name)

    if benchmark_family == 'apex-agents':
        return _apex_agents_patch(raw_evaluation_name)

    if benchmark_family == 'bfcl':
        return _bfcl_patch(raw_evaluation_name, metric_config)

    if benchmark_family.startswith('helm_'):
        return _helm_patch(
            benchmark_family=benchmark_family,
            raw_evaluation_name=raw_evaluation_name,
            metric_config=metric_config,
        )

    return None


def _apply_patch(
    result: dict[str, Any], patch: CanonicalPatch | None
) -> tuple[bool, str]:
    raw_evaluation_name = str(result.get('evaluation_name') or '')
    if patch is None:
        return False, raw_evaluation_name

    metric_config = result.setdefault('metric_config', {})
    changed = False

    if (
        patch.evaluation_name is not None
        and result.get('evaluation_name') != patch.evaluation_name
    ):
        _store_raw_evaluation_name(metric_config, raw_evaluation_name)
        result['evaluation_name'] = patch.evaluation_name
        changed = True

    changed |= _set_metric_field(metric_config, 'metric_id', patch.metric_id)
    changed |= _set_metric_field(
        metric_config, 'metric_name', patch.metric_name
    )
    changed |= _set_metric_field(
        metric_config, 'metric_kind', patch.metric_kind
    )
    changed |= _set_metric_field(
        metric_config, 'metric_unit', patch.metric_unit
    )

    if patch.metric_parameters and metric_config.get('metric_parameters') in (
        None,
        {},
    ):
        metric_config['metric_parameters'] = patch.metric_parameters
        changed = True

    return changed, raw_evaluation_name


def _metric_fragment(metric_config: dict[str, Any]) -> str:
    metric_id = str(
        metric_config.get('metric_id')
        or metric_config.get('metric_name')
        or 'score'
    )
    fragment = _slug(metric_id)
    parameters = metric_config.get('metric_parameters')
    if isinstance(parameters, dict):
        for key, value in sorted(parameters.items()):
            fragment += f'__{_slug(str(key))}_{_slug(str(value))}'
    return fragment


def _raw_evaluation_name(result: dict[str, Any]) -> str:
    metric_config = result.get('metric_config', {})
    if not isinstance(metric_config, dict):
        return str(result.get('evaluation_name') or '')

    additional = metric_config.get('additional_details')
    if isinstance(additional, dict):
        raw_name = additional.get('raw_evaluation_name')
        if raw_name:
            return str(raw_name)

    return str(result.get('evaluation_name') or '')


def _assign_evaluation_result_ids(
    payload: dict[str, Any],
    sample_updates: dict[str, dict[str, str]],
) -> bool:
    evaluation_id = str(payload.get('evaluation_id') or '')
    counts: dict[str, int] = {}
    changed = False

    for result in payload.get('evaluation_results', []):
        raw_evaluation_name = _raw_evaluation_name(result)
        eval_name = str(result.get('evaluation_name') or '')
        base = (
            f'{evaluation_id}'
            f'#{_slug(eval_name)}'
            f'#{_metric_fragment(result.get("metric_config", {}))}'
        )
        counts[base] = counts.get(base, 0) + 1
        evaluation_result_id = base
        if counts[base] > 1:
            evaluation_result_id = f'{base}__{counts[base]}'

        if result.get('evaluation_result_id') != evaluation_result_id:
            result['evaluation_result_id'] = evaluation_result_id
            changed = True

        sample_updates[raw_evaluation_name] = {
            'evaluation_name': eval_name,
            'evaluation_result_id': evaluation_result_id,
        }

    return changed


def infer_benchmark_family(
    file_path: Path, payload: dict[str, Any] | None = None
) -> str:
    parts = list(file_path.parts)
    if 'data' in parts:
        idx = parts.index('data')
        if idx + 1 < len(parts):
            return parts[idx + 1]

    if payload is None:
        return 'unknown'

    evaluation_id = str(payload.get('evaluation_id') or '')
    if '/' in evaluation_id:
        return evaluation_id.split('/', 1)[0]
    return 'unknown'


def augment_aggregate_payload(
    payload: dict[str, Any], benchmark_family: str | None = None
) -> tuple[dict[str, Any], int, dict[str, dict[str, str]], bool]:
    benchmark_family = benchmark_family or infer_benchmark_family(
        Path('.'), payload
    )
    changed_results = 0
    sample_updates: dict[str, dict[str, str]] = {}

    for result in payload.get('evaluation_results', []):
        patch = _canonical_patch_for_result(benchmark_family, result)
        changed, _ = _apply_patch(result, patch)
        if changed:
            changed_results += 1

    ids_changed = _assign_evaluation_result_ids(payload, sample_updates)
    return payload, changed_results, sample_updates, ids_changed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def augment_sample_rows(
    rows: list[dict[str, Any]],
    sample_updates: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], bool]:
    changed = False

    for row in rows:
        raw_evaluation_name = str(row.get('evaluation_name') or '')
        update = sample_updates.get(raw_evaluation_name)
        if update is None:
            continue

        if row.get('evaluation_name') != update['evaluation_name']:
            row['evaluation_name'] = update['evaluation_name']
            changed = True

        if row.get('evaluation_result_id') != update['evaluation_result_id']:
            row['evaluation_result_id'] = update['evaluation_result_id']
            changed = True

    return rows, changed


def augment_aggregate_file(
    file_path: Path,
    *,
    write_changes: bool = False,
    update_samples: bool = True,
) -> AugmentationReport:
    payload = json.loads(file_path.read_text(encoding='utf-8'))
    benchmark_family = infer_benchmark_family(file_path, payload)
    payload, changed_results, sample_updates, ids_changed = (
        augment_aggregate_payload(payload, benchmark_family=benchmark_family)
    )
    aggregate_changed = changed_results > 0 or ids_changed

    sample_file_path = file_path.with_name(f'{file_path.stem}_samples.jsonl')
    sample_changed = False
    sample_rows = 0

    if update_samples and sample_file_path.exists():
        rows = _read_jsonl(sample_file_path)
        sample_rows = len(rows)
        rows, sample_changed = augment_sample_rows(rows, sample_updates)

        detailed = payload.get('detailed_evaluation_results')
        if (
            isinstance(detailed, dict)
            and detailed.get('total_rows') != sample_rows
        ):
            detailed['total_rows'] = sample_rows
            aggregate_changed = True

        if write_changes and sample_changed:
            _write_jsonl(sample_file_path, rows)

    if write_changes and aggregate_changed:
        _write_json(file_path, payload)

    return AugmentationReport(
        file_path=file_path,
        benchmark_family=benchmark_family,
        aggregate_changed=aggregate_changed,
        sample_file_path=sample_file_path
        if sample_file_path.exists()
        else None,
        sample_changed=sample_changed,
        changed_results=changed_results,
        sample_rows=sample_rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='every_eval_ever augment-canonical-identity',
        description=(
            'Backfill metric identity and canonical eval names into existing '
            'EEE aggregate JSON files.'
        ),
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Aggregate JSON files or directories containing datastore files.',
    )
    parser.add_argument(
        '--write',
        action='store_true',
        help='Write changes in place. Without this flag, run as a dry run.',
    )
    parser.add_argument(
        '--skip-samples',
        action='store_true',
        help='Do not update companion *_samples.jsonl files.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    aggregate_files = [
        path
        for path in expand_paths(args.paths)
        if path.suffix == '.json' and not path.name.endswith('_samples.json')
    ]

    reports = [
        augment_aggregate_file(
            path,
            write_changes=args.write,
            update_samples=not args.skip_samples,
        )
        for path in aggregate_files
    ]

    changed_reports = [report for report in reports if report.changed]

    action = 'Updated' if args.write else 'Would update'
    for report in changed_reports:
        print(
            f'{action}: {report.file_path} '
            f'[{report.benchmark_family}] '
            f'(results={report.changed_results}, '
            f'sample_rows={report.sample_rows}, '
            f'samples_changed={report.sample_changed})'
        )

    print(
        f'Scanned {len(reports)} aggregate file(s); '
        f'{len(changed_reports)} would change.'
        if not args.write
        else f'Scanned {len(reports)} aggregate file(s); '
        f'updated {len(changed_reports)}.'
    )
    return 0
