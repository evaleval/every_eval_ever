"""Audit canonical metric/result identity coverage in aggregate datastore JSON."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from every_eval_ever.validate import expand_paths

IDENTITY_FIELDS = (
    'evaluation_result_id',
    'metric_id',
    'metric_name',
    'metric_kind',
    'metric_unit',
)

EVALUATION_RESULT_ID_PATTERN = re.compile(r'.+#.+#.+')


@dataclass
class CanonicalIdentityReport:
    files_scanned: int = 0
    results_scanned: int = 0
    missing: Counter[str] = field(default_factory=Counter)
    malformed: Counter[str] = field(default_factory=Counter)
    benchmark_missing: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )

    @property
    def has_issues(self) -> bool:
        return bool(self.missing) or bool(self.malformed)

    def to_dict(self, top: int = 20) -> dict[str, Any]:
        top_missing_by_benchmark: dict[str, list[tuple[str, int]]] = {}
        for field_name in IDENTITY_FIELDS:
            pairs = [
                (benchmark, counts[field_name])
                for benchmark, counts in self.benchmark_missing.items()
                if counts[field_name] > 0
            ]
            pairs.sort(key=lambda item: item[1], reverse=True)
            top_missing_by_benchmark[field_name] = pairs[:top]

        return {
            'files_scanned': self.files_scanned,
            'results_scanned': self.results_scanned,
            'missing': dict(self.missing),
            'malformed': dict(self.malformed),
            'top_missing_by_benchmark': top_missing_by_benchmark,
        }


def infer_benchmark_family(file_path: Path, payload: dict[str, Any]) -> str:
    parts = list(file_path.parts)
    if 'data' in parts:
        idx = parts.index('data')
        if idx + 1 < len(parts):
            return parts[idx + 1]

    evaluation_id = str(payload.get('evaluation_id') or '')
    if '/' in evaluation_id:
        return evaluation_id.split('/', 1)[0]
    return 'unknown'


def _record_missing(
    report: CanonicalIdentityReport, benchmark_family: str, field_name: str
) -> None:
    report.missing[field_name] += 1
    report.benchmark_missing[benchmark_family][field_name] += 1


def _check_result_fields(
    report: CanonicalIdentityReport,
    benchmark_family: str,
    result: dict[str, Any],
) -> None:
    evaluation_result_id = result.get('evaluation_result_id')
    if not isinstance(evaluation_result_id, str) or not evaluation_result_id.strip():
        _record_missing(report, benchmark_family, 'evaluation_result_id')
    elif not EVALUATION_RESULT_ID_PATTERN.fullmatch(evaluation_result_id):
        report.malformed['evaluation_result_id_pattern'] += 1

    metric_config = result.get('metric_config')
    if not isinstance(metric_config, dict):
        report.malformed['metric_config_not_object'] += 1
        for field_name in IDENTITY_FIELDS[1:]:
            _record_missing(report, benchmark_family, field_name)
        return

    for field_name in IDENTITY_FIELDS[1:]:
        value = metric_config.get(field_name)
        if value is None:
            _record_missing(report, benchmark_family, field_name)
            continue
        if not isinstance(value, str):
            report.malformed[f'{field_name}_type'] += 1
            continue
        if not value.strip():
            _record_missing(report, benchmark_family, field_name)


def check_file(file_path: Path, report: CanonicalIdentityReport) -> None:
    try:
        payload = json.loads(file_path.read_text(encoding='utf-8'))
    except Exception:
        report.malformed['aggregate_json_parse_error'] += 1
        return

    benchmark_family = infer_benchmark_family(file_path, payload)

    evaluation_results = payload.get('evaluation_results')
    if not isinstance(evaluation_results, list):
        report.malformed['evaluation_results_not_list'] += 1
        return

    report.results_scanned += len(evaluation_results)
    for result in evaluation_results:
        if not isinstance(result, dict):
            report.malformed['evaluation_result_not_object'] += 1
            continue
        _check_result_fields(report, benchmark_family, result)


def check_paths(paths: list[str]) -> CanonicalIdentityReport:
    report = CanonicalIdentityReport()
    aggregate_files = [
        path
        for path in expand_paths(paths)
        if path.suffix == '.json' and not path.name.endswith('_samples.json')
    ]
    report.files_scanned = len(aggregate_files)

    for file_path in aggregate_files:
        check_file(file_path, report)

    return report


def render_text(report: CanonicalIdentityReport, top: int = 20) -> str:
    payload = report.to_dict(top=top)
    lines = [
        f'Scanned {payload["files_scanned"]} aggregate file(s) '
        f'with {payload["results_scanned"]} result row(s).',
        '',
    ]

    if payload['missing']:
        lines.append('Missing canonical fields:')
        for field_name in IDENTITY_FIELDS:
            missing_count = payload['missing'].get(field_name, 0)
            if missing_count:
                lines.append(f'- {field_name}: {missing_count}')
        lines.append('')

    if payload['malformed']:
        lines.append('Malformed fields:')
        for key, value in sorted(payload['malformed'].items()):
            lines.append(f'- {key}: {value}')
        lines.append('')

    for field_name in IDENTITY_FIELDS:
        pairs = payload['top_missing_by_benchmark'].get(field_name, [])
        if not pairs:
            continue
        lines.append(f'Top missing benchmarks for {field_name}:')
        for benchmark, count in pairs:
            lines.append(f'- {benchmark}: {count}')
        lines.append('')

    if not report.has_issues:
        lines.append('No canonical identity issues found.')

    return '\n'.join(lines).rstrip() + '\n'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='every_eval_ever check-canonical-identity',
        description=(
            'Audit aggregate JSON files for missing or malformed canonical '
            'metric/result identity fields.'
        ),
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Aggregate JSON files or directories containing datastore files.',
    )
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        dest='output_format',
        help='Output format.',
    )
    parser.add_argument(
        '--top',
        type=int,
        default=20,
        help='How many benchmarks to show per missing field in text/json output.',
    )
    parser.add_argument(
        '--fail-on-issues',
        action='store_true',
        help='Exit with status 1 if any missing/malformed identity fields exist.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = check_paths(args.paths)
    payload = report.to_dict(top=args.top)

    if args.output_format == 'json':
        print(json.dumps(payload, indent=2))
    else:
        print(render_text(report, top=args.top), end='')

    if args.fail_on_issues and report.has_issues:
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
