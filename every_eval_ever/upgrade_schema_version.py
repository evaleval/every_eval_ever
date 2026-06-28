"""Normalize datastore schema_version fields to the current 0.2.2 schemas."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from every_eval_ever.validate import expand_paths

AGGREGATE_SCHEMA_VERSION = '0.2.2'
INSTANCE_SCHEMA_VERSION = 'instance_level_eval_0.2.2'


@dataclass
class SchemaVersionUpgradeReport:
    file_path: Path
    file_type: str
    target_version: str
    changed: bool
    rows_changed: int = 0
    rows_scanned: int = 0
    old_versions: tuple[str | None, ...] = ()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def upgrade_aggregate_file(
    file_path: Path, *, write_changes: bool = False
) -> SchemaVersionUpgradeReport:
    payload = json.loads(file_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'Aggregate payload is not a JSON object: {file_path}')

    old_version = payload.get('schema_version')
    changed = old_version != AGGREGATE_SCHEMA_VERSION
    if changed:
        payload['schema_version'] = AGGREGATE_SCHEMA_VERSION
        if write_changes:
            _write_json(file_path, payload)

    return SchemaVersionUpgradeReport(
        file_path=file_path,
        file_type='aggregate',
        target_version=AGGREGATE_SCHEMA_VERSION,
        changed=changed,
        rows_scanned=1,
        rows_changed=1 if changed else 0,
        old_versions=(old_version,) if changed else (),
    )


def upgrade_instance_file(
    file_path: Path, *, write_changes: bool = False
) -> SchemaVersionUpgradeReport:
    rows: list[dict] = []
    old_versions: set[str | None] = set()
    changed_rows = 0

    with file_path.open(encoding='utf-8') as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(
                    f'Instance payload line is not a JSON object: {file_path}'
                )
            old_version = row.get('schema_version')
            if old_version != INSTANCE_SCHEMA_VERSION:
                old_versions.add(old_version)
                row['schema_version'] = INSTANCE_SCHEMA_VERSION
                changed_rows += 1
            rows.append(row)

    if write_changes and changed_rows:
        _write_jsonl(file_path, rows)

    return SchemaVersionUpgradeReport(
        file_path=file_path,
        file_type='instance',
        target_version=INSTANCE_SCHEMA_VERSION,
        changed=changed_rows > 0,
        rows_scanned=len(rows),
        rows_changed=changed_rows,
        old_versions=tuple(sorted(old_versions, key=lambda value: str(value))),
    )


def upgrade_file(
    file_path: Path, *, write_changes: bool = False
) -> SchemaVersionUpgradeReport:
    if file_path.suffix == '.json':
        return upgrade_aggregate_file(file_path, write_changes=write_changes)
    if file_path.suffix == '.jsonl':
        return upgrade_instance_file(file_path, write_changes=write_changes)
    raise ValueError(
        f"Unsupported file extension '{file_path.suffix}' for {file_path}"
    )


def render_text(reports: list[SchemaVersionUpgradeReport]) -> str:
    aggregate_changed = sum(
        1
        for report in reports
        if report.file_type == 'aggregate' and report.changed
    )
    instance_changed = sum(
        1
        for report in reports
        if report.file_type == 'instance' and report.changed
    )
    instance_rows_changed = sum(
        report.rows_changed
        for report in reports
        if report.file_type == 'instance'
    )

    lines = [
        f'Scanned {len(reports)} file(s).',
        f'- Aggregate files updated: {aggregate_changed}',
        f'- Instance files updated: {instance_changed}',
        f'- Instance rows updated: {instance_rows_changed}',
        '',
    ]

    for report in reports:
        if not report.changed:
            continue
        suffix = (
            f' ({report.rows_changed}/{report.rows_scanned} rows)'
            if report.file_type == 'instance'
            else ''
        )
        previous = ', '.join(str(v) for v in report.old_versions)
        lines.append(
            f'Updated {report.file_path}: {previous} -> {report.target_version}{suffix}'
        )

    if all(not report.changed for report in reports):
        lines.append('No schema_version changes were needed.')

    return '\n'.join(lines).rstrip() + '\n'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='every_eval_ever upgrade-schema-version',
        description=(
            'Normalize aggregate and instance datastore files to the current '
            '0.2.2 schema_version values.'
        ),
    )
    parser.add_argument(
        'paths',
        nargs='+',
        help='Files or directories containing aggregate JSON and sample JSONL files.',
    )
    parser.add_argument(
        '--write',
        action='store_true',
        help='Write changes in place. Without this flag, run as a dry run.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    file_paths = expand_paths(args.paths)
    reports = [
        upgrade_file(file_path, write_changes=args.write)
        for file_path in file_paths
    ]
    print(render_text(reports), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
