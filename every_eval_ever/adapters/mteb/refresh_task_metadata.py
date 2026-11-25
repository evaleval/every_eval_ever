#!/usr/bin/env python3
"""Regenerate ``task_metadata.json`` from the installed ``mteb`` package.

The adapter needs each task's ``main_score``, its evaluation dataset and that
dataset's revision. Vendoring them keeps ``mteb`` (and its torch dependency) out
of the adapter's runtime and keeps a conversion reproducible, at the cost of a
snapshot that has to be refreshed deliberately.

Run outside the project environment so the heavy dependency stays there::

    uv run --with mteb --no-project python \\
      every_eval_ever/adapters/mteb/refresh_task_metadata.py

``tests/test_mteb_adapter.py`` fails if a refresh introduces a ``main_score``
the adapter has no ``MetricSpec`` for, so the snapshot cannot quietly outrun the
metric table.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

SUITES = ('MTEB(eng, v2)', 'MTEB(Multilingual, v2)')
OUTPUT = Path(__file__).resolve().parent / 'task_metadata.json'

NOTE = (
    'Vendored so the adapter needs neither the mteb package (a heavy optional '
    'dependency) nor a network call to name a metric. Each task records the '
    'main_score mteb declares for it; the adapter re-checks that name against '
    'the score dump it converts and drops a row where they disagree, so a '
    'results file written by an older mteb than this snapshot cannot be '
    'silently mislabelled. Regenerate with '
    'every_eval_ever/adapters/mteb/refresh_task_metadata.py.'
)


def main() -> None:
    import mteb  # imported here so the module stays importable without it

    benchmarks = {b.name: b for b in mteb.get_benchmarks()}
    missing = [name for name in SUITES if name not in benchmarks]
    if missing:
        raise SystemExit(
            f'mteb {mteb.__version__} has no benchmark(s): {missing}'
        )

    suites = {
        name: sorted(task.metadata.name for task in benchmarks[name].tasks)
        for name in SUITES
    }

    tasks: dict[str, dict] = {}
    for name in sorted({n for names in suites.values() for n in names}):
        metadata = mteb.get_task(name).metadata
        dataset = dict(metadata.dataset or {})
        tasks[name] = {
            'type': metadata.type,
            'main_score': metadata.main_score,
            'eval_splits': list(metadata.eval_splits or []),
            'hf_repo': dataset.get('path'),
            'dataset_revision': dataset.get('revision'),
            'license': metadata.license,
            'reference': metadata.reference,
            'domains': list(metadata.domains or []),
        }

    payload = {
        '_meta': {
            'source': 'mteb python package task + benchmark metadata',
            'mteb_version': mteb.__version__,
            'retrieved_date': datetime.date.today().isoformat(),
            'suites': sorted(SUITES),
            'note': NOTE,
        },
        'suites': suites,
        'tasks': tasks,
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(
        f'wrote {OUTPUT} with {len(tasks)} tasks from mteb {mteb.__version__}'
    )


if __name__ == '__main__':
    main()
