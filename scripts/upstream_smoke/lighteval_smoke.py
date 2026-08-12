"""Run lighteval's own fake model, then convert and validate what it wrote.

Answers a question the offline tests cannot: did a new lighteval release change
the output the converter reads? The fixture under tests/data/lighteval_smoke pins
a *past* release; this script asks the installed one to produce a run right now,
converts it, and requires the result to pass the datastore gate.

No weights and no inference: DummyModelConfig returns random logprobs and fixed
text, so the whole thing is a dataset download and some file writing.

    UV_TORCH_BACKEND=cpu uv run -p 3.12 --extra lighteval --with lighteval \\
        python scripts/upstream_smoke/lighteval_smoke.py

Add --refresh to overwrite the committed fixture with what this run produced.
Read the diff before committing it: --refresh will just as happily record a real
upstream regression as an intended change.

lighteval is deliberately not a dependency of this project. It needs datasets>=4
while crfm-helm pins datasets~=3.1, so the two cannot share a lockfile; this
script is therefore run with `uv run --with lighteval`, never from the project
environment.

Scope: shape, not semantics. A metric that silently switches from percent to
proportion passes this. So does a task whose prompt template changed. And a fake
model only exercises the metrics its task defines, which is why the default task
list pairs a multiple-choice task with a generative one.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Deliberately not under tests/data/lighteval/. That tree is a hand-collected
# SmolLM2 evaluation, and a test there walks it recursively and counts what it
# finds, so a second run dropped inside would change what their fixture means --
# besides putting a generated tree where --refresh could overwrite a collected
# one.
FIXTURE_DIR = REPO_ROOT / 'tests' / 'data' / 'lighteval_smoke'

# One multiple-choice task and one generative task: a fake model only produces
# the metrics its task defines, so a single task would leave either the
# loglikelihood path or the generation path unexercised.
DEFAULT_TASKS = 'anli:r1|0,squad_v2|0'

# Slashed on purpose. The datastore path is
# data/<collection>/<developer>/<model>/, and the developer comes from the part
# before the slash, so a bare model name cannot be published at all.
DEFAULT_MODEL_NAME = 'eee-smoke/dummy-model'

PROVENANCE_NAME = 'PROVENANCE.md'


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ or '')
    parser.add_argument(
        '--tasks',
        default=DEFAULT_TASKS,
        help='lighteval task specification, comma-separated '
        f'(default: {DEFAULT_TASKS}). Exposed so a task that upstream breaks '
        'or removes can be swapped without editing this file.',
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=2,
        help='Samples per task (default: 2)',
    )
    parser.add_argument(
        '--model-name',
        default=DEFAULT_MODEL_NAME,
        help=f'Name to record for the fake model (default: {DEFAULT_MODEL_NAME})',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Where lighteval writes its run. Defaults to a temporary '
        'directory that is removed on exit.',
    )
    parser.add_argument(
        '--refresh',
        action='store_true',
        help=f'Replace {FIXTURE_DIR.relative_to(REPO_ROOT)} with this run',
    )
    return parser.parse_args(argv)


def lighteval_version() -> str:
    try:
        return version('lighteval')
    except PackageNotFoundError:
        return 'unknown'


def run_lighteval(
    output_dir: Path, tasks: str, max_samples: int, model_name: str
) -> None:
    """Produce a real lighteval run with lighteval's own fake model.

    Only the public entry points their docs use, so an internal refactor
    upstream does not break this and a user-facing change does.
    """
    from lighteval.logging.evaluation_tracker import EvaluationTracker
    from lighteval.models.dummy.dummy_model import DummyModelConfig
    from lighteval.pipeline import (
        ParallelismManager,
        Pipeline,
        PipelineParameters,
    )

    tracker = EvaluationTracker(
        output_dir=str(output_dir),
        save_details=True,
    )
    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=PipelineParameters(
            launcher_type=ParallelismManager.NONE,
            max_samples=max_samples,
            dataset_loading_processes=1,
        ),
        evaluation_tracker=tracker,
        model_config=DummyModelConfig(model_name=model_name, seed=42),
    )
    pipeline.evaluate()
    pipeline.save_and_push_results()


def find_results_files(output_dir: Path) -> list[Path]:
    return sorted((output_dir / 'results').rglob('results_*.json'))


def convert_and_validate(results_file: Path, data_dir: Path) -> int:
    """Convert one run through the real CLI, then the real datastore gate."""
    from every_eval_ever import cli

    exit_code = cli.main(
        [
            'convert',
            'lighteval',
            '--log_path',
            str(results_file),
            '--include_details',
            '--output_dir',
            str(data_dir),
            '--eval_library_version',
            lighteval_version(),
        ]
    )
    if exit_code != 0:
        print(f'FAIL: conversion exited {exit_code}', file=sys.stderr)
        return exit_code

    published = sorted(data_dir.glob('*/*/*/*.json'))
    samples = sorted(data_dir.glob('*/*/*/*_samples.jsonl'))
    print(
        f'Converted {len(published)} record(s) and {len(samples)} '
        f'instance-level file(s) from {results_file.name}'
    )
    if not published:
        print(
            f'FAIL: conversion wrote no records under {data_dir}',
            file=sys.stderr,
        )
        return 1
    if not samples:
        print(
            'FAIL: --include_details produced no instance-level output; '
            'lighteval may have changed its details layout',
            file=sys.stderr,
        )
        return 1

    # The validator's own entry point rather than validate_file: the semantic
    # checks only run when a record sits at data/<collection>/<developer>/
    # <model>/, and this is what resolves that context. 0 clean, 1 errors,
    # 2 warnings only -- and a warning is not merge-ready, so only 0 passes.
    return cli.main(
        [
            'validate',
            str(data_dir / '*' / '*' / '*' / '*.json'),
            '--format',
            'rich',
        ]
    ) or cli.main(
        [
            'validate',
            str(data_dir / '*' / '*' / '*' / '*.jsonl'),
            '--format',
            'rich',
        ]
    )


def write_provenance(fixture_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        '# Provenance',
        '',
        'Produced by `scripts/upstream_smoke/lighteval_smoke.py --refresh`,',
        'not hand-copied from a real evaluation.',
        '',
        f'- lighteval version: `{lighteval_version()}`',
        f'- tasks: `{args.tasks}`',
        f'- samples per task: `{args.max_samples}`',
        f'- model: `{args.model_name}`',
        '',
        "The model is lighteval's own `DummyModelConfig` (seed 42): random",
        'logprobs and fixed text, no weights and no inference. The scores here',
        'are therefore meaningless as measurements — this tree exists to pin',
        'the *shape* of lighteval output that the converter reads.',
        '',
        'To update after an upstream change, re-run the command above and read',
        'the diff before committing it.',
        '',
    ]
    (fixture_dir / PROVENANCE_NAME).write_text(
        '\n'.join(lines), encoding='utf-8'
    )


def refresh_fixture(output_dir: Path, args: argparse.Namespace) -> None:
    """Replace the generated fixture subtree with this run's output."""
    if FIXTURE_DIR.exists():
        shutil.rmtree(FIXTURE_DIR)
    FIXTURE_DIR.mkdir(parents=True)
    for name in ('results', 'details'):
        source = output_dir / name
        if source.is_dir():
            shutil.copytree(source, FIXTURE_DIR / name)
    write_provenance(FIXTURE_DIR, args)
    print(f'Refreshed {FIXTURE_DIR.relative_to(REPO_ROOT)}')


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(f'lighteval {lighteval_version()}, tasks {args.tasks}')

    with tempfile.TemporaryDirectory(prefix='eee-lighteval-smoke-') as scratch:
        scratch_dir = Path(scratch)
        output_dir = (
            Path(args.output_dir) if args.output_dir else scratch_dir / 'run'
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        run_lighteval(output_dir, args.tasks, args.max_samples, args.model_name)

        results_files = find_results_files(output_dir)
        if not results_files:
            print(
                f'FAIL: lighteval wrote no results file under {output_dir}. '
                'Its output layout may have changed.',
                file=sys.stderr,
            )
            return 1

        exit_code = 0
        for index, results_file in enumerate(results_files):
            data_dir = scratch_dir / f'converted-{index}' / 'data'
            exit_code = (
                convert_and_validate(results_file, data_dir) or exit_code
            )

        if exit_code == 0 and args.refresh:
            refresh_fixture(output_dir, args)

    if exit_code == 0:
        print('OK')
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
