"""Shared machinery for testing a converter against a committed upstream log.

One `ConverterCase` per converter is all it takes to be covered by
`tests/test_converter_conversion.py`: point it at a real log the upstream tool wrote,
give the CLI arguments a user would give, and state what the conversion should yield.

Adding a converter here does not require writing a test.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ConverterCase:
    """One converter, one committed upstream log, and what converting it must yield."""

    source: str
    log_path: Path
    # What the conversion is expected to produce. Kept small on purpose: enough that a
    # silently dropped score, task, or sidecar fails, without restating the whole record.
    aggregates: int
    sidecars: int = 0
    # Total `evaluation_results` across every aggregate. For a converter that emits more
    # results than are worth listing in `scores`, this is what catches one going missing.
    results: int | None = None
    model_id: str | None = None
    # Keyed by `<evaluation_name>/<metric>`, since one task can be scored by several
    # metrics and each becomes its own result.
    scores: dict[str, float] | None = None
    extra_argv: tuple[str, ...] = ()
    # Upstream key paths the converter cannot work without, `*` matching any one key.
    required_source_paths: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return self.source

    def source_payload(self) -> Any:
        return json.loads(self.log_path.read_text(encoding='utf-8'))


CASES: tuple[ConverterCase, ...] = (
    ConverterCase(
        source='lm_eval',
        log_path=REPO_ROOT
        / 'tests/data/lm_eval/results_2026-01-21T03-44-18.458309.json',
        aggregates=2,
        # Two tasks, one `exact_match` result each. Stated even though `scores`
        # already lists both, because `scores` is a dict: without a count, a
        # third result that collided on an existing key would be merged away.
        results=2,
        # The fixture ships a samples file for only one of its two tasks, so
        # --include_samples would (correctly) report a partial conversion.
        model_id=(
            'RylanSchaeffer/mem_Qwen3-93M_minerva_math_rep_0_sbst_1.0000_epch_1_ot_1'
        ),
        scores={
            'math_perturbed_full/exact_match': 0.0,
            'math_rephrased_full/exact_match': 0.0004,
        },
        required_source_paths=(
            'config.model',
            'config.model_args',
            'results.*',
            'configs.*.dataset_path',
        ),
    ),
    ConverterCase(
        source='inspect',
        log_path=REPO_ROOT
        / 'tests/data/inspect/data_cyse2_vuln_exploit_challenges.json',
        aggregates=1,
        sidecars=1,
        # One scorer reporting three metrics, which is what makes this fixture worth
        # using: a converter that collapses them to one result fails here.
        results=3,
        model_id='mistral/mistral-large-latest',
        required_source_paths=(
            'eval.model',
            'eval.task',
            'eval.dataset.name',
            'results.scores.*.name',
            'results.scores.*.scorer',
            'results.scores.*.metrics',
            'results.total_samples',
        ),
    ),
    ConverterCase(
        source='helm',
        log_path=REPO_ROOT
        / 'tests/data/helm'
        / 'commonsense-dataset=hellaswag,method=multiple_choice_joint,'
        'model=eleutherai_pythia-1b-v0',
        aggregates=1,
        sidecars=1,
        # Eight metrics on the `valid` split, each also reported worst-case over the
        # robustness and fairness perturbations.
        results=24,
        model_id='eleutherai/pythia-1b-v0',
        # `--log_path` is a HELM run directory, not one file, so there is no single
        # payload for `missing_paths` to address. The gate and the counts above are
        # what cover this converter.
        required_source_paths=(),
    ),
)


def unavailable(case: ConverterCase) -> str | None:
    """Why this case cannot run here, or None if it can.

    A converter behind an optional extra states its own missing dependency in an
    `_..._IMPORT_ERROR` module global and raises from it when used. Reading that is what
    lets a case be declared once and skip, rather than fail, in the core install.
    """
    module = importlib.import_module(
        f'every_eval_ever.converters.{case.source}.adapter'
    )
    for name, value in vars(module).items():
        if name.endswith('_IMPORT_ERROR') and value is not None:
            return (
                f'{case.source} converter dependencies are missing: {value!r}. '
                f'Install with: uv sync --extra {case.source}'
            )
    return None


def convert(case: ConverterCase, tmp_path: Path) -> list[Path]:
    """Run the real CLI over a case's log; return the published record files."""
    from every_eval_ever import cli

    data_dir = tmp_path / 'data'
    exit_code = cli.main(
        [
            'convert',
            case.source,
            '--log_path',
            str(case.log_path),
            '--output_dir',
            str(data_dir),
            '--source_organization_name',
            'every-eval-ever-tests',
            *case.extra_argv,
        ]
    )
    assert exit_code == 0, f'{case.source} conversion exited {exit_code}'
    return sorted(
        path
        for path in data_dir.rglob('*')
        if path.is_file() and path.suffix in {'.json', '.jsonl'}
    )


def gate_complaints(paths: list[Path], capsys) -> list[dict[str, Any]]:
    """Run the validator CLI over `paths`; return every error and warning it reports.

    This is the merge gate: the semantic checks only run for a file at a canonical
    `data/<collection>/<developer>/<model>/` path, which is what the converters
    publish to, and `validate_file()` on its own leaves them off.
    """
    from every_eval_ever.validate import main as validate_main

    capsys.readouterr()  # drop what the conversion printed
    exit_code = validate_main(
        [str(path) for path in paths] + ['--format', 'json']
    )
    reports = json.loads(capsys.readouterr().out)
    complaints = [
        {
            'file': report['file'],
            'errors': report['errors'],
            'warnings': report['warnings'],
        }
        for report in reports
        if not report['valid'] or report['errors'] or report['warnings']
    ]
    if exit_code != 0 and not complaints:
        complaints.append(
            {
                'file': '<all>',
                'errors': ['validate exited non-zero'],
                'warnings': [],
            }
        )
    return complaints


def missing_paths(payload: Any, paths: tuple[str, ...]) -> list[str]:
    """Return the declared key paths that the given source payload does not have."""

    def resolve(node: Any, parts: list[str]) -> bool:
        if not parts:
            return True
        head, rest = parts[0], parts[1:]
        if head == '*':
            if isinstance(node, dict):
                return any(resolve(value, rest) for value in node.values())
            if isinstance(node, list):
                return any(resolve(item, rest) for item in node)
            return False
        if isinstance(node, dict) and head in node:
            return resolve(node[head], rest)
        return False

    return [path for path in paths if not resolve(payload, path.split('.'))]
