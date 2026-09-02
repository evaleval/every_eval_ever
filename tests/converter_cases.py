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
    # Rows in the instance-level sidecars, which is one per aggregate result a sample
    # contributed to, not one per sample.
    sidecar_rows: int | None = None
    model_id: str | None = None
    # Keyed by `<evaluation_name>/<evaluation_result_id>`, the pair that addresses one
    # result, falling back to the metric description for a converter that sets no
    # `evaluation_result_id`. A metric name alone would not do: HELM reports the same
    # metric on the `valid` split and worst-case over each perturbation.
    scores: dict[str, float] | None = None
    # Distinct `metric_config.metric_name` values across every result. The metric
    # belongs in this field rather than in `evaluation_name` or the description, and a
    # converter that leaves it unset shows up here as `None` in the set.
    metric_names: frozenset[str] | None = None
    # Distinct `metric_config.metric_id` values, which is what a consumer joins on
    # across sources. Listed per converter rather than derived, because the whole
    # question is whether this converter's spelling of a metric reaches the same id
    # another converter's does — a rule that generates both sides cannot answer it.
    metric_ids: frozenset[str] | None = None
    # The `score_details.uncertainty` keys, unioned over every result. Setting this
    # also requires every result to carry an uncertainty, since dropping the standard
    # error from a score leaves a record that still validates.
    uncertainty_keys: frozenset[str] | None = None
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
        metric_names=frozenset({'exact_match'}),
        # The registry carries exact match, so this joins with HELM's `exact_match`
        # and with any adapter reporting `em`.
        metric_ids=frozenset({'exact-match'}),
        # No `num_bootstrap_samples`: `exact_match` aggregates with `mean`, whose
        # standard error lm-eval computes analytically rather than by resampling.
        uncertainty_keys=frozenset({'standard_error', 'num_samples'}),
        required_source_paths=(
            'config.model',
            'config.model_args',
            'results.*',
            'configs.*.dataset_path',
        ),
    ),
    ConverterCase(
        source='lm_eval',
        log_path=REPO_ROOT
        / 'tests/data/lm_eval_v03/results_v03_no_comma_metric_keys.json',
        # lm-eval v0.3 and earlier write bare `acc` / `acc_stderr` keys, without the
        # `,filter` suffix v0.4 added. Only the comma form was recognised, so every
        # standard error here fell through as a metric of its own and a record
        # reported the spread as the score. Two tasks, two real metrics each.
        aggregates=2,
        results=4,
        model_id='bigscience/bloom-3b',
        scores={
            'sciq/acc': 0.891,
            'sciq/acc_norm': 0.816,
            'arc_easy/acc': 0.5942760942760943,
            'arc_easy/acc_norm': 0.5328282828282829,
        },
        metric_names=frozenset({'acc', 'acc_norm'}),
        # `acc_norm` is length-normalized accuracy, which the registry carries as
        # `normalized-accuracy` rather than folding into `accuracy`. Both ids being
        # present is what says the two were not merged.
        metric_ids=frozenset({'accuracy', 'normalized-accuracy'}),
        # Each score keeps its own companion standard error. No `num_samples`: the
        # v0.3 format records no `n-samples` block.
        uncertainty_keys=frozenset({'standard_error'}),
        required_source_paths=(
            'config.model',
            'config.model_args',
            'results.*',
        ),
    ),
    ConverterCase(
        source='inspect',
        log_path=REPO_ROOT
        / 'tests/data/inspect/data_cyse2_vuln_exploit_challenges.json',
        aggregates=1,
        sidecars=1,
        # One scorer reporting three metrics, which is what makes this fixture worth
        # using: a converter that collapses them to one result fails here. The third
        # is the scorer's `std`, which belongs in `uncertainty`, not in a score of
        # its own.
        results=2,
        # One sample, two aggregate results, so two rows.
        sidecar_rows=2,
        model_id='mistral/mistral-large-latest',
        scores={
            'inspect_evals/cyse2_vulnerability_exploit/'
            'vul_exploit_scorer:accuracy': 0.38108974358974373,
            'inspect_evals/cyse2_vulnerability_exploit/'
            'vul_exploit_scorer:mean': 0.38108974358974357,
        },
        metric_names=frozenset({'accuracy', 'mean'}),
        # `mean` is not a metric the registry can carry: it names an aggregation, and
        # what it averaged is the scorer's business, so it stays namespaced.
        metric_ids=frozenset({'accuracy', 'inspect_ai.mean'}),
        uncertainty_keys=frozenset({'standard_deviation', 'num_samples'}),
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
        # 10 instances against the 8 `valid` results; the perturbation results report no
        # per-instance stats, so no row joins to them.
        sidecar_rows=80,
        model_id='eleutherai/pythia-1b-v0',
        # The 24 results are these 8 metrics on `valid` plus each one's worst case over
        # the robustness and fairness perturbations, so the names are listed rather than
        # the 24 scores; `results` above is what counts them.
        metric_names=frozenset(
            {
                'exact_match',
                'exact_match@5',
                'quasi_exact_match',
                'quasi_exact_match@5',
                'prefix_exact_match',
                'prefix_exact_match@5',
                'quasi_prefix_exact_match',
                'quasi_prefix_exact_match@5',
            }
        ),
        # Only plain `exact_match` resolves; HELM's near-miss variants and its
        # best-of-k forms have no registry entry yet, so seven of the eight are
        # namespaced. This set is the concrete list of gaps to file upstream.
        metric_ids=frozenset(
            {
                'exact-match',
                'helm.exact_match@5',
                'helm.quasi_exact_match',
                'helm.quasi_exact_match@5',
                'helm.prefix_exact_match',
                'helm.prefix_exact_match@5',
                'helm.quasi_prefix_exact_match',
                'helm.quasi_prefix_exact_match@5',
            }
        ),
        # No standard deviation: HELM's spread is over train trials, and this run,
        # like nearly every published HELM run, has one.
        uncertainty_keys=frozenset({'num_samples'}),
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
