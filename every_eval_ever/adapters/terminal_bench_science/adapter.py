"""Convert Terminal-Bench-Science leaderboard data to the EvalEval schema.

Data source:
- Terminal-Bench-Science leaderboard: https://terminal-bench-science.ai
- JSON behind it: ``/api/leaderboard?package=<package>&name=<name>``

Terminal-Bench-Science is an agentic benchmark of expert-curated scientific
research workflows, run in a terminal sandbox by Harbor. Release 0.1 holds 70
tasks across five scientific domains, each attempted 3 times, so one
leaderboard row is 210 trials of one agent+model pair. The published accuracy
is the share of those *trials* that the task's verifier accepted.

Each row therefore becomes one ``EvaluationLog`` carrying six results: the
overall trial resolution rate plus one per domain (life, physical, earth,
mathematical, engineering). Every result records its own ``n`` and standard
error, and names its aggregation level, so a consumer can take the overall or
the parts without double-counting them.

Agent metadata, which EEE has no typed home for, is stored in
``model_info.additional_details``, as in the sibling ``terminal_bench_2``
adapter.

Usage:
    uv run python -m every_eval_ever.adapters.terminal_bench_science.adapter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from every_eval_ever.eval_types import (
    AgenticEvalConfig,
    AvailableTool,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    GenerationArgs,
    GenerationConfig,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataUrl,
    SourceMetadata,
    StandardError,
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    default_failure_report_path,
    fetch_json,
    require_finite_number,
    require_identity,
    sanitize_filename,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers import eval_card_registry as registry_mod

SITE_URL = 'https://terminal-bench-science.ai'
API_PATH = '/api/leaderboard'
#: Harbor package the leaderboard is published under.
PACKAGE = 'terminal-bench-science/terminal-bench-science'
#: The supported leaderboard; other releases need their own task configuration.
LEADERBOARD_NAME = 'v0-1-eval'
#: The benchmark release these rows were produced on.
BENCHMARK_VERSION = '0.1'

#: One collection for this source (fields.md §collection).
COLLECTION = 'terminal-bench-science'
OUTPUT_DIR = f'/tmp/{COLLECTION}-smoke/data/{COLLECTION}'

#: The dataset the eval ran on: the released task set, not these results.
DATASET_URLS = [
    'https://hub.harborframework.com/datasets/terminal-bench-science/terminal-bench-science/latest',
    'https://github.com/harbor-framework/terminal-bench-science',
]
BENCHMARK_DOI = 'https://doi.org/10.5281/zenodo.22110253'

TASK_COUNT = 70
TRIALS_PER_TASK = 3

#: Leaderboard domain key -> the name the benchmark publishes it under.
DOMAIN_NAMES = {
    'life': 'life-sciences',
    'physical': 'physical-sciences',
    'earth': 'earth-sciences',
    'mathematical': 'mathematical-sciences',
    'engineering': 'engineering-sciences',
}

#: Whether an organization published weights for the model lines on this
#: leaderboard, keyed by the canonical registry org id. The source states
#: neither deployment axis, so this is a curated, reviewable map rather than
#: something read off the page, and an organization missing from it yields
#: ``unknown`` rather than a guess. Every row is an API-served model driven by
#: an agent harness, so ``deployment_type`` is uniformly
#: ``externally_managed``.
MODEL_AVAILABILITY_BY_ORG = {
    'anthropic': 'closed_weights',
    'openai': 'closed_weights',
    'google-deepmind': 'closed_weights',
    'xai': 'closed_weights',
    'deepseek': 'open_weights',
    'moonshotai': 'open_weights',
    'zai': 'open_weights',
}


def leaderboard_api_url(
    package: str = PACKAGE,
    name: str = LEADERBOARD_NAME,
    site_url: str = SITE_URL,
) -> str:
    """Build the JSON endpoint the leaderboard page itself reads."""
    query = urlencode({'package': package, 'name': name})
    return f'{site_url.rstrip("/")}{API_PATH}?{query}'


def fetch_payload(url: str) -> dict[str, Any]:
    """Fetch the leaderboard payload (raw-captured by ``fetch_json``)."""
    payload = fetch_json(url, headers={'Accept': 'application/json'})
    if not isinstance(payload, dict):
        raise ValueError(
            f'leaderboard endpoint returned {type(payload).__name__}, '
            'expected an object'
        )
    return payload


def load_payload(path: Path) -> dict[str, Any]:
    """Load a saved payload for offline replay."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('--input-json must contain a leaderboard object')
    return payload


def save_payload(payload: dict[str, Any], path: Path | None) -> None:
    """Persist the exact fetched source outside the validated data tree."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8'
    )


def is_subpath(path: Path, parent: Path) -> bool:
    """Return whether a raw artifact would be placed inside output data."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _validated_leaderboard(payload: dict[str, Any]) -> dict[str, Any]:
    """Require the leaderboard whose release and trial configuration we support."""
    leaderboard = payload.get('leaderboard')
    if (
        not isinstance(leaderboard, dict)
        or leaderboard.get('package') != PACKAGE
        or leaderboard.get('name') != LEADERBOARD_NAME
    ):
        raise ValueError(
            f'unsupported leaderboard: expected {PACKAGE}/{LEADERBOARD_NAME}'
        )
    return leaderboard


def source_version(payload: dict[str, Any]) -> str:
    """A token that changes exactly when the published rows change.

    The leaderboard has no revision of its own, so this digests the row
    identities and their published state. A re-run over an unchanged
    leaderboard produces the same token, which is what lets the scheduler skip
    it.
    """
    leaderboard = _validated_leaderboard(payload)
    rows = payload.get('rows')
    rows = rows if isinstance(rows, list) else []
    fingerprint = [
        {
            'id': row.get('id'),
            'status': row.get('status'),
            'updated_at': row.get('updated_at'),
            'metrics': row.get('metrics'),
        }
        for row in rows
        if isinstance(row, dict)
    ]
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(',', ':')).encode(
            'utf-8'
        )
    ).hexdigest()
    leaderboard_id = leaderboard.get('id')
    return f'{leaderboard_id or "unknown"}:{len(fingerprint)}:{digest[:16]}'


def _link_label(value: Any, field_name: str) -> str:
    """Read the label out of one of the leaderboard's ``{url, label}`` cells."""
    if not isinstance(value, dict):
        raise ValueError(f'{field_name} must be a link object')
    return require_identity(value.get('label'), field_name)


def _link_url(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    url = value.get('url')
    return url if isinstance(url, str) and url.strip() else None


def slugify(value: str) -> str:
    """Lowercase a display name into an id component.

    Dots are kept: registry model ids spell versions that way
    (``gemini-2.5-pro``), and the sibling ``terminal_bench_2`` adapter builds
    its ids the same way.
    """
    return sanitize_filename(value.strip().lower().replace(' ', '-'))


def _percentage(value: Any, field_name: str) -> float:
    score = require_finite_number(value, field_name)
    if not 0.0 <= score <= 100.0:
        raise ValueError(
            f'{field_name} must be a percentage between 0 and 100, '
            f'got {value!r}'
        )
    return score


def _details(values: dict[str, Any]) -> dict[str, str]:
    """Stringify a detail map; EEE string-maps reject non-strings."""
    return {
        key: value if isinstance(value, str) else json.dumps(value)
        for key, value in values.items()
        if value is not None
    }


def _metric_config(description: str) -> MetricConfig:
    """The one metric this leaderboard publishes, on its own percent scale.

    Namespaced rather than the registry's global ``accuracy``: the published
    figure is the share of *trials* a verifier accepted across 3 attempts at
    each of 70 tasks, on the leaderboard's percent scale, and joining a
    trial-level resolution rate to plain ``accuracy`` would merge two different
    quantities. ``terminal_bench_2`` names its metric the same way, so the two
    Terminal-Bench leaderboards stay comparable with each other.
    """
    return MetricConfig(
        evaluation_description=description,
        metric_id='terminal-bench-science.accuracy',
        metric_name='Accuracy',
        metric_kind='accuracy',
        metric_unit='percent',
        lower_is_better=False,
        score_type=ScoreType.continuous,
        min_score=0.0,
        max_score=100.0,
    )


def _count(value: Any, field_name: str) -> int:
    """Read a non-negative whole count without truncating fractional values."""
    number = require_finite_number(value, field_name)
    if number < 0 or not number.is_integer():
        raise ValueError(
            f'{field_name} must be a non-negative integer, got {value!r}'
        )
    return int(number)


def _uncertainty(
    metrics: dict[str, Any], field_prefix: str, trials: int
) -> Uncertainty:
    """Standard error over the trials the score was computed on.

    The published figure matches the binomial standard error of the pass rate
    over ``n`` trials, so it is recorded as analytic rather than resampled.
    """
    stderr = require_finite_number(
        metrics.get('accuracy_stderr'), f'{field_prefix} accuracy_stderr'
    )
    if stderr < 0.0:
        raise ValueError(
            f'{field_prefix} accuracy_stderr must be non-negative, '
            f'got {stderr!r}'
        )
    return Uncertainty(
        standard_error=StandardError(value=stderr, method='analytic'),
        num_samples=trials,
    )


def _score_details(
    metrics: dict[str, Any],
    aggregation_level: str,
    field_prefix: str,
) -> ScoreDetails:
    trials = _count(metrics.get('tasks'), f'{field_prefix} tasks')
    passes = _count(metrics.get('passes'), f'{field_prefix} passes')
    if trials == 0 or trials % TRIALS_PER_TASK:
        raise ValueError(
            f'{field_prefix} tasks must be a positive multiple of '
            f'{TRIALS_PER_TASK} trials, got {trials!r}'
        )
    if not 0 <= passes <= trials:
        raise ValueError(
            f'{field_prefix} passes must be between 0 and {trials}, '
            f'got {passes!r}'
        )
    return ScoreDetails(
        score=_percentage(metrics.get('accuracy'), f'{field_prefix} accuracy'),
        uncertainty=_uncertainty(metrics, field_prefix, trials),
        details=_details(
            {
                # `trials` and `tasks` are separate numbers here: the
                # leaderboard's `tasks` field counts attempts, not tasks.
                'aggregation_level': aggregation_level,
                'trials': trials,
                'trials_passed': passes,
                'trials_per_task': TRIALS_PER_TASK,
                'tasks': trials // TRIALS_PER_TASK,
                'total_tokens': metrics.get('total_tokens'),
                'total_cost_usd': metrics.get('total_cost_usd'),
            }
        ),
    )


def _generation_config(
    agent: str,
    model_name: str,
    reasoning_effort: str | None,
    run_dataset: str,
) -> GenerationConfig:
    return GenerationConfig(
        generation_args=GenerationArgs(
            # Every published row ran a reasoning model at a named effort. The
            # schema has no typed effort field, so the level itself is kept
            # below rather than folded into this boolean.
            reasoning=bool(reasoning_effort),
            agentic_eval_config=AgenticEvalConfig(
                available_tools=[
                    AvailableTool(
                        name='terminal',
                        description=(
                            'Full terminal/shell access inside the task sandbox'
                        ),
                    ),
                ],
            ),
            execution_command=(
                f'harbor run -d {run_dataset} '
                f'-a "{agent}" -m "{model_name}" -k {TRIALS_PER_TASK}'
            ),
        ),
        additional_details=_details(
            {
                'agent_name': agent,
                'reasoning_effort': reasoning_effort,
                'trials_per_task': TRIALS_PER_TASK,
            }
        ),
    )


def _results(
    row: dict[str, Any],
    evaluation_id: str,
    evaluation_timestamp: str,
    generation_config: GenerationConfig,
    leaderboard_url: str,
) -> list[EvaluationResult]:
    """The overall trial resolution rate, then one result per domain."""
    metrics = row.get('metrics')
    if not isinstance(metrics, dict):
        raise ValueError('leaderboard row has no metrics object')
    source_data = SourceDataUrl(
        dataset_name=COLLECTION,
        source_type='url',
        url=list(DATASET_URLS),
        additional_details=_details(
            {
                'benchmark_version': BENCHMARK_VERSION,
                'benchmark_doi': BENCHMARK_DOI,
                'leaderboard_url': leaderboard_url,
            }
        ),
    )

    results = [
        EvaluationResult(
            evaluation_result_id=f'{evaluation_id}#accuracy',
            evaluation_name=f'terminal-bench-science-{BENCHMARK_VERSION}',
            source_data=source_data,
            evaluation_timestamp=evaluation_timestamp,
            metric_config=_metric_config(
                'Share of trials resolved across '
                f'{TASK_COUNT} expert-curated scientific research tasks, '
                f'{TRIALS_PER_TASK} trials each'
            ),
            score_details=_score_details(metrics, 'overall', 'row'),
            generation_config=generation_config,
        )
    ]

    domain_metrics = metrics.get('domain_metrics')
    if not isinstance(domain_metrics, dict):
        raise ValueError('row domain_metrics must be an object')
    for domain, name in DOMAIN_NAMES.items():
        scoped = domain_metrics.get(domain)
        if not isinstance(scoped, dict):
            raise ValueError(f'domain {domain} metrics must be an object')
        results.append(
            EvaluationResult(
                evaluation_result_id=f'{evaluation_id}#accuracy.{name}',
                evaluation_name=(
                    f'terminal-bench-science-{BENCHMARK_VERSION}.{name}'
                ),
                source_data=source_data,
                evaluation_timestamp=evaluation_timestamp,
                metric_config=_metric_config(
                    f'Share of trials resolved on the {name} subset'
                ),
                score_details=_score_details(
                    scoped, f'domain:{domain}', f'domain {domain}'
                ),
                generation_config=generation_config,
            )
        )
    return results


def convert_row(
    row: dict[str, Any],
    retrieved_timestamp: str,
    registry: registry_mod.Registry,
    leaderboard: dict[str, Any],
    leaderboard_url: str,
) -> tuple[EvaluationLog, str, str]:
    """Convert one leaderboard row into a log plus its datastore path components."""
    metadata = row.get('metadata')
    if not isinstance(metadata, dict):
        raise ValueError('leaderboard row has no metadata object')

    agent = _link_label(metadata.get('agent_display'), 'agent')
    agent_org = _link_label(metadata.get('agent_org'), 'agent organization')
    model_name = _link_label(metadata.get('model_display'), 'model')
    model_org = _link_label(metadata.get('model_org'), 'model developer')
    row_id = require_identity(row.get('id'), 'leaderboard row id')
    reasoning_effort = metadata.get('reasoning_effort')
    reasoning_effort = (
        reasoning_effort.strip()
        if isinstance(reasoning_effort, str) and reasoning_effort.strip()
        else None
    )

    # The registry has no model entity for these releases, so the id is built
    # from the source's own organization and model labels and published
    # unverified. Only the organization half is canonicalized.
    developer = registry.org(model_org)
    org_slug = slugify(developer.canonical_id or model_org)
    model_slug = slugify(model_name)
    model_id = f'{org_slug}/{model_slug}'

    # Keyed on the leaderboard's own stable row id, never on `now`: the same
    # row re-scraped tomorrow converts to the same record. The agent, model and
    # effort ride along so a reader can tell the variants apart, since one
    # model appears under several scaffolds.
    evaluation_id = '/'.join(
        (
            f'terminal-bench-science-{BENCHMARK_VERSION}',
            '__'.join(
                part
                for part in (slugify(agent), model_slug, reasoning_effort)
                if part
            ),
            row_id,
        )
    )
    evaluation_timestamp = require_identity(
        row.get('created_at'), 'leaderboard row created_at'
    )

    generation_config = _generation_config(
        agent=agent,
        model_name=model_name,
        reasoning_effort=reasoning_effort,
        run_dataset=f'{PACKAGE}@v{BENCHMARK_VERSION}',
    )

    return (
        EvaluationLog(
            schema_version=SCHEMA_VERSION,
            evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_timestamp,
            evaluation_timestamp=evaluation_timestamp,
            source_metadata=SourceMetadata(
                source_name=(
                    leaderboard.get('title')
                    or f'Terminal-Bench-Science {BENCHMARK_VERSION}'
                ),
                source_type='documentation',
                source_organization_name='Terminal-Bench-Science',
                source_organization_url=SITE_URL,
                evaluator_relationship=EvaluatorRelationship.third_party,
                additional_details=_details(
                    {
                        'leaderboard_id': leaderboard.get('id'),
                        'leaderboard_name': leaderboard.get('name'),
                        'leaderboard_url': leaderboard_url,
                        'harbor_package': PACKAGE,
                        'benchmark_version': BENCHMARK_VERSION,
                        'benchmark_doi': BENCHMARK_DOI,
                        'task_count': TASK_COUNT,
                        'trials_per_task': TRIALS_PER_TASK,
                        'row_updated_at': row.get('updated_at'),
                    }
                ),
            ),
            eval_library=EvalLibrary(
                name='harbor',
                version='unknown',
                # No canonical harness exists for Harbor yet; the strategy
                # recorded here says so rather than implying one was found.
                additional_details=_details(
                    registry.harness('harbor').provenance('harness')
                ),
            ),
            model_info=ModelInfo(
                name=model_name,
                id=model_id,
                developer=developer.canonical_id or model_org,
                additional_details=_details(
                    {
                        'deployment_type': 'externally_managed',
                        'model_availability': MODEL_AVAILABILITY_BY_ORG.get(
                            org_slug, 'unknown'
                        ),
                        'model_id_verified': 'false',
                        'model_id_source': 'leaderboard_labels',
                        'model_release_date': metadata.get(
                            'model_release_date'
                        ),
                        'model_url': _link_url(metadata.get('model_display')),
                        'agent_name': agent,
                        'agent_organization': agent_org,
                        'agent_url': _link_url(metadata.get('agent_display')),
                        **developer.provenance('developer'),
                    }
                ),
            ),
            evaluation_results=_results(
                row,
                evaluation_id,
                evaluation_timestamp,
                generation_config,
                leaderboard_url,
            ),
        ),
        org_slug,
        model_slug,
    )


def convert_payload(
    payload: dict[str, Any],
    output_dir: Path,
    retrieved_timestamp: str | None = None,
    registry: registry_mod.Registry | None = None,
    leaderboard_url: str = SITE_URL,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Convert every published row; account for the rest."""
    timestamp = retrieved_timestamp or str(time.time())
    registry = registry if registry is not None else registry_mod.Registry()
    leaderboard = _validated_leaderboard(payload)
    rows = payload.get('rows')
    if not isinstance(rows, list):
        raise ValueError('leaderboard payload has no rows array')

    outputs: list[EvaluationLogOutput] = []
    failures: list[SourceRecordFailure] = []
    exclusions: list[SourceRecordExclusion] = []
    for index, row in enumerate(rows, start=1):
        source_ref = f'leaderboard row {index}'
        if not isinstance(row, dict):
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason='leaderboard row is not an object',
                )
            )
            continue
        status = row.get('status')
        if status != 'display':
            # A row the leaderboard is not publishing is not a result to
            # convert; reported, but it does not fail the refresh.
            exclusions.append(
                SourceRecordExclusion(
                    source_ref=source_ref,
                    reason=f'row status is {status!r}, not "display"',
                    source_record=row,
                )
            )
            continue
        try:
            log, org_slug, model_slug = convert_row(
                row, timestamp, registry, leaderboard, leaderboard_url
            )
            outputs.append(
                EvaluationLogOutput(
                    eval_log=EvaluationLog.model_validate(log.model_dump()),
                    base_dir=output_dir,
                    developer=org_slug,
                    model_name=model_slug,
                )
            )
        except Exception as error:
            failures.append(
                SourceRecordFailure(
                    source_ref=source_ref,
                    reason=str(error),
                    source_record=row,
                )
            )
    if not outputs and not failures and not exclusions:
        failures.append(
            SourceRecordFailure(
                source_ref='Terminal-Bench-Science leaderboard',
                reason='converted 0 source records',
            )
        )
    return SourceConversionResult(
        source_name=f'Terminal-Bench-Science {BENCHMARK_VERSION}',
        total_records=len(rows),
        records=outputs,
        failures=failures,
        exclusions=exclusions,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Fetch and convert the Terminal-Bench-Science leaderboard.'
        ),
    )
    parser.add_argument(
        '--input-json',
        type=Path,
        help='Replay a saved leaderboard payload instead of fetching.',
    )
    parser.add_argument(
        '--save-raw-json',
        type=Path,
        help='Save the fetched leaderboard payload outside --output-dir.',
    )
    parser.add_argument(
        '--site-url',
        default=SITE_URL,
        help=f'Leaderboard site (default: {SITE_URL}).',
    )
    parser.add_argument(
        '--package',
        default=PACKAGE,
        choices=[PACKAGE],
        help='The supported Terminal-Bench-Science Harbor package.',
    )
    parser.add_argument(
        '--leaderboard-name',
        default=LEADERBOARD_NAME,
        choices=[LEADERBOARD_NAME],
        help='The supported release 0.1 leaderboard.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(OUTPUT_DIR),
        help=f'Collection directory (default: {OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--failure-report',
        type=Path,
        help=(
            'Write rejected source rows and reasons here. Defaults beside '
            '--output-dir when any row fails.'
        ),
    )
    parser.add_argument(
        '--no-registry-resolve',
        action='store_true',
        help='Skip eval-card-registry lookups and keep the source spellings.',
    )
    parser.add_argument(
        '--registry-live',
        action='store_true',
        help='Also consult the hosted registry for values the snapshot lacks.',
    )
    parser.add_argument(
        '--emit-source-version',
        action='store_true',
        help=(
            'Print a token for the current leaderboard state and exit, so the '
            'scheduler can skip a run whose source has not changed.'
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    url = leaderboard_api_url(
        args.package, args.leaderboard_name, args.site_url
    )
    if args.emit_source_version:
        payload = (
            load_payload(args.input_json)
            if args.input_json is not None
            else fetch_payload(url)
        )
        print(source_version(payload))
        return 0
    if args.save_raw_json is not None and is_subpath(
        args.save_raw_json, args.output_dir
    ):
        raise SystemExit(
            '--save-raw-json must point outside --output-dir so the '
            'validator cannot mistake source JSON for evaluation data'
        )

    if args.input_json is not None:
        payload = load_payload(args.input_json)
    else:
        payload = fetch_payload(url)
        save_payload(payload, args.save_raw_json)

    registry = registry_mod.Registry(
        enabled=not args.no_registry_resolve,
        live=args.registry_live,
    )
    result = convert_payload(
        payload,
        args.output_dir,
        registry=registry,
        leaderboard_url=args.site_url,
    )
    paths = save_evaluation_logs(result.records)
    for path in paths:
        print(path)
    print(f'Generated {len(paths)} files in {args.output_dir}/')
    for exclusion in result.exclusions:
        print(f'Excluded {exclusion.source_ref}: {exclusion.reason}')
    if result.failures or result.exclusions:
        report_path = save_failure_report(
            result,
            args.failure_report or default_failure_report_path(args.output_dir),
        )
        print(f'Failure report: {report_path}')
        result.raise_if_incomplete()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
