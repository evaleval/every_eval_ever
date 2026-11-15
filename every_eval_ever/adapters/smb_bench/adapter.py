"""Convert SMB Bench results to the EvalEval schema.

Data source:
- Hugging Face dataset ``oro-agents/smb-bench`` (CC BY 4.0,
  https://doi.org/10.57967/hf/11641), files ``data/runs.parquet`` (one row
  per run), ``data/models.parquet`` (served model ids and request settings)
  and ``data/prices.parquet`` (how each model's cost was computed), read at
  a pinned revision.

SMB Bench (Small-to-Midsize Business Bench, by ORO AI) is an agentic benchmark:
a model runs the buying desk of a simulated produce wholesaler for a 90-day
quarter, and the score is the business's profit in US dollars. Six businesses
are each played as a normal quarter, a customer quarter and a supplier quarter
(18 tasks), and every model runs every task 3 times.

Each model becomes one ``EvaluationLog`` carrying six results: mean profit
over all 18 tasks (the benchmark's headline), the same statistic on each of
the three kinds of quarter, mean profit minus the benchmark's fixed buying
rule, and mean inference cost per scored run. A figure is the mean over tasks
of the mean of a model's scored runs on that task, as the benchmark publishes
it. Failed runs carry no score upstream; they are left out of every mean and
counted in ``score_details.details``.

Usage:
    uv run python -m every_eval_ever.adapters.smb_bench.adapter
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    SourceDataHf,
    SourceMetadata,
    StandardError,
    Uncertainty,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordFailure,
    default_failure_report_path,
    raw_capture,
    require_finite_number,
    require_identity,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers import eval_card_registry as registry_mod
from every_eval_ever.helpers.io import datastore_path_components

HF_REPO = 'oro-agents/smb-bench'
#: The dataset revision the published records were converted from. A new
#: revision that adds models is a deliberate bump of this constant.
DEFAULT_REVISION = '23486462ddbb916dc241adb1c4f8f8150f9bc1db'
RUNS_FILE = 'data/runs.parquet'
MODELS_FILE = 'data/models.parquet'
PRICES_FILE = 'data/prices.parquet'

#: One collection for this source.
COLLECTION = 'smb-bench'
OUTPUT_DIR = f'/tmp/{COLLECTION}-smoke/data/{COLLECTION}'

SOURCE_ORGANIZATION = 'ORO AI'
SOURCE_ORGANIZATION_URL = 'https://oroagents.com'
DATASET_URL = f'https://huggingface.co/datasets/{HF_REPO}'
DATASET_DOI = 'https://doi.org/10.57967/hf/11641'
DATASET_LICENSE = 'CC BY 4.0'
REPORT_URL = 'https://oroagents.com/benchmarks/smb-bench'
RUN_PAGES_URL = 'https://oroagents.com/smb-bench/runs'
GITHUB_URL = 'https://github.com/ORO-AI/smb-bench'
HARBOR_HUB_URL = 'https://hub.harborframework.com/datasets/oro-agents/smb-bench'

#: Which kind of quarter each task is. The run table separates the normal
#: quarters (``split == 'control'``) from the changed ones
#: (``split == 'leaderboard'``) but does not say which changed quarter is a
#: customer quarter and which a supplier quarter; the benchmark's public run
#: pages (RUN_PAGES_URL) list every task under its kind, and this is that
#: listing. ``aggregate_model`` checks it against ``split`` on every row.
QUARTER_KIND_BY_TASK = {
    'smb-bench-control-1': 'normal',
    'smb-bench-control-2': 'normal',
    'smb-bench-control-3': 'normal',
    'smb-bench-control-4': 'normal',
    'smb-bench-control-5': 'normal',
    'smb-bench-control-6': 'normal',
    'smb-bench-public-1': 'customer',
    'smb-bench-public-2': 'customer',
    'smb-bench-public-5': 'customer',
    'smb-bench-public-7': 'customer',
    'smb-bench-public-9': 'customer',
    'smb-bench-public-11': 'customer',
    'smb-bench-public-3': 'supplier',
    'smb-bench-public-4': 'supplier',
    'smb-bench-public-6': 'supplier',
    'smb-bench-public-8': 'supplier',
    'smb-bench-public-10': 'supplier',
    'smb-bench-public-12': 'supplier',
}
SPLIT_BY_KIND = {
    'normal': 'control',
    'customer': 'leaderboard',
    'supplier': 'leaderboard',
}
KIND_DESCRIPTIONS = {
    'normal': 'the normal quarters, where nothing unusual happens',
    'customer': (
        'the customer quarters, where accounts that are delivered short '
        'quietly cut their orders'
    ),
    'supplier': (
        'the supplier quarters, where shorted accounts cut their orders and '
        'the cheapest supplier also starts arriving late and short'
    ),
}

#: The agent every leaderboard run used, as named in the dataset's published
#: Harbor job configs (``traces/runs/<model>/config.json``).
AGENT_NAME = 'smb-bench-buyer'
#: Per-run limits stated in the dataset card ("How a run works") that the
#: ``models`` config does not carry.
MAX_MODEL_TURNS_PER_DAY = 100
RUN_COST_CAP_USD = 500
#: The tools the dataset card lists for the buying desk.
#: Descriptions keep the card's wording; ``memory_*`` is how the card names
#: its family of note-keeping tools, not one tool's name.
_SQL_TOOL = "Query the business's records with SQL"
TOOLS = (
    ('query_records', _SQL_TOOL),
    ('list_tables', _SQL_TOOL),
    ('describe_table', _SQL_TOOL),
    ('get_business_status', 'Read the status'),
    ('list_offers', 'See supplier offers'),
    ('place_purchase_order', 'Place purchase orders'),
    ('set_shortfall_cover', 'Set local-market cover for account orders'),
    (
        'memory_*',
        'A family of tools for keeping notes (at most 16 files and '
        '32,000 bytes)',
    ),
    (
        'advance_day',
        'End the morning; does not count against the tool-call limit',
    ),
)

FIXED_RULE_DESCRIPTION = (
    "The benchmark's fixed buying rule plays every task in the same world "
    'and never reacts to the hidden consequences. It is a scripted '
    'reference, not a model; its profit per task is the run table column '
    'reference_return_cents.'
)
TASK_MEAN_NOTE = (
    "mean over tasks of the mean of the model's scored runs on that task, "
    'so every task weighs the same'
)
SE_METHOD = (
    'analytic: sample standard deviation (n-1) of the per-task values '
    'divided by sqrt(number of tasks); the task is the sampling unit and '
    'repeats are averaged within a task first'
)

REGISTRY_SNAPSHOT = Path(__file__).with_name('registry_snapshot.json')


def _details(values: dict[str, Any]) -> dict[str, str]:
    """Stringify a detail map; EEE string-maps reject non-strings."""
    return {
        key: value if isinstance(value, str) else json.dumps(value)
        for key, value in values.items()
        if value is not None
    }


def load_registry_snapshot() -> dict[str, Any]:
    """Model ids pinned from the eval-card-registry (see the README)."""
    return json.loads(REGISTRY_SNAPSHOT.read_text(encoding='utf-8'))


# -- source ---------------------------------------------------------------


def fetch_source(revision: str, target: Path) -> Path:
    """Download the three source tables at a pinned revision."""
    from huggingface_hub import hf_hub_download

    for name in (RUNS_FILE, MODELS_FILE, PRICES_FILE):
        hf_hub_download(
            HF_REPO,
            name,
            repo_type='dataset',
            revision=revision,
            local_dir=target,
        )
    raw_capture.record_hf_dataset(
        HF_REPO, revision=revision, label='SMB Bench run table'
    )
    return target


def _read_parquet(path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Rows and file-level key/value metadata of one parquet file."""
    import duckdb

    connection = duckdb.connect()
    try:
        cursor = connection.execute(
            'SELECT * FROM read_parquet(?)', [str(path)]
        )
        columns = [column[0] for column in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        metadata = {
            bytes(key).decode('utf-8'): bytes(value).decode(
                'utf-8', errors='replace'
            )
            for key, value in connection.execute(
                'SELECT key, value FROM parquet_kv_metadata(?)', [str(path)]
            ).fetchall()
        }
    finally:
        connection.close()
    return rows, metadata


def load_source(source_dir: Path, revision: str) -> dict[str, Any]:
    """Read the run, model and price tables from a dataset checkout."""
    runs_path = source_dir / RUNS_FILE
    runs, metadata = _read_parquet(runs_path)
    models, _ = _read_parquet(source_dir / MODELS_FILE)
    prices, _ = _read_parquet(source_dir / PRICES_FILE)
    cutoff = metadata.get('campaign_cutoff_utc')
    return {
        'revision': revision,
        'runs': runs,
        'models': {row['model']: row for row in models},
        'prices': {row['model']: row for row in prices},
        'runs_sha256': hashlib.sha256(runs_path.read_bytes()).hexdigest(),
        'dataset_version': require_identity(
            metadata.get('dataset'), 'run table metadata "dataset"'
        ),
        'campaign_cutoff_utc': _utc_iso(cutoff) if cutoff else None,
    }


# -- aggregation ----------------------------------------------------------


def _utc_iso(value: str) -> str:
    """The run table's naive UTC timestamps, with the offset made explicit."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc).isoformat()
    return parsed.astimezone(timezone.utc).isoformat()


def _rows_digest(rows: list[dict[str, Any]]) -> str:
    """A token that changes exactly when one model's run rows change."""
    canonical = json.dumps(
        sorted(rows, key=lambda row: str(row.get('cell_id'))),
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _task_mean_stats(values: list[float]) -> dict[str, Any]:
    """Mean over tasks with its standard error; the task is the unit."""
    count = len(values)
    spread = statistics.stdev(values) if count > 1 else None
    return {
        'mean': statistics.fmean(values),
        'n_tasks': count,
        'standard_deviation': spread,
        'standard_error': (
            spread / math.sqrt(count) if spread is not None else None
        ),
    }


def _reasoning_flag(setting: Any) -> bool | None:
    """True when the setting names an effort or thinking level, else unset."""
    text = str(setting or '').lower()
    if not any(word in text for word in ('effort', 'thinking')):
        return None
    if any(word in text for word in ('none', 'off', 'disabled', 'default')):
        return None
    return True


def aggregate_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce one model's run rows to the figures the benchmark publishes.

    A run is scored when its ``status`` is ``valid``. A task's value is the
    mean of the model's scored runs on it; a task with no scored run is left
    out of the task mean, as upstream does.
    """
    profit_by_task: dict[str, list[float]] = defaultdict(list)
    rule_by_task: dict[str, set[float]] = defaultdict(set)
    costs: list[float] = []
    days: set[int] = set()
    unscored: list[dict[str, Any]] = []
    status_counts: dict[str, int] = defaultdict(int)
    tasks_seen: set[str] = set()
    cells_seen: set[tuple[str, int]] = set()
    for row in rows:
        task = require_identity(row.get('public_task_id'), 'public_task_id')
        cell_key = (task, int(row['repeat']))
        if cell_key in cells_seen:
            raise ValueError(
                f'task {task!r} repeat {cell_key[1]} appears more than once'
            )
        cells_seen.add(cell_key)
        kind = QUARTER_KIND_BY_TASK.get(task)
        if kind is None:
            raise ValueError(f'task {task!r} has no known kind of quarter')
        if row.get('split') != SPLIT_BY_KIND[kind]:
            raise ValueError(
                f'task {task!r} is listed as a {kind} quarter but its split '
                f'is {row.get("split")!r}'
            )
        tasks_seen.add(task)
        status = require_identity(row.get('status'), 'status')
        status_counts[status] += 1
        if status != 'valid':
            unscored.append({'cell_id': row.get('cell_id'), 'status': status})
            continue
        cell = row.get('cell_id')
        profit_by_task[task].append(
            require_finite_number(
                row.get('business_return_cents'),
                f'{cell} business_return_cents',
            )
            / 100.0
        )
        rule_by_task[task].add(
            require_finite_number(
                row.get('reference_return_cents'),
                f'{cell} reference_return_cents',
            )
            / 100.0
        )
        cost = require_finite_number(row.get('cost_usd'), f'{cell} cost_usd')
        if cost < 0.0:
            raise ValueError(f'{cell} cost_usd is negative: {cost!r}')
        costs.append(cost)
        days.add(int(row['days_completed']))
    if not costs:
        raise ValueError('model has no scored run')
    for task, values in rule_by_task.items():
        if len(values) != 1:
            raise ValueError(
                f'task {task!r} has more than one fixed-rule profit: '
                f'{sorted(values)!r}'
            )

    task_profit = {
        task: statistics.fmean(values)
        for task, values in profit_by_task.items()
    }
    task_rule = {task: next(iter(rule_by_task[task])) for task in task_profit}

    def scope(kinds: tuple[str, ...]) -> dict[str, Any]:
        tasks = sorted(
            task for task in task_profit if QUARTER_KIND_BY_TASK[task] in kinds
        )
        if not tasks:
            raise ValueError(f'no scored task among the {kinds} quarters')
        scoped_rows = [
            row
            for row in rows
            if QUARTER_KIND_BY_TASK[row['public_task_id']] in kinds
        ]
        return {
            'profit': _task_mean_stats([task_profit[task] for task in tasks]),
            'gain': _task_mean_stats(
                [task_profit[task] - task_rule[task] for task in tasks]
            ),
            'fixed_rule_mean_profit_usd': statistics.fmean(
                task_rule[task] for task in tasks
            ),
            'n_tasks_in_scope': len(
                {row['public_task_id'] for row in scoped_rows}
            ),
            'n_runs': len(scoped_rows),
            'n_scored_runs': sum(
                1 for row in scoped_rows if row['status'] == 'valid'
            ),
        }

    return {
        'overall': scope(('normal', 'customer', 'supplier')),
        'kinds': {kind: scope((kind,)) for kind in KIND_DESCRIPTIONS},
        'cost_per_scored_run_usd': statistics.fmean(costs),
        'cost_total_scored_runs_usd': math.fsum(costs),
        'n_tasks': len(tasks_seen),
        'n_runs': len(rows),
        'n_scored_runs': len(costs),
        'status_counts': dict(sorted(status_counts.items())),
        'unscored_runs': unscored,
        'n_resumed_runs': sum(1 for row in rows if row.get('recovered')),
        'repeats_per_task': max(int(row['repeat']) for row in rows),
        'days_per_run': sorted(days),
        'first_run_started_utc': _utc_iso(
            min(row['started_at'] for row in rows if row.get('started_at'))
        ),
        'last_run_finished_utc': _utc_iso(
            max(row['finished_at'] for row in rows if row.get('finished_at'))
        ),
    }


# -- records --------------------------------------------------------------


def _uncertainty(stats: dict[str, Any]) -> Uncertainty | None:
    if stats['standard_error'] is None:
        return None
    return Uncertainty(
        standard_error=StandardError(
            value=stats['standard_error'], method=SE_METHOD
        ),
        standard_deviation=stats['standard_deviation'],
        num_samples=stats['n_tasks'],
    )


def _profit_metric(description: str) -> MetricConfig:
    """Profit in US dollars: unbounded in both directions by definition."""
    return MetricConfig(
        evaluation_description=description,
        metric_id='smb-bench.mean-profit-usd',
        metric_name='Mean profit per quarter',
        metric_kind='profit',
        metric_unit='usd',
        lower_is_better=False,
        score_type=ScoreType.continuous,
        min_score=-math.inf,
        max_score=math.inf,
        additional_details=_details(
            {
                'aggregation': TASK_MEAN_NOTE,
                'score_definition': (
                    "graded change in the business's net value over the "
                    '90-day quarter (run table column '
                    'business_return_cents, converted to dollars)'
                ),
                'fixed_rule_baseline': FIXED_RULE_DESCRIPTION,
            }
        ),
    )


def _task_phrase(scoped: dict[str, Any]) -> str:
    """Say how many tasks a figure covers, never more than were scored."""
    total, scored = scoped['n_tasks_in_scope'], scoped['profit']['n_tasks']
    if scored == total:
        return f'all {total} tasks'
    return f'the {scored} of {total} tasks that have a scored run'


def _scope_details(
    level: str, scoped: dict[str, Any], aggregate: dict[str, Any]
) -> dict[str, str]:
    return _details(
        {
            'aggregation_level': level,
            'tasks': scoped['n_tasks_in_scope'],
            'tasks_with_a_scored_run': scoped['profit']['n_tasks'],
            'runs': scoped['n_runs'],
            'scored_runs': scoped['n_scored_runs'],
            'unscored_runs': scoped['n_runs'] - scoped['n_scored_runs'],
            'repeats_per_task': aggregate['repeats_per_task'],
            'fixed_rule_mean_profit_usd': scoped['fixed_rule_mean_profit_usd'],
        }
    )


def _results(
    aggregate: dict[str, Any],
    evaluation_id: str,
    evaluation_name: str,
    source_data: Any,
    generation_config: GenerationConfig,
    price_row: dict[str, Any],
) -> list[EvaluationResult]:
    """The headline, the three kinds of quarter, the gain and the cost.

    ``source_data`` builds the dataset reference for a given number of tasks,
    so a per-kind result states the 6 tasks it covers rather than all 18.
    """
    overall = aggregate['overall']
    repeats = aggregate['repeats_per_task']
    unscored = aggregate['unscored_runs']

    def result(suffix, name, metric, score_details, tasks=None):
        return EvaluationResult(
            evaluation_result_id=f'{evaluation_id}#{suffix}',
            evaluation_name=name,
            source_data=source_data(
                aggregate['n_tasks'] if tasks is None else tasks
            ),
            metric_config=metric,
            score_details=score_details,
            generation_config=generation_config,
        )

    results = [
        result(
            'mean-profit-usd',
            evaluation_name,
            _profit_metric(
                f'Mean profit in US dollars per simulated 90-day quarter '
                f'over {_task_phrase(overall)}, {repeats} runs each'
            ),
            ScoreDetails(
                score=overall['profit']['mean'],
                uncertainty=_uncertainty(overall['profit']),
                details={
                    **_scope_details('overall', overall, aggregate),
                    **_details(
                        {
                            'run_status_counts': aggregate['status_counts'],
                            'unscored_run_ids': unscored or None,
                            'resumed_runs': aggregate['n_resumed_runs'],
                        }
                    ),
                },
            ),
        )
    ]
    for kind, description in KIND_DESCRIPTIONS.items():
        scoped = aggregate['kinds'][kind]
        results.append(
            result(
                f'mean-profit-usd.{kind}-quarters',
                f'{evaluation_name}.{kind}-quarters',
                _profit_metric(
                    'Mean profit in US dollars per simulated 90-day quarter '
                    f'on {description} ({_task_phrase(scoped)})'
                ),
                ScoreDetails(
                    score=scoped['profit']['mean'],
                    uncertainty=_uncertainty(scoped['profit']),
                    details=_scope_details(f'kind:{kind}', scoped, aggregate),
                ),
                tasks=scoped['n_tasks_in_scope'],
            )
        )
    results.append(
        result(
            'mean-profit-vs-fixed-rule-usd',
            evaluation_name,
            MetricConfig(
                evaluation_description=(
                    "Mean profit minus the fixed buying rule's profit on "
                    f'the same task, in US dollars, over {_task_phrase(overall)}'
                ),
                metric_id='smb-bench.mean-profit-vs-fixed-rule-usd',
                metric_name='Mean profit against the fixed buying rule',
                metric_kind='profit',
                metric_unit='usd',
                lower_is_better=False,
                score_type=ScoreType.continuous,
                min_score=-math.inf,
                max_score=math.inf,
                additional_details=_details(
                    {
                        'aggregation': (
                            'mean over tasks of (the mean of the '
                            "model's scored runs on the task minus the fixed "
                            "rule's profit on that task)"
                        ),
                        'fixed_rule_baseline': FIXED_RULE_DESCRIPTION,
                    }
                ),
            ),
            ScoreDetails(
                score=overall['gain']['mean'],
                uncertainty=_uncertainty(overall['gain']),
                details=_scope_details('overall', overall, aggregate),
            ),
        )
    )
    results.append(
        result(
            'cost-per-task',
            evaluation_name,
            MetricConfig(
                evaluation_description=(
                    'Mean inference cost in US dollars of one scored run '
                    '(one task attempt, a 90-day quarter), at the prices the '
                    "benchmark pins in its dataset's prices table"
                ),
                # The registry's canonical metric: [0, inf], lower is better.
                metric_id='cost-per-task',
                metric_name='Inference cost per run',
                metric_kind='cost',
                metric_unit='usd',
                lower_is_better=True,
                score_type=ScoreType.continuous,
                min_score=0.0,
                max_score=math.inf,
                additional_details=_details(
                    {
                        'aggregation': (
                            "mean of cost_usd over the model's scored runs"
                        ),
                        'cost_basis': (
                            'completed model calls costed from the price '
                            "table in the dataset's prices config; an "
                            'estimate, not invoices'
                        ),
                    }
                ),
            ),
            ScoreDetails(
                score=aggregate['cost_per_scored_run_usd'],
                details=_details(
                    {
                        'aggregation_level': 'overall',
                        'scored_runs': aggregate['n_scored_runs'],
                        'total_cost_of_scored_runs_usd': aggregate[
                            'cost_total_scored_runs_usd'
                        ],
                        # How this model's cost was computed, in the price
                        # table's own words.
                        'cost_method': price_row.get('cost_method'),
                        'price_table_date': price_row.get('price_table_date'),
                    }
                ),
            ),
        )
    )
    return results


def _model_identity(
    label: str,
    model_row: dict[str, Any],
    registry: registry_mod.Registry,
    snapshot: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Return ``(model id, developer, provenance details)`` for one model.

    The id comes from the vendored registry snapshot. A model the snapshot
    does not hold, or any model when the registry is disabled, keeps the
    source's own spelling and is marked unverified; that never fails the run.
    """
    require_identity(model_row.get('model_id'), 'served model id')
    # The models table's `provider` column is the model's maker as the source
    # lists it. It is only the fallback developer, and is not published.
    provider = require_identity(model_row.get('provider'), 'provider')
    pinned = (
        (snapshot.get('models') or {}).get(label) if registry.enabled else None
    )
    if pinned and pinned.get('canonical_id'):
        open_weights = pinned.get('open_weights')
        return (
            pinned['canonical_id'],
            pinned.get('org_id') or pinned['canonical_id'].split('/')[0],
            {
                'model_availability': (
                    'unknown'
                    if open_weights is None
                    else 'open_weights'
                    if open_weights
                    else 'closed_weights'
                ),
                'model_id_source': 'eval-card-registry',
                'model_registry_strategy': f'snapshot_{pinned["strategy"]}',
                'model_registry_review_status': pinned.get('review_status'),
                'model_registry_matched_value': pinned.get('matched_value'),
                'model_registry_matched_field': pinned.get('matched_field'),
                'model_registry_revision': snapshot['_meta'][
                    'registry_revision'
                ],
            },
        )
    developer = registry.org(provider)
    return (
        f'{developer.canonical_id or provider}/{label}',
        developer.canonical_id or provider,
        {
            'model_availability': 'unknown',
            'model_id_verified': 'false',
            'model_id_source': 'source_labels',
            'model_registry_strategy': (
                'no_canonical' if registry.enabled else 'registry_disabled'
            ),
            **developer.provenance('developer'),
        },
    )


def convert_model(
    label: str,
    rows: list[dict[str, Any]],
    source: dict[str, Any],
    retrieved_timestamp: str,
    registry: registry_mod.Registry,
    snapshot: dict[str, Any],
) -> tuple[EvaluationLog, str, str]:
    """Convert one model's runs into a log plus its datastore path."""
    model_row = source['models'].get(label)
    if model_row is None:
        raise ValueError(f'model {label!r} is not in the models table')
    aggregate = aggregate_model(rows)
    model_id, developer, identity = _model_identity(
        label, model_row, registry, snapshot
    )
    _, developer_dir, model_dir = datastore_path_components(
        COLLECTION, model_id, developer
    )
    dataset_version = source['dataset_version']
    reasoning = model_row.get('reasoning')
    # Keyed on the source's own model label and on this model's own run rows,
    # so a re-run gives the same id, and neither a card-only revision of the
    # dataset nor another model being added changes it. The registry id is a
    # join key and stays out of the record's identity.
    evaluation_id = '/'.join(
        (
            dataset_version,
            f'{AGENT_NAME}__{label}',
            f'runs-{_rows_digest(rows)[:12]}',
        )
    )

    def source_data(tasks: int) -> SourceDataHf:
        return SourceDataHf(
            dataset_name=COLLECTION,
            source_type='hf_dataset',
            hf_repo=HF_REPO,
            samples_number=tasks,
            additional_details=_details(
                {
                    'hf_revision': source['revision'],
                    'dataset_version': dataset_version,
                    'dataset_doi': DATASET_DOI,
                    'dataset_license': DATASET_LICENSE,
                    'harbor_hub_url': HARBOR_HUB_URL,
                    'report_url': REPORT_URL,
                }
            ),
        )

    max_tokens = model_row.get('max_output_tokens')
    generation_config = GenerationConfig(
        generation_args=GenerationArgs(
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            # The schema has no typed effort field; the setting itself is
            # kept in additional_details. Only a setting that names an effort
            # or thinking level is recorded as reasoning; anything else (the
            # provider default, for one) leaves the flag unset.
            reasoning=_reasoning_flag(reasoning),
            agentic_eval_config=AgenticEvalConfig(
                available_tools=[
                    AvailableTool(name=name, description=description)
                    for name, description in TOOLS
                ],
            ),
        ),
        additional_details=_details(
            {
                'agent_name': AGENT_NAME,
                'served_model_id': model_row.get('model_id'),
                'reasoning_setting': reasoning,
                'temperature': model_row.get('temperature'),
                'prompt_version': model_row.get('prompt_version'),
                'max_tool_calls_per_day': model_row.get(
                    'max_tool_calls_per_day'
                ),
                'max_model_turns_per_day': MAX_MODEL_TURNS_PER_DAY,
                'run_cost_cap_usd': RUN_COST_CAP_USD,
                'simulated_days_per_run': aggregate['days_per_run'],
                'repeats_per_task': aggregate['repeats_per_task'],
            }
        ),
    )
    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        retrieved_timestamp=retrieved_timestamp,
        evaluation_timestamp=aggregate['last_run_finished_utc'],
        source_metadata=SourceMetadata(
            source_name='SMB Bench',
            source_type='evaluation_run',
            source_organization_name=SOURCE_ORGANIZATION,
            source_organization_url=SOURCE_ORGANIZATION_URL,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details=_details(
                {
                    'dataset_url': DATASET_URL,
                    'dataset_doi': DATASET_DOI,
                    'dataset_license': DATASET_LICENSE,
                    'hf_revision': source['revision'],
                    'runs_file': RUNS_FILE,
                    'runs_file_sha256': source['runs_sha256'],
                    'dataset_version': dataset_version,
                    'campaign_cutoff_utc': source['campaign_cutoff_utc'],
                    'report_url': REPORT_URL,
                    'run_pages_url': RUN_PAGES_URL,
                    'github_url': GITHUB_URL,
                    'first_run_started_utc': aggregate['first_run_started_utc'],
                    'last_run_finished_utc': aggregate['last_run_finished_utc'],
                }
            ),
        ),
        eval_library=EvalLibrary(
            name=COLLECTION,
            version='unknown',
            additional_details=_details(
                {
                    'harness': (
                        "the benchmark's own buyer loop (agent "
                        f'{AGENT_NAME}); harness builds per model are '
                        'listed in the dataset card, section "Runs, '
                        'harness build and reruns"'
                    ),
                }
            ),
        ),
        model_info=ModelInfo(
            name=label,
            id=model_id,
            developer=developer,
            additional_details=_details(
                {
                    # Every model was served through a provider API.
                    'deployment_type': 'externally_managed',
                    'served_model_id': model_row.get('model_id'),
                    **identity,
                }
            ),
        ),
        evaluation_results=_results(
            aggregate,
            evaluation_id,
            dataset_version,
            source_data,
            generation_config,
            source['prices'].get(label) or {},
        ),
    )
    return log, developer_dir, model_dir


def convert_source(
    source: dict[str, Any],
    output_dir: Path,
    retrieved_timestamp: str | None = None,
    registry: registry_mod.Registry | None = None,
    snapshot: dict[str, Any] | None = None,
) -> SourceConversionResult[EvaluationLogOutput]:
    """Convert every model in the run table; account for the rest."""
    timestamp = retrieved_timestamp or str(time.time())
    registry = registry if registry is not None else registry_mod.Registry()
    snapshot = snapshot if snapshot is not None else load_registry_snapshot()
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source['runs']:
        by_model[row.get('model')].append(row)

    outputs: list[EvaluationLogOutput] = []
    failures: list[SourceRecordFailure] = []
    for label in sorted(by_model, key=str):
        rows = by_model[label]
        try:
            log, developer_dir, model_dir = convert_model(
                require_identity(label, 'model'),
                rows,
                source,
                timestamp,
                registry,
                snapshot,
            )
            outputs.append(
                EvaluationLogOutput(
                    eval_log=EvaluationLog.model_validate(log.model_dump()),
                    base_dir=output_dir,
                    developer=developer_dir,
                    model_name=model_dir,
                )
            )
        except Exception as error:
            failures.append(
                SourceRecordFailure(
                    source_ref=f'model {label!r} ({len(rows)} run rows)',
                    reason=str(error),
                    source_record={
                        'model': label,
                        'cell_ids': [row.get('cell_id') for row in rows],
                    },
                )
            )
    if not outputs and not failures:
        failures.append(
            SourceRecordFailure(
                source_ref='SMB Bench run table',
                reason='converted 0 source records',
            )
        )
    return SourceConversionResult(
        source_name=f'SMB Bench {source["dataset_version"]}',
        total_records=len(by_model),
        records=outputs,
        failures=failures,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Fetch and convert the SMB Bench run table.',
    )
    parser.add_argument(
        '--revision',
        default=DEFAULT_REVISION,
        help=f'Dataset revision to convert (default: {DEFAULT_REVISION}).',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        help=(
            'Replay a local copy of the dataset (holding '
            f'{RUNS_FILE}, {MODELS_FILE} and {PRICES_FILE}) instead of '
            'fetching.'
        ),
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
            'Write rejected models and reasons here. Defaults beside '
            '--output-dir when any model fails.'
        ),
    )
    parser.add_argument(
        '--no-registry-resolve',
        action='store_true',
        help='Skip eval-card-registry ids and keep the source spellings.',
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = registry_mod.Registry(enabled=not args.no_registry_resolve)
    if args.input_dir is not None:
        source = load_source(args.input_dir, args.revision)
    else:
        with tempfile.TemporaryDirectory(prefix='smb-bench-') as scratch:
            source = load_source(
                fetch_source(args.revision, Path(scratch)), args.revision
            )
    result = convert_source(source, args.output_dir, registry=registry)
    paths = save_evaluation_logs(result.records)
    for path in paths:
        print(path)
    print(
        f'Generated {len(paths)} files in {args.output_dir}/ from '
        f'{len(source["runs"])} run rows'
    )
    if result.failures:
        report_path = save_failure_report(
            result,
            args.failure_report or default_failure_report_path(args.output_dir),
        )
        print(f'Failure report: {report_path}')
        result.raise_if_incomplete()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
