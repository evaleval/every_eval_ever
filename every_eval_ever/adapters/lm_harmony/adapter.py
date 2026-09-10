"""Convert the LM-Harmony results matrix into EEE records.

Source
------
``socialfoundations/lm-harmony`` publishes one file, ``notebooks/all_results.json``,
holding the full matrix behind "Train-before-Test Harmonizes Language Model Rankings"
(arXiv:2507.05195). Shape is ``{block: {task: {hf_model_id: score}}}`` over four
blocks -- ``direct_eval``, ``train_before_test`` and a ``_stderr`` counterpart for
each -- 27 tasks and 61 models.

Two protocols, and they are not one measurement
-----------------------------------------------
``direct_eval`` is the ordinary zero-shot lm-evaluation-harness number. Under
``train_before_test`` the model is first fine-tuned on the task's own training split
and then evaluated, which is the paper's contribution and a different quantity. Both
are published and both are worth keeping, so both are converted -- but only
``direct_eval`` carries a canonical global ``metric_id``. The fine-tuned protocol gets
a source-namespaced id, because a consumer joining on ``accuracy`` must not pool a
zero-shot score with a task-trained one.

What is not converted
---------------------
The matrix also carries ``wiki_2025``, ``arxiv_2025`` and ``stackexchange_2025``, the
paper's post-cutoff perplexity corpora scored as ``bits_per_byte`` for 53 of the 61
models. The repository commits no task definition for them, so there is no dataset to
name as provenance, and ``bits_per_byte`` resolves to no registry metric. They are
recorded as exclusions rather than published under an invented dataset.

Run
---
    uv run python -m every_eval_ever.adapters.lm_harmony.adapter \
        --output-dir /tmp/lm-harmony-smoke/data/lm-harmony
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from every_eval_ever.eval_types import (
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
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
    SourceRecordExclusion,
    SourceRecordFailure,
    default_failure_report_path,
    raw_capture,
    save_evaluation_logs,
    save_failure_report,
)

SRC = 'lm_harmony'
COLLECTION = 'lm-harmony'

OWNER_REPO = 'socialfoundations/lm-harmony'
RESULTS_PATH = 'notebooks/all_results.json'
PROJECT_URL = f'https://github.com/{OWNER_REPO}'
PAPER_URL = 'https://arxiv.org/abs/2507.05195'
COMMITS_API = f'https://api.github.com/repos/{OWNER_REPO}/commits/'
RAW_BASE = f'https://raw.githubusercontent.com/{OWNER_REPO}/'

#: The harness LM-Harmony vendors and calls. ``lm_eval/`` in the repository is a
#: fork of v0.4.9, which its README names as the task reference.
EVAL_LIBRARY = EvalLibrary(name='lm-evaluation-harness', version='0.4.9')

#: Hosted eval-card-registry resolver (public HF Space, no auth).
RESOLVER_URL = 'https://evaleval-entity-registry.hf.space/api/v1/resolve'
#: Below this the resolver's answer is recorded but flagged for review.
RESOLVE_CONFIDENCE_FLOOR = 0.9

#: ``simple_evaluate(num_fewshot=0, ...)`` in ``eval_utils.py``.
NUM_FEWSHOT = 0
#: ``gen_kwargs=f'max_gen_toks={self.max_new_tokens}'``, default 256.
MAX_GEN_TOKS = 256
#: The published runs cap the scored split with ``--dataset_param.max_num_test``,
#: a seeded permutation. The README's own command uses 10000, so a split larger
#: than that was scored on a random subsample and the number is not over the full
#: split. Recorded per result rather than assumed away.
MAX_NUM_TEST = 10000

DIRECT = 'direct_eval'
TRAINED = 'train_before_test'
MODES = (DIRECT, TRAINED)

#: Tasks present in the matrix with no committed task definition, so no dataset
#: provenance exists for them. See the module docstring.
UNDEFINED_TASKS = ('wiki_2025', 'arxiv_2025', 'stackexchange_2025')


@dataclass(frozen=True)
class Task:
    """One benchmark as LM-Harmony's vendored harness actually configured it.

    ``lm_eval_dataset_path`` is the string in the task yaml; ``hf_repo`` is what
    the Hub resolves that string to today (11 of the 16 are legacy names that
    redirect, e.g. ``sciq`` -> ``allenai/sciq``). Both are recorded, because the
    first says what was run and the second is the repo a reader can open.
    """

    lm_eval_dataset_path: str
    hf_repo: str
    hf_config: str | None
    split: str
    #: Which yaml key named ``split`` -- ``test_split`` or, absent one,
    #: ``validation_split``. lm-eval scores its "test docs", which for a
    #: benchmark whose real test labels are hidden is the validation rows.
    split_key: str
    metric: str


#: Derived from the ``task:`` field of every yaml under the pinned commit's
#: vendored ``lm_eval/tasks/``, never from a filename -- ``qnli.yaml`` also exists
#: under ``basqueglue/`` and ``social_iqa.yaml`` under ``bigbench/``, and matching
#: by path would have silently converted the wrong dataset.
TASKS: dict[str, Task] = {
    'anli_r1': Task('anli', 'facebook/anli', None, 'test_r1', 'test_split', 'acc'),
    'arc_challenge': Task(
        'allenai/ai2_arc', 'allenai/ai2_arc', 'ARC-Challenge', 'test',
        'test_split', 'acc_norm',
    ),
    'arc_easy': Task(
        'allenai/ai2_arc', 'allenai/ai2_arc', 'ARC-Easy', 'test',
        'test_split', 'acc_norm',
    ),
    'boolq': Task(
        'super_glue', 'aps/super_glue', 'boolq', 'validation',
        'validation_split', 'acc',
    ),
    'cola': Task(
        'glue', 'nyu-mll/glue', 'cola', 'validation', 'validation_split', 'mcc',
    ),
    'commonsense_qa': Task(
        'tau/commonsense_qa', 'tau/commonsense_qa', None, 'validation',
        'validation_split', 'acc',
    ),
    'gsm8k': Task(
        'gsm8k', 'openai/gsm8k', 'main', 'test', 'test_split', 'exact_match',
    ),
    'headqa_en': Task(
        'EleutherAI/headqa', 'EleutherAI/headqa', 'en', 'test', 'test_split',
        'acc_norm',
    ),
    'hellaswag': Task(
        'hellaswag', 'Rowan/hellaswag', None, 'validation', 'validation_split',
        'acc_norm',
    ),
    'mathqa': Task(
        'math_qa', 'allenai/math_qa', None, 'test', 'test_split', 'acc_norm',
    ),
    'medmcqa': Task(
        'medmcqa', 'openlifescienceai/medmcqa', None, 'validation', 'test_split',
        'acc_norm',
    ),
    'mnli': Task(
        'glue', 'nyu-mll/glue', 'mnli', 'validation_matched', 'validation_split',
        'acc',
    ),
    'mrpc': Task(
        'glue', 'nyu-mll/glue', 'mrpc', 'validation', 'validation_split', 'acc',
    ),
    'nq_open': Task(
        'nq_open', 'google-research-datasets/nq_open', None, 'validation',
        'validation_split', 'exact_match',
    ),
    'openbookqa': Task(
        'openbookqa', 'allenai/openbookqa', 'main', 'test', 'test_split',
        'acc_norm',
    ),
    'piqa': Task(
        'baber/piqa', 'baber/piqa', None, 'validation', 'validation_split',
        'acc_norm',
    ),
    'qnli': Task(
        'glue', 'nyu-mll/glue', 'qnli', 'validation', 'validation_split', 'acc',
    ),
    'qqp': Task(
        'glue', 'nyu-mll/glue', 'qqp', 'validation', 'validation_split', 'acc',
    ),
    'rte': Task(
        'glue', 'nyu-mll/glue', 'rte', 'validation', 'validation_split', 'acc',
    ),
    'sciq': Task('sciq', 'allenai/sciq', None, 'test', 'test_split', 'acc_norm'),
    'social_iqa': Task(
        'social_i_qa', 'allenai/social_i_qa', None, 'validation',
        'validation_split', 'acc',
    ),
    'sst2': Task(
        'glue', 'nyu-mll/glue', 'sst2', 'validation', 'validation_split', 'acc',
    ),
    'wic': Task(
        'super_glue', 'aps/super_glue', 'wic', 'validation', 'validation_split',
        'acc',
    ),
    'winogrande': Task(
        'winogrande', 'allenai/winogrande', 'winogrande_xl', 'validation',
        'validation_split', 'acc',
    ),
}


@dataclass(frozen=True)
class Metric:
    """One harness metric, on the scale its own definition fixes.

    ``registry_id`` is ``None`` where the eval-card-registry carries no canonical
    entry for the metric. Every metric this adapter emits does have one, including
    ``acc_norm`` (``normalized-accuracy``), which scores 9 of the 24 tasks --
    folding it into ``accuracy`` would merge two different computations.
    """

    #: lm-eval's own metric key, which is also the dict key in METRICS.
    key: str
    name: str
    registry_id: str | None
    fallback_id: str
    kind: str
    unit: str
    min_score: float
    max_score: float
    lower_is_better: bool = False


METRICS: dict[str, Metric] = {
    'acc': Metric('acc', 'Accuracy', 'accuracy', 'accuracy', 'accuracy', 'proportion', 0.0, 1.0),
    # Length-normalized accuracy, a distinct quantity from `acc`. The registry
    # carries it as `normalized-accuracy` with `acc_norm` already an alias; the
    # hosted resolver misses it only because the live Space lags the seed, so
    # resolve against the seed rather than trusting a no_match from the API.
    'acc_norm': Metric(
        'acc_norm', 'Length-normalized accuracy', 'normalized-accuracy',
        'normalized-accuracy', 'accuracy', 'proportion', 0.0, 1.0,
    ),
    'exact_match': Metric(
        'exact_match', 'Exact match', 'exact-match', 'exact-match', 'exact_match', 'proportion',
        0.0, 1.0,
    ),
    # Matthews correlation is on [-1, 1]. Declaring [0, 1] here would make a
    # legitimate negative score a hard validator error.
    'mcc': Metric(
        'mcc', 'Matthews correlation coefficient', 'matthews-correlation',
        'matthews-correlation', 'correlation', 'coefficient', -1.0, 1.0,
    ),
}

#: lm-eval's own filter key for the metric each task is scored on, as
#: ``notebooks/analyze_results.ipynb`` selects it.
METRIC_FILTER = {
    'acc': 'none',
    'acc_norm': 'none',
    'mcc': 'none',
    'exact_match': 'remove_whitespace',
}
#: gsm8k is the one task the notebook reads under a different filter.
GSM8K_FILTER = 'flexible-extract'


# --------------------------------------------------------------------------- #
# source
# --------------------------------------------------------------------------- #
def resolve_commit(revision: str, *, timeout: float = 30.0) -> str | None:
    """Return the 40-hex commit ``revision`` names, or None if it cannot be read.

    Resolved once, before the results file is fetched, so every URL a record
    cites is immutable. A branch name recorded as provenance would let two runs
    claim the same source and mean different bytes.
    """
    if len(revision) == 40 and all(c in '0123456789abcdef' for c in revision.lower()):
        return revision.lower()
    try:
        resp = requests.get(f'{COMMITS_API}{revision}', timeout=timeout)
        resp.raise_for_status()
        return str(resp.json()['sha'])
    except Exception:  # noqa: BLE001 -- caller decides whether this is fatal
        return None


def results_url(sha: str) -> str:
    return f'{RAW_BASE}{sha}/{RESULTS_PATH}'


def fetch_results(sha: str, *, timeout: float = 60.0) -> dict[str, Any]:
    url = results_url(sha)
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    raw_capture.record(
        url=url, content=resp.content, content_type='application/json',
        label='LM-Harmony all_results.json',
    )
    return payload


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
def resolve_entity(
    raw_value: str, entity_type: str, *, enabled: bool = True, timeout: float = 15.0
) -> tuple[str | None, dict[str, Any]]:
    """Resolve one entity through the hosted registry. Never fatal.

    Returns ``(canonical_id_or_None, provenance)``. The opt-out and any network
    error fall back to ``None`` with the reason recorded, so a sleeping Space
    cannot fail a conversion.
    """
    if not enabled:
        return None, {'resolution': 'offline'}
    try:
        resp = requests.post(
            RESOLVER_URL,
            json={'raw_value': raw_value, 'entity_type': entity_type},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001 -- best effort by design
        return None, {'resolution': 'unreachable', 'resolution_error': str(exc)[:200]}
    return data.get('canonical_id'), {
        'resolution': 'registry',
        'resolution_strategy': data.get('strategy'),
        'resolution_confidence': data.get('confidence'),
        'created_new': data.get('created_new'),
        'review_status': data.get('review_status'),
    }


def needs_review(provenance: dict[str, Any]) -> bool:
    """True when a resolved id is not a confident, already-reviewed canonical."""
    if provenance.get('resolution') == 'unreachable':
        return True
    if provenance.get('created_new'):
        return True
    if provenance.get('review_status') not in (None, 'reviewed'):
        return True
    confidence = provenance.get('resolution_confidence')
    return isinstance(confidence, (int, float)) and confidence < RESOLVE_CONFIDENCE_FLOOR


# --------------------------------------------------------------------------- #
# building records
# --------------------------------------------------------------------------- #
def _stringify(details: dict[str, Any]) -> dict[str, str]:
    """``additional_details`` is ``dict[str, str]`` everywhere in the schema."""
    out: dict[str, str] = {}
    for key, value in details.items():
        if value is None:
            continue
        out[key] = value if isinstance(value, str) else json.dumps(value)
    return out


def metric_id_for(metric: Metric, mode: str) -> str:
    """The cross-source join key.

    ``direct_eval`` takes the canonical global id where the registry has one, so
    a zero-shot accuracy here joins with a zero-shot accuracy anywhere else.
    ``train_before_test`` is namespaced without exception: it is the same metric
    computed after task-specific fine-tuning, and letting it share the global id
    would merge two quantities the paper exists to distinguish.
    """
    if mode == TRAINED:
        return f'{SRC}.{TRAINED}.{metric.fallback_id}'
    return metric.registry_id or metric.fallback_id


def build_metric_config(
    metric: Metric, mode: str, task_name: str, metric_prov: dict[str, Any]
) -> MetricConfig:
    filter_key = GSM8K_FILTER if task_name == 'gsm8k' else METRIC_FILTER[metric.key]
    details: dict[str, Any] = {
        'evaluation_protocol': mode,
        'lm_eval_metric_key': f'{metric.key},{filter_key}',
        'num_fewshot': NUM_FEWSHOT,
    }
    if mode == TRAINED:
        details['protocol_note'] = (
            'Fine-tuned on this task\'s own training split before evaluation '
            '(train-before-test). Not comparable with a zero-shot score, which is '
            'why this result carries a source-namespaced metric_id.'
        )
        details['canonical_metric_id_withheld'] = metric.registry_id or metric.fallback_id
    else:
        details['protocol_note'] = (
            'Zero-shot lm-evaluation-harness evaluation, no task-specific training.'
        )
    if metric.registry_id is None:
        details['metric_id_registry_status'] = 'unresolved'
        details['metric_id_note'] = (
            'The eval-card-registry carries no canonical entry for this metric, so '
            'the id is the harness\'s own metric key. Registering it would let this '
            'result join with other harness-derived sources.'
        )
    else:
        details.update(metric_prov)
    return MetricConfig(
        metric_name=metric.name,
        metric_id=metric_id_for(metric, mode),
        metric_kind=metric.kind,
        metric_unit=metric.unit,
        lower_is_better=metric.lower_is_better,
        score_type=ScoreType.continuous,
        min_score=metric.min_score,
        max_score=metric.max_score,
        additional_details=_stringify(details),
    )


def build_source_data(task_name: str, task: Task) -> SourceDataHf:
    """The dataset the eval ran on, not the results file and not the model."""
    return SourceDataHf(
        dataset_name=task_name,
        source_type='hf_dataset',
        hf_repo=task.hf_repo,
        hf_split=task.split,
        additional_details=_stringify({
            'lm_eval_task': task_name,
            'lm_eval_dataset_path': task.lm_eval_dataset_path,
            'lm_eval_split_key': task.split_key,
            'hf_config': task.hf_config,
            'scored_split_note': (
                'The split lm-eval treats as this task\'s test docs. Where the '
                'benchmark withholds test labels that is its validation split.'
            ),
            'max_num_test_subsample': MAX_NUM_TEST,
            'subsample_note': (
                'The published runs cap the scored split at max_num_test with a '
                'seeded permutation, so a split larger than the cap was scored on '
                'a random subsample of it. The source does not state the resulting '
                'n per task, so num_samples is left unset rather than guessed.'
            ),
        }),
    )


def build_result(
    task_name: str,
    task: Task,
    mode: str,
    score: float,
    stderr: float | None,
    metric_prov: dict[str, Any],
) -> EvaluationResult:
    metric = METRICS[task.metric]
    uncertainty = None
    if stderr is not None:
        uncertainty = Uncertainty(standard_error=StandardError(value=float(stderr)))
    return EvaluationResult(
        # The protocol is part of the result identity, so the two blocks cannot
        # collide inside one model's log.
        evaluation_result_id=f'{SRC}.{task_name}.{mode}',
        # The benchmark, so a consumer can still find every lm-harmony result for
        # a task. Keeping the protocols apart is metric_id's job, above.
        evaluation_name=f'{SRC}.{task_name}',
        source_data=build_source_data(task_name, task),
        metric_config=build_metric_config(metric, mode, task_name, metric_prov),
        score_details=ScoreDetails(
            score=float(score),
            uncertainty=uncertainty,
            details=_stringify({
                'evaluation_protocol': mode,
                'source_block': mode,
                'stderr_source': 'published' if stderr is not None else 'absent',
            }),
        ),
    )


def build_log(
    model_raw: str,
    results: list[EvaluationResult],
    sha: str,
    retrieved_ts: str,
    model_id: str,
    model_prov: dict[str, Any],
) -> EvaluationLog:
    developer, _, model_name = model_raw.partition('/')
    return EvaluationLog(
        schema_version=SCHEMA_VERSION,
        # Keyed on the pinned commit and the RAW source id, never on the resolved
        # canonical id (which the registry may re-map) and never on now.
        evaluation_id=f'{SRC}/{sha}/{model_raw}',
        retrieved_timestamp=retrieved_ts,
        source_metadata=SourceMetadata(
            source_name='LM-Harmony',
            source_type='documentation',
            source_organization_name='Max Planck Institute for Intelligent Systems',
            source_organization_url=PROJECT_URL,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details=_stringify({
                'source_role': 'research_project',
                'source_commit': sha,
                'source_file': RESULTS_PATH,
                'source_file_url': results_url(sha),
                'paper_url': PAPER_URL,
                'paper_title': (
                    'Train-before-Test Harmonizes Language Model Rankings'
                ),
            }),
        ),
        eval_library=EVAL_LIBRARY,
        model_info=ModelInfo(
            name=model_name or model_raw,
            id=model_id,
            developer=developer or 'unknown',
            additional_details=_stringify({
                # Open-weights checkpoints the authors ran locally through
                # lm-eval's HFLM, so the model was served by whoever evaluated it.
                'deployment_type': 'self_deployed',
                'model_availability': 'open_weights',
                'source_model_string': model_raw,
                **model_prov,
            }),
        ),
        evaluation_results=results,
    )


def convert(
    payload: dict[str, Any],
    sha: str,
    retrieved_ts: str,
    output_dir: Path,
    *,
    resolve_enabled: bool = True,
    limit: int | None = None,
) -> tuple[SourceConversionResult, list[str]]:
    """Convert the matrix, keeping every valid cell and reporting the rest."""
    for block in (*MODES, f'{DIRECT}_stderr', f'{TRAINED}_stderr'):
        if block not in payload:
            raise SystemExit(
                f'{RESULTS_PATH} has no {block!r} block; the source layout '
                'changed and the mapping in this adapter no longer describes it.'
            )

    models = sorted({
        model
        for mode in MODES
        for scores in payload[mode].values()
        for model in scores
    })
    if limit is not None:
        models = models[:limit]

    metric_prov: dict[str, dict[str, Any]] = {}
    flagged: list[str] = []
    for key, metric in METRICS.items():
        if metric.registry_id is None:
            continue
        canonical, prov = resolve_entity(
            metric.registry_id, 'metric', enabled=resolve_enabled
        )
        metric_prov[key] = {f'metric_id_{k}': v for k, v in prov.items()}
        if canonical and canonical != metric.registry_id:
            metric_prov[key]['metric_id_registry_canonical'] = canonical
        if resolve_enabled and needs_review(prov):
            flagged.append(f'metric {metric.registry_id}')

    records: list[tuple[EvaluationLog, str, str]] = []
    failures: list[SourceRecordFailure] = []
    exclusions: list[SourceRecordExclusion] = []
    total = 0

    for task_name in sorted(UNDEFINED_TASKS):
        if task_name not in payload[DIRECT]:
            continue
        covered = len(payload[DIRECT][task_name])
        total += covered
        exclusions.append(SourceRecordExclusion(
            source_ref=f'task={task_name}',
            reason=(
                'no task definition is committed at this revision, so the dataset '
                'the scores cover cannot be named, and its bits_per_byte metric '
                'resolves to no registry entry'
            ),
            source_record={'task': task_name, 'models_covered': covered},
        ))

    unknown = sorted(set(payload[DIRECT]) - set(TASKS) - set(UNDEFINED_TASKS))
    for task_name in unknown:
        covered = len(payload[DIRECT][task_name])
        total += covered
        failures.append(SourceRecordFailure(
            source_ref=f'task={task_name}',
            reason=(
                f'task {task_name!r} is in the source matrix but not in this '
                'adapter\'s task table, so its dataset, split and metric are '
                'unknown; add it deliberately rather than converting it blind'
            ),
            source_record={'task': task_name, 'models_covered': covered},
        ))

    for model_raw in models:
        model_id, prov = resolve_entity(model_raw, 'model', enabled=resolve_enabled)
        if resolve_enabled and needs_review(prov):
            flagged.append(f'model {model_raw}')
        model_prov = {f'model_id_{k}': v for k, v in prov.items()}

        results: list[EvaluationResult] = []
        for task_name, task in sorted(TASKS.items()):
            for mode in MODES:
                scores = payload[mode].get(task_name) or {}
                if model_raw not in scores:
                    continue
                total += 1
                raw_score = scores[model_raw]
                stderr = (payload[f'{mode}_stderr'].get(task_name) or {}).get(model_raw)
                try:
                    results.append(build_result(
                        task_name, task, mode, raw_score, stderr,
                        metric_prov.get(task.metric, {}),
                    ))
                except Exception as exc:  # noqa: BLE001 -- one cell must not kill the run
                    failures.append(SourceRecordFailure(
                        source_ref=f'{model_raw} {task_name} {mode}',
                        reason=str(exc),
                        source_record={
                            'model': model_raw, 'task': task_name,
                            'block': mode, 'score': raw_score, 'stderr': stderr,
                        },
                    ))
        if not results:
            failures.append(SourceRecordFailure(
                source_ref=f'model={model_raw}',
                reason='no cell in the matrix converted to a valid result',
                source_record={'model': model_raw},
            ))
            continue
        developer, _, model_name = model_raw.partition('/')
        log = build_log(
            model_raw, results, sha, retrieved_ts,
            model_id or model_raw, model_prov,
        )
        records.append((log, developer, model_name or model_raw))

    outputs = [
        EvaluationLogOutput(
            eval_log=EvaluationLog.model_validate(log.model_dump()),
            base_dir=output_dir,
            developer=developer,
            model_name=model_name,
        )
        for log, developer, model_name in records
    ]
    result = SourceConversionResult(
        source_name='LM-Harmony',
        total_records=total,
        records=outputs,
        failures=failures,
        exclusions=exclusions,
    )
    return result, flagged


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--output-dir', type=Path, default=Path(f'/tmp/{COLLECTION}-smoke/data/{COLLECTION}'),
        help='Collection directory to write into, i.e. <root>/data/lm-harmony. '
             'Defaults outside any checkout: generated records belong in the '
             'datastore PR, not in this repository.',
    )
    ap.add_argument(
        '--revision', default='main',
        help='Branch, tag or commit of socialfoundations/lm-harmony to convert. '
             'Resolved to a commit sha before anything is fetched.',
    )
    ap.add_argument(
        '--allow-unpinned-source', action='store_true',
        help='Proceed when the revision cannot be resolved to a commit sha. '
             'Without it an unresolvable revision stops the run rather than '
             'publishing records that cite a moving reference.',
    )
    ap.add_argument(
        '--input-json', type=Path, default=None,
        help='Replay a saved all_results.json instead of fetching. Requires '
             '--revision to name the commit those bytes came from.',
    )
    ap.add_argument(
        '--save-raw-json', type=Path, default=None,
        help='Write the fetched payload here, for offline replay.',
    )
    ap.add_argument('--limit', type=int, default=None, help='Convert only the first N models.')
    ap.add_argument(
        '--no-registry-resolve', action='store_true',
        help='Skip the eval-card-registry lookups. model_info.id becomes the raw '
             'source id and metric ids stay as declared, both marked unresolved.',
    )
    ap.add_argument(
        '--replace-existing', action='store_true',
        help='Replace records already present in the output directory. Record '
             'filenames are fresh uuid4s, so without this a populated directory '
             'is an error rather than a second copy of every evaluation.',
    )
    ap.add_argument(
        '--failure-report', type=Path, default=None,
        help='Where to write the conversion report (default: adapter_reports/ '
             'beside the output directory).',
    )
    ap.add_argument(
        '--emit-source-version', action='store_true',
        help='Print the resolved source commit and exit without converting. The '
             'scheduler uses this to skip a run whose source has not moved.',
    )
    return ap.parse_args(argv)


def existing_records(output_dir: Path, routes: list[tuple[str, str]]) -> list[Path]:
    return sorted(
        path
        for developer, model in routes
        for path in Path(output_dir).joinpath(developer, model).glob('*.json')
    )


def run(args: argparse.Namespace) -> list[Path]:
    sha = resolve_commit(args.revision)
    if sha is None:
        if not args.allow_unpinned_source:
            raise SystemExit(
                f'could not resolve {args.revision!r} to a commit sha in '
                f'{OWNER_REPO}. Every record cites its source commit, so a '
                'moving reference is refused. Pass --allow-unpinned-source to '
                'proceed with the revision string as given.'
            )
        sha = args.revision

    if args.emit_source_version:
        print(sha)
        return []

    if args.input_json is not None:
        if args.revision == 'main':
            raise SystemExit(
                '--input-json needs --revision to name the commit those bytes '
                'came from, otherwise the records would cite whatever main '
                'happens to be now rather than the source that produced them.'
            )
        payload = json.loads(Path(args.input_json).read_text())
    else:
        payload = fetch_results(sha)
        if args.save_raw_json:
            Path(args.save_raw_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_raw_json).write_text(json.dumps(payload))

    retrieved_ts = str(time.time())
    output_dir = Path(args.output_dir)
    if output_dir.name != COLLECTION:
        raise SystemExit(
            f'--output-dir must end in {COLLECTION!r}, the collection directory '
            f'records are published into; got {output_dir}. Otherwise the run '
            'reports one destination and writes to another.'
        )

    result, flagged = convert(
        payload, sha, retrieved_ts, output_dir,
        resolve_enabled=not args.no_registry_resolve,
        limit=args.limit,
    )

    report = save_failure_report(
        result, args.failure_report or default_failure_report_path(output_dir)
    )
    print(
        f'Conversion accounting: {report} '
        f'({len(result.failures)} unconverted, {len(result.exclusions)} excluded)'
    )

    routes = [(o.developer, o.model_name) for o in result.records]
    stale = existing_records(output_dir, routes)
    if stale and not args.replace_existing:
        raise SystemExit(
            f'{len(stale)} record(s) already exist under {output_dir}, e.g. '
            f'{stale[0]}. Filenames are fresh uuid4s, so writing now would add a '
            'second copy of every evaluation_id. Pass --replace-existing.'
        )
    if stale and args.replace_existing:
        written = save_evaluation_logs(result.records)
        for path in stale:
            path.unlink()
    else:
        written = save_evaluation_logs(result.records)

    n_results = sum(len(o.eval_log.evaluation_results) for o in result.records)
    print(
        f'Coverage: {result.total_records} source cell(s) -> {len(written)} record(s) '
        f'carrying {n_results} result(s); {len(result.failures)} dropped, '
        f'{len(result.exclusions)} excluded -> {output_dir}'
    )
    if flagged:
        print(
            f'  {len(flagged)} id(s) are not confident reviewed canonicals and need '
            f'a registry follow-up: {", ".join(sorted(set(flagged))[:8])}'
            + (' ...' if len(set(flagged)) > 8 else ''),
            file=sys.stderr,
        )
    result.raise_if_incomplete()
    return written


def main() -> None:
    run(parse_args())


if __name__ == '__main__':
    main()
