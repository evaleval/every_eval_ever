#!/usr/bin/env python3
"""Convert MTEB embedding-benchmark results into Every Eval Ever aggregate logs.

Data source: the ``embeddings-benchmark/results`` git repository, which is the
dump MTEB's own leaderboard is built from. Layout::

    results/<org>__<model>/<model_revision>/<Task>.json
    results/<org>__<model>/<model_revision>/model_meta.json

Read from that repository rather than from its ``mteb/results`` Hugging Face
parquet mirror: the mirror flattens each score to one ``(model, task, split,
subset, language)`` row and drops the harness version, the evaluation dataset
revision, and every named sub-metric, all of which this schema has a field for.

Scope is the two headline suites, ``MTEB(eng, v2)`` and ``MTEB(Multilingual,
v2)`` -- 151 tasks, the results people cite. The full dump is ~8.8M scores of
which one task (``FloresBitextMining``, one row per language *pair*) is 85%, so
"all of MTEB" is a re-hosting decision rather than a coverage one and is
deliberately not what this adapter publishes.

One ``EvaluationLog`` per evaluated ``(model, model_revision)``, one
``EvaluationResult`` per ``(task, split, hf_subset)`` -- the grain the source
reports at.

Run from the EEE repo dir::

    uv run python -m every_eval_ever.adapters.mteb.adapter --output-dir data/mteb
    uv run python -m every_eval_ever validate 'data/mteb/*/*/*.json'
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

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
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    default_failure_report_path,
    raw_capture,
    require_finite_number,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.eval_card_registry import Registry
from every_eval_ever.helpers.io import datastore_path_components

SRC = 'mteb'
COLLECTION = 'mteb'

RESULTS_REPO = 'embeddings-benchmark/results'
RESULTS_REPO_URL = f'https://github.com/{RESULTS_REPO}'
#: The commit this adapter reads by default. Pinned rather than tracking
#: ``main`` so a re-run converts the same bytes; ``--ref`` overrides it and the
#: scheduled run passes the current head.
SOURCE_COMMIT = 'fe1eb0571d7c1c261d3cb892c558477451ba5232'
TARBALL = 'https://codeload.github.com/{repo}/tar.gz/{ref}'
COMMITS_API = 'https://api.github.com/repos/{repo}/commits/{ref}'

#: MTEB is an open benchmark run locally against downloaded weights, so every
#: published run is self-deployed. ``model_availability`` is **not** inferred
#: the same way -- it comes from each model's own ``model_meta.open_weights``.
DEPLOYMENT_TYPE = 'self_deployed'

MODEL_META = 'model_meta.json'

#: Tolerance when re-checking a task's declared ``main_score`` against the named
#: sub-metric in the score dump. The dump stores both as rounded decimals, so an
#: exact comparison would reject rows that agree.
MAIN_SCORE_TOLERANCE = 1e-6

#: Subsets above which a task is reporting per-language(-pair) cells rather than
#: a task score, and is left out by default.
#:
#: Measured over the two scoped suites: 141 of 151 tasks hold 573 scores between
#: them (median subsets per task: 1), while 10 tasks hold 46,495 -- 41,412 of
#: them ``FloresBitextMining`` alone, one cell per ordered language pair. Kept
#: in, one model's record is ~70MB and 88% of it is that single task; the
#: benchmark's own leaderboard shows a mean over those cells, which this dump
#: does not carry and which an adapter computing it would be inventing. The
#: default sits in the observed gap (112 -> 197 subsets), so it is a boundary in
#: the data rather than a round number. ``0`` disables the limit; every task it
#: skips is reported as an exclusion naming its subset count.
DEFAULT_MAX_SUBSETS_PER_TASK = 150


@dataclass(frozen=True)
class MetricSpec:
    """How one MTEB ``main_score`` is published.

    ``registry_id`` is set only where the eval-card-registry already carries the
    canonical metric; otherwise ``proposed_id`` is published and the record says
    so, so a consumer can tell a joined metric from one still awaiting an id.
    See ``README.md`` -- the proposals are listed there for registration.
    """

    kind: str
    unit: str
    min_score: float
    max_score: float
    registry_id: Optional[str] = None
    proposed_id: Optional[str] = None
    lower_is_better: bool = False

    @property
    def metric_id(self) -> str:
        return self.registry_id or self.proposed_id or ''

    @property
    def registry_backed(self) -> bool:
        return self.registry_id is not None


#: MTEB ``main_score`` name -> how to publish it. Bounds are the metric's own
#: range, not the observed spread: a rank/retrieval proportion is 0-1 and a
#: correlation runs -1 to 1. Every metric here is higher-is-better.
#:
#: Registry ids confirmed present in the vendored eval-card-registry snapshot:
#: ``accuracy``, ``f1``, ``ndcg-at-10``, ``recall-at-10``. The rest are proposed
#: (see README) and published as such rather than silently namespaced, because a
#: made-up global id fragments the one join the datastore exists for.
METRICS: dict[str, MetricSpec] = {
    'accuracy': MetricSpec(
        kind='accuracy',
        unit='proportion',
        min_score=0.0,
        max_score=1.0,
        registry_id='accuracy',
    ),
    'f1': MetricSpec(
        kind='f1',
        unit='proportion',
        min_score=0.0,
        max_score=1.0,
        registry_id='f1',
    ),
    'ndcg_at_10': MetricSpec(
        kind='ndcg',
        unit='proportion',
        min_score=0.0,
        max_score=1.0,
        registry_id='ndcg-at-10',
    ),
    'recall_at_10': MetricSpec(
        kind='recall',
        unit='proportion',
        min_score=0.0,
        max_score=1.0,
        registry_id='recall-at-10',
    ),
    'ndcg_at_1': MetricSpec(
        kind='ndcg',
        unit='proportion',
        min_score=0.0,
        max_score=1.0,
        proposed_id='ndcg-at-1',
    ),
    'map_at_1000': MetricSpec(
        kind='map',
        unit='proportion',
        min_score=0.0,
        max_score=1.0,
        proposed_id='map-at-1000',
    ),
    'mrr_at_10': MetricSpec(
        kind='mrr',
        unit='proportion',
        min_score=0.0,
        max_score=1.0,
        proposed_id='mrr-at-10',
    ),
    # The cutoff-free spellings older mteb wrote for reranking tasks. Published
    # under their own names rather than folded into the cutoff-suffixed ones --
    # see LEGACY_MAIN_SCORE_SPELLINGS.
    'map': MetricSpec(
        kind='map',
        unit='proportion',
        min_score=0.0,
        max_score=1.0,
        registry_id='map',
    ),
    'mrr': MetricSpec(
        kind='mrr',
        unit='proportion',
        min_score=0.0,
        max_score=1.0,
        registry_id='mrr',
    ),
    'v_measure': MetricSpec(
        kind='v_measure',
        unit='proportion',
        min_score=0.0,
        max_score=1.0,
        proposed_id='v-measure',
    ),
    'max_ap': MetricSpec(
        kind='average_precision',
        unit='proportion',
        min_score=0.0,
        max_score=1.0,
        proposed_id='max-ap',
    ),
    'cosine_spearman': MetricSpec(
        kind='spearman',
        unit='correlation',
        min_score=-1.0,
        max_score=1.0,
        proposed_id='cosine-spearman',
    ),
    # A signed difference between two MRR values, so it is bounded by the same
    # range as the correlation metrics rather than by a proportion's.
    'p-MRR': MetricSpec(
        kind='mrr_delta',
        unit='delta',
        min_score=-1.0,
        max_score=1.0,
        proposed_id='p-mrr',
    ),
    # Specific to MTEB's multi-subquery reranking protocol, so namespaced by
    # design rather than for want of a canonical id.
    'max_over_subqueries_map_at_1000': MetricSpec(
        kind='map',
        unit='proportion',
        min_score=0.0,
        max_score=1.0,
        proposed_id='mteb.max-over-subqueries-map-at-1000',
    ),
}


#: Reranking metrics mteb renamed. A results file written before the rename
#: carries the score under the older key, and the two spellings never appear in
#: one entry, so there is no run in which they can be checked against each
#: other. Rather than assert ``map == map_at_1000`` on that absence, a row whose
#: declared name is missing is published under **the name its own file uses**,
#: with the declared name recorded beside it. Both legacy spellings are
#: registry canonicals, so this costs no join.
#:
#: ``max_over_subqueries_map_at_1000`` is deliberately absent: older files spell
#: it ``map`` too, but that is also the plain metric's name, and publishing a
#: multi-subquery protocol's score as ordinary MAP would merge two different
#: quantities. Those rows fail instead.
LEGACY_MAIN_SCORE_SPELLINGS: dict[str, tuple[str, ...]] = {
    'map_at_1000': ('map',),
    'mrr_at_10': ('mrr',),
}


def effective_main_score(
    entry: dict[str, Any], declared: str
) -> Optional[tuple[str, MetricSpec]]:
    """Return the metric name this entry actually carries, and how to publish it.

    ``None`` when neither the declared name nor a known older spelling of it is
    present, which is the caller's signal to fail the row rather than guess.
    """
    candidates = (declared, *LEGACY_MAIN_SCORE_SPELLINGS.get(declared, ()))
    for name in candidates:
        if name in entry and name in METRICS:
            return name, METRICS[name]
    return None


def load_task_metadata() -> dict[str, Any]:
    """Return the vendored MTEB task/suite snapshot."""
    resource = resources.files('every_eval_ever.adapters.mteb').joinpath(
        'task_metadata.json'
    )
    return json.loads(resource.read_text(encoding='utf-8'))


def stringify(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(',', ':'))
    return str(value)


def clean_details(details: dict[str, Any]) -> dict[str, str]:
    return {k: stringify(v) for k, v in details.items() if v is not None}


def model_repo_from_dir(directory: str) -> str:
    """``intfloat__multilingual-e5-small`` -> ``intfloat/multilingual-e5-small``.

    The dump encodes the namespace separator as a double underscore. A name with
    no separator is a model published without a namespace and is returned as-is.
    """
    return directory.replace('__', '/', 1) if '__' in directory else directory


# -- source acquisition ----------------------------------------------------


def resolve_ref(ref: str, timeout: int = 30) -> str:
    """Resolve a ref to its commit sha, so a run names the bytes it read."""
    url = COMMITS_API.format(repo=RESULTS_REPO, ref=ref)
    request = urllib.request.Request(
        url, headers={'User-Agent': 'every-eval-ever-mteb-adapter'}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)['sha']


def download_results(ref: str, destination: Path, timeout: int = 900) -> Path:
    """Fetch the results repository at ``ref`` and return the extracted root.

    One tarball rather than a request per file: the dump holds hundreds of
    thousands of small JSONs, and a per-file fetch would neither finish in a
    scheduled run's budget nor leave a manifest anyone could check.
    """
    url = TARBALL.format(repo=RESULTS_REPO, ref=ref)
    archive = destination / 'results.tar.gz'
    request = urllib.request.Request(
        url, headers={'User-Agent': 'every-eval-ever-mteb-adapter'}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        with archive.open('wb') as handle:
            shutil.copyfileobj(response, handle)
    extracted = destination / 'extracted'
    extracted.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        _safe_extract(tar, extracted)
    archive.unlink(missing_ok=True)
    roots = [p for p in extracted.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f'unexpected archive layout: {roots}')
    return roots[0] / 'results'


def _safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    """Extract without letting a member escape ``destination``."""
    base = destination.resolve()
    for member in tar.getmembers():
        target = (base / member.name).resolve()
        if not str(target).startswith(str(base)):
            raise RuntimeError(
                f'archive member escapes destination: {member.name}'
            )
    # filter='data' is the hardened extractor; it is the default from 3.14 and
    # this package targets >=3.12, where it must be asked for.
    tar.extractall(destination, filter='data')


# -- reading the dump ------------------------------------------------------


@dataclass(frozen=True)
class RunDirectory:
    """One evaluated ``(model, revision)`` in the dump."""

    model_repo: str
    revision: str
    path: Path


def iter_runs(results_root: Path) -> Iterator[RunDirectory]:
    for model_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        repo = model_repo_from_dir(model_dir.name)
        for revision_dir in sorted(
            p for p in model_dir.iterdir() if p.is_dir()
        ):
            yield RunDirectory(repo, revision_dir.name, revision_dir)


def read_json(path: Path) -> Any:
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def named_submetric(
    entry: dict[str, Any], main_score_name: str
) -> Optional[float]:
    """Return the named sub-metric a task's ``main_score`` should equal.

    The dump carries both, so the declared metric name can be checked against
    the number being published instead of trusted. A results file written by an
    older ``mteb`` than the vendored snapshot is exactly where that disagrees.
    """
    value = entry.get(main_score_name)
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


# -- building records ------------------------------------------------------


def make_result(
    task_name: str,
    task_meta: dict[str, Any],
    split: str,
    entry: dict[str, Any],
    *,
    suites: tuple[str, ...],
    training_datasets: frozenset[str],
) -> EvaluationResult:
    """Build one result from a single ``(task, split, hf_subset)`` score.

    Raises ``ValueError`` for a row that cannot be represented; the caller turns
    that into a ``SourceRecordFailure`` rather than dropping it silently.
    """
    declared_name = task_meta['main_score']
    if declared_name not in METRICS:
        raise ValueError(f'no metric mapping for main_score {declared_name!r}')

    resolved = effective_main_score(entry, declared_name)
    if resolved is None:
        raise ValueError(
            f'{task_name}.{split}: dump carries no {declared_name!r} (nor a '
            'known older spelling of it) to check main_score against'
        )
    main_score_name, spec = resolved

    score = entry.get('main_score')
    score = require_finite_number(score, f'{task_name}.{split}.main_score')
    if not spec.min_score <= score <= spec.max_score:
        raise ValueError(
            f'{task_name}.{split}: score {score} outside '
            f'[{spec.min_score}, {spec.max_score}] declared for {main_score_name}'
        )

    named = named_submetric(entry, main_score_name)
    if named is None:
        raise ValueError(
            f'{task_name}.{split}: {main_score_name!r} is not a number'
        )
    if abs(named - score) > MAIN_SCORE_TOLERANCE:
        raise ValueError(
            f'{task_name}.{split}: main_score {score} does not match '
            f'{main_score_name}={named}; the snapshot names a different metric '
            'than this results file was written with'
        )

    subset = entry.get('hf_subset') or 'default'
    evaluation_name = f'{SRC}.{task_name}.{split}.{subset}'
    languages = entry.get('languages') or []

    return EvaluationResult(
        evaluation_result_id=evaluation_name,
        evaluation_name=evaluation_name,
        source_data=SourceDataHf(
            dataset_name=task_name,
            source_type='hf_dataset',
            hf_repo=task_meta['hf_repo'],
            hf_split=split,
            additional_details=clean_details(
                {
                    'dataset_revision': task_meta.get('dataset_revision'),
                    'hf_subset': subset,
                    'languages': languages or None,
                    'dataset_license': task_meta.get('license'),
                    'task_reference': task_meta.get('reference'),
                    'task_domains': task_meta.get('domains') or None,
                }
            ),
        ),
        metric_config=MetricConfig(
            evaluation_description=(
                f'MTEB {task_meta["type"]} task {task_name}; '
                f'{main_score_name} on split {split}, subset {subset}'
            ),
            metric_name=main_score_name,
            metric_kind=spec.kind,
            metric_unit=spec.unit,
            metric_id=spec.metric_id,
            lower_is_better=spec.lower_is_better,
            score_type=ScoreType.continuous,
            min_score=spec.min_score,
            max_score=spec.max_score,
            additional_details=clean_details(
                {
                    'mteb_task_type': task_meta['type'],
                    'mteb_suites': list(suites),
                    # Present only when this file predates a metric rename: the
                    # name published above is the file's, this is the name the
                    # current mteb declares for the task.
                    'mteb_declared_main_score': (
                        declared_name
                        if declared_name != main_score_name
                        else None
                    ),
                    # A metric id not yet in the registry is published with the
                    # fact attached, so a consumer joining on metric_id can tell
                    # a canonical id from one awaiting review.
                    'metric_id_status': (
                        'eval_card_registry'
                        if spec.registry_backed
                        else 'proposed'
                    ),
                    # MTEB records which datasets a model declares it trained
                    # on. Where that includes this task, the score is not a
                    # held-out measurement, and saying so is the source's own
                    # statement rather than an inference.
                    'model_trained_on_this_dataset': task_name
                    in training_datasets,
                }
            ),
        ),
        score_details=ScoreDetails(score=score),
    )


def make_log(
    run: RunDirectory,
    model_meta: dict[str, Any],
    task_files: Iterable[tuple[str, dict[str, Any]]],
    task_metadata: dict[str, Any],
    suites_by_task: dict[str, tuple[str, ...]],
    retrieved_ts: str,
    *,
    source_commit: str,
    model_id: Optional[str] = None,
    resolution_details: Optional[dict[str, Any]] = None,
    max_subsets_per_task: int = DEFAULT_MAX_SUBSETS_PER_TASK,
) -> tuple[Optional[EvaluationLog], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build one log for a ``(model, revision)``; report drops and skips.

    Returns the log, the rows that could not be converted, and the tasks left
    out by ``max_subsets_per_task``. Pure and offline: it takes already-read
    JSON so the unit tests exercise the same code path the live run does.
    """
    results: list[EvaluationResult] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    library_versions: set[str] = set()

    training_datasets = frozenset(
        str(name) for name in (model_meta.get('training_datasets') or [])
    )

    for task_name, payload in task_files:
        task_meta = task_metadata['tasks'].get(task_name)
        if task_meta is None:
            continue
        scores = payload.get('scores') or {}
        subset_count = sum(len(entries or []) for entries in scores.values())
        if max_subsets_per_task and subset_count > max_subsets_per_task:
            skipped.append(
                {
                    'model': run.model_repo,
                    'model_revision': run.revision,
                    'task': task_name,
                    'subsets': subset_count,
                    'reason': (
                        f'{task_name} reports {subset_count} per-subset cells, '
                        f'over the {max_subsets_per_task} limit; pass '
                        '--max-subsets-per-task 0 to include it'
                    ),
                }
            )
            continue
        version = payload.get('mteb_version')
        if version:
            library_versions.add(str(version))
        for split, entries in sorted(scores.items()):
            for entry in entries or []:
                if not isinstance(entry, dict):
                    continue
                try:
                    results.append(
                        make_result(
                            task_name,
                            task_meta,
                            split,
                            entry,
                            suites=suites_by_task.get(task_name, ()),
                            training_datasets=training_datasets,
                        )
                    )
                except (ValueError, TypeError) as exc:
                    failures.append(
                        {
                            'model': run.model_repo,
                            'model_revision': run.revision,
                            'task': task_name,
                            'split': split,
                            'hf_subset': entry.get('hf_subset'),
                            'reason': str(exc),
                        }
                    )

    if not results:
        return None, failures, skipped

    results.sort(key=lambda r: r.evaluation_name)

    open_weights = model_meta.get('open_weights')
    model_details = clean_details(
        {
            'deployment_type': DEPLOYMENT_TYPE,
            'model_availability': (
                None
                if open_weights is None
                else ('open_weights' if open_weights else 'closed_weights')
            ),
            'model_revision': run.revision,
            'n_parameters': model_meta.get('n_parameters'),
            'n_embedding_parameters': model_meta.get('n_embedding_parameters'),
            'embed_dim': model_meta.get('embed_dim'),
            'max_tokens': model_meta.get('max_tokens'),
            'memory_usage_mb': model_meta.get('memory_usage_mb'),
            'similarity_fn_name': model_meta.get('similarity_fn_name'),
            'model_license': model_meta.get('license'),
            'release_date': model_meta.get('release_date'),
            'framework': model_meta.get('framework'),
            'model_type': model_meta.get('model_type'),
            'modalities': model_meta.get('modalities'),
            'adapted_from': model_meta.get('adapted_from'),
            'use_instructions': model_meta.get('use_instructions'),
            'reference': model_meta.get('reference'),
            'public_training_code': model_meta.get('public_training_code'),
            'public_training_data': model_meta.get('public_training_data'),
        }
    )
    if resolution_details:
        model_details.update(clean_details(resolution_details))
    resolved_id = model_id or run.model_repo
    if resolved_id != run.model_repo:
        model_details['source_model_repo'] = run.model_repo

    developer = (
        run.model_repo.split('/', 1)[0] if '/' in run.model_repo else 'unknown'
    )

    # Several mteb versions may have written one model's task files. Name the
    # one that wrote the most of them and keep the full set visible, rather than
    # asserting a single version the run did not have.
    library_version = (
        sorted(library_versions)[-1] if library_versions else 'unknown'
    )

    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        # Keyed on the raw source identity -- the dump's own model directory and
        # revision -- so re-ingesting the same run is idempotent even if the
        # registry later remaps the canonical id. The dump records how long each
        # task took but never when it ran, so there is no run timestamp to fold
        # in and none is invented.
        evaluation_id=f'{SRC}/{run.model_repo}/{run.revision}',
        retrieved_timestamp=retrieved_ts,
        source_metadata=SourceMetadata(
            source_name='MTEB',
            # Harness output files, not reported numbers copied off a
            # leaderboard: these are the dump the leaderboard itself reads.
            source_type='evaluation_run',
            source_organization_name='embeddings-benchmark',
            source_organization_url=RESULTS_REPO_URL,
            # Results are contributed by model authors and third parties alike
            # and the dump does not record which, so the relationship to the
            # model developer is genuinely not stated per run.
            evaluator_relationship=EvaluatorRelationship.other,
            additional_details=clean_details(
                {
                    'source_repo': RESULTS_REPO,
                    'source_commit': source_commit,
                    'evaluator_relationship_note': (
                        'the results dump does not record who ran each '
                        'evaluation'
                    ),
                    'mteb_versions_in_run': sorted(library_versions) or None,
                    'task_metadata_snapshot_mteb_version': task_metadata[
                        '_meta'
                    ].get('mteb_version'),
                }
            ),
        ),
        eval_library=EvalLibrary(name='mteb', version=library_version),
        model_info=ModelInfo(
            name=run.model_repo,
            id=resolved_id,
            developer=developer,
            additional_details=model_details,
        ),
        evaluation_results=results,
    )
    return log, failures, skipped


# -- conversion ------------------------------------------------------------


def suites_index(task_metadata: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    index: dict[str, list[str]] = {}
    for suite, names in task_metadata['suites'].items():
        for name in names:
            index.setdefault(name, []).append(suite)
    return {name: tuple(sorted(v)) for name, v in index.items()}


def convert(
    results_root: Path,
    out_root: Path,
    *,
    source_commit: str,
    registry: Registry,
    limit: Optional[int] = None,
    models: Optional[set[str]] = None,
    max_subsets_per_task: int = DEFAULT_MAX_SUBSETS_PER_TASK,
) -> SourceConversionResult:
    """Convert every in-scope run, accounting for what could not be published."""
    task_metadata = load_task_metadata()
    in_scope = set(task_metadata['tasks'])
    suites_by_task = suites_index(task_metadata)
    retrieved_ts = str(time.time())

    outputs: list[EvaluationLogOutput] = []
    failures: list[SourceRecordFailure] = []
    exclusions: list[SourceRecordExclusion] = []
    converted = 0
    seen_runs = 0

    for run in iter_runs(results_root):
        if models is not None and run.model_repo not in models:
            continue
        if limit is not None and converted >= limit:
            break
        seen_runs += 1
        run_ref = f'{run.model_repo}@{run.revision}'

        meta_path = run.path / MODEL_META
        if not meta_path.is_file():
            exclusions.append(
                SourceRecordExclusion(
                    source_ref=run_ref,
                    reason=f'no {MODEL_META} in the run directory',
                )
            )
            continue
        try:
            model_meta = read_json(meta_path)
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=run_ref,
                    reason=f'unreadable {MODEL_META}: {exc}',
                )
            )
            continue

        task_files: list[tuple[str, dict[str, Any]]] = []
        for path in sorted(run.path.glob('*.json')):
            task_name = path.stem
            if task_name not in in_scope:
                continue
            try:
                task_files.append((task_name, read_json(path)))
            except (OSError, json.JSONDecodeError) as exc:
                failures.append(
                    SourceRecordFailure(
                        source_ref=f'{run_ref}/{task_name}',
                        reason=f'unreadable task result: {exc}',
                    )
                )
        if not task_files:
            exclusions.append(
                SourceRecordExclusion(
                    source_ref=run_ref,
                    reason='no results for any task in the two scoped suites',
                )
            )
            continue

        resolution = registry.org(run.model_repo.split('/', 1)[0])
        resolution_details = resolution.provenance('developer')

        log, row_failures, row_skips = make_log(
            run,
            model_meta,
            task_files,
            task_metadata,
            suites_by_task,
            retrieved_ts,
            source_commit=source_commit,
            resolution_details=resolution_details,
            max_subsets_per_task=max_subsets_per_task,
        )
        for skip in row_skips:
            exclusions.append(
                SourceRecordExclusion(
                    source_ref=f'{run_ref}/{skip["task"]}',
                    reason=skip['reason'],
                    source_record=skip,
                )
            )
        for failure in row_failures:
            failures.append(
                SourceRecordFailure(
                    source_ref=(
                        f'{run_ref}/{failure["task"]}/{failure["split"]}'
                        f'/{failure.get("hf_subset") or "default"}'
                    ),
                    reason=failure['reason'],
                    source_record=failure,
                )
            )
        if log is None:
            exclusions.append(
                SourceRecordExclusion(
                    source_ref=run_ref,
                    reason='every scoped score for this run was unconvertible',
                )
            )
            continue

        _, route_developer, route_model = datastore_path_components(
            COLLECTION, log.model_info.id, log.model_info.developer
        )
        outputs.append(
            EvaluationLogOutput(
                eval_log=EvaluationLog.model_validate(log.model_dump()),
                # ``out_root`` is the collection directory itself
                # (``data/mteb``), matching output_scope='collection' in the
                # catalog and the convention hle/ uses.
                base_dir=out_root,
                developer=route_developer,
                model_name=route_model,
            )
        )
        converted += 1

    return SourceConversionResult(
        source_name=f'{RESULTS_REPO}@{source_commit}',
        total_records=seen_runs,
        records=outputs,
        failures=failures,
        exclusions=exclusions,
    )


# -- CLI -------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir',
        default='.',
        help=(
            'the collection directory to write into, i.e. data/mteb; records '
            'land in <output-dir>/<developer>/<model>/<uuid>.json'
        ),
    )
    parser.add_argument(
        '--ref',
        default=SOURCE_COMMIT,
        help=f'{RESULTS_REPO} ref to convert (default: the pinned commit)',
    )
    parser.add_argument(
        '--results-dir',
        default=None,
        help=(
            'convert an already-extracted results tree instead of downloading; '
            'the directory holding the <org>__<model>/ run folders'
        ),
    )
    parser.add_argument(
        '--limit', type=int, default=None, help='convert at most N runs'
    )
    parser.add_argument(
        '--max-subsets-per-task',
        type=int,
        default=DEFAULT_MAX_SUBSETS_PER_TASK,
        help=(
            'leave out a task reporting more than N per-subset cells '
            f'(default {DEFAULT_MAX_SUBSETS_PER_TASK}; 0 disables the limit). '
            'Each skipped task is reported as an exclusion with its count.'
        ),
    )
    parser.add_argument(
        '--models',
        default=None,
        help='comma-separated developer/model ids to restrict the run to',
    )
    parser.add_argument(
        '--no-registry-resolve',
        action='store_true',
        help='skip eval-card-registry resolution entirely',
    )
    parser.add_argument(
        '--registry-live',
        action='store_true',
        help='consult the registry for values the vendored snapshot cannot place',
    )
    parser.add_argument(
        '--emit-source-version',
        action='store_true',
        help='print the resolved source commit and exit without converting',
    )
    parser.add_argument('--failure-report', default=None)
    return parser.parse_args(argv)


def main() -> dict:
    args = parse_args()

    if args.emit_source_version:
        try:
            print(resolve_ref(args.ref))
        except Exception as exc:  # noqa: BLE001 - probe must not fail a run
            print(f'unknown: {type(exc).__name__}: {exc}', file=sys.stderr)
            return {'status': 'source_version_unavailable'}
        return {'status': 'ok'}

    out_root = Path(args.output_dir).resolve()
    registry = Registry(
        enabled=not args.no_registry_resolve, live=args.registry_live
    )
    models = (
        {m.strip() for m in args.models.split(',') if m.strip()}
        if args.models
        else None
    )

    temporary: Optional[tempfile.TemporaryDirectory] = None
    if args.results_dir:
        results_root = Path(args.results_dir).resolve()
        source_commit = args.ref
    else:
        source_commit = resolve_ref(args.ref)
        temporary = tempfile.TemporaryDirectory(prefix='eee-mteb-')
        results_root = download_results(source_commit, Path(temporary.name))
        raw_capture.record_pointer(
            kind='git',
            reference=RESULTS_REPO,
            revision=source_commit,
            url=RESULTS_REPO_URL,
            label='mteb-results',
            note=f'ref={args.ref}',
            revision_required=True,
        )

    try:
        result = convert(
            results_root,
            out_root,
            source_commit=source_commit,
            registry=registry,
            limit=args.limit,
            models=models,
            max_subsets_per_task=args.max_subsets_per_task,
        )
        paths = save_evaluation_logs(result.records)
    finally:
        if temporary is not None:
            temporary.cleanup()

    report_path = Path(
        args.failure_report or default_failure_report_path(args.output_dir)
    )
    if result.failures or result.exclusions:
        save_failure_report(result, report_path)

    summary = {
        'records': len(paths),
        'failures': len(result.failures),
        'exclusions': len(result.exclusions),
        'source_commit': source_commit,
        'output_dir': str(out_root),
    }
    print(json.dumps(summary, indent=2))
    if result.failures:
        print(
            f'{len(result.failures)} source rows could not be converted; '
            f'see {report_path}',
            file=sys.stderr,
        )
        sys.exit(1)
    return summary


if __name__ == '__main__':
    main()
