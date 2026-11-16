#!/usr/bin/env python3
"""Convert public tau-bench leaderboard submissions into EEE records.

Data source:
- tau-bench leaderboard: https://taubench.com
- Static submissions JSON:
  https://github.com/sierra-research/tau2-bench/tree/main/web/leaderboard/public/submissions

The adapter emits one ``EvaluationLog`` per tau-bench submission. Each log
contains one ``EvaluationResult`` per populated domain metric, for example
``tau_bench.text.retail.pass_1`` or
``tau_bench.text.banking_knowledge.cost``.

A remote run resolves the base URL's ref to a commit before fetching anything,
so every record cites bytes that cannot change under it; a local run records
the payload hash and the path it read instead of claiming an upstream URL.

Usage:
    uv run python -m every_eval_ever.adapters.tau_bench.adapter \\
        --output-dir /tmp/eee-tau-bench/tau-bench
    uv run python -m every_eval_ever.adapters.tau_bench.adapter \\
        --input-dir /tmp/tau2-submissions --output-dir /tmp/eee-tau-bench
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
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
    SourceDataUrl,
    SourceMetadata,
    SourceType,
)
from every_eval_ever.helpers import (
    SCHEMA_VERSION,
    EvaluationLogOutput,
    FetchError,
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    default_failure_report_path,
    fetch_json,
    require_finite_number,
    sanitize_filename,
    save_evaluation_logs,
    save_failure_report,
)

SOURCE_NAME = 'tau-bench Leaderboard'
SOURCE_ORGANIZATION = 'Sierra'
SOURCE_ORGANIZATION_URL = 'https://taubench.com'
LEADERBOARD_URL = 'https://taubench.com'
SUBMISSIONS_TREE_URL = (
    'https://github.com/sierra-research/tau2-bench/tree/main/'
    'web/leaderboard/public/submissions'
)
RAW_SUBMISSIONS_BASE_URL = (
    'https://raw.githubusercontent.com/sierra-research/tau2-bench/main/'
    'web/leaderboard/public/submissions'
)
DEFAULT_OUTPUT_DIR = 'data/tau-bench'

#: A raw-GitHub URL, split so a mutable ref can be swapped for a commit sha.
_RAW_GITHUB_RE = re.compile(
    r'^https://raw\.githubusercontent\.com/'
    r'(?P<owner>[^/]+)/(?P<repo>[^/]+)/(?P<ref>[^/]+)/(?P<path>.+)$'
)
_COMMIT_SHA_RE = re.compile(r'^[0-9a-f]{40}$')

MANIFEST_FILE_NAME = 'manifest.json'
SUBMISSION_FILE_NAME = 'submission.json'
MANIFEST_SECTIONS = (
    'submissions',
    'voice_submissions',
    'legacy_submissions',
)
DOMAINS = ('retail', 'airline', 'telecom', 'banking_knowledge')
PASS_METRICS = ('pass_1', 'pass_2', 'pass_3', 'pass_4')
#: The scales the leaderboard reports on, as ``(minimum, maximum)`` with
#: ``None`` for unbounded. Pass^k is a percentage; cost is USD per trajectory.
PASS_SCORE_BOUNDS: tuple[float, float | None] = (0.0, 100.0)
COST_SCORE_BOUNDS: tuple[float, float | None] = (0.0, None)


@dataclass(frozen=True)
class InteractionMetricSpec:
    """One per-domain voice interaction metric and how it is scored.

    ``bounds`` is the reported scale as ``(minimum, maximum)`` with ``None``
    for unbounded above, and is what ``parse_score`` range-checks. Direction
    follows the tau-bench voice panel: latency is seconds and lower is better,
    the response/yield/selectivity rates are proportions and higher is better,
    and the agent interruption rate is a proportion where lower is better.
    """

    key: str
    metric_name: str
    metric_kind: str
    metric_unit: str
    lower_is_better: bool
    bounds: tuple[float, float | None]
    description: str


#: Voice submissions carry a per-domain interaction panel next to the Pass^k
#: results; without these the voice-specific measurements are dropped on the
#: floor. Rates are proportions in ``[0, 1]``; latency is seconds, unbounded
#: above.
INTERACTION_METRICS: tuple[InteractionMetricSpec, ...] = (
    InteractionMetricSpec(
        'response_latency_mean',
        'Response latency (mean)',
        'latency',
        'seconds',
        True,
        (0.0, None),
        'mean time before the agent responds when it should',
    ),
    InteractionMetricSpec(
        'yield_latency_mean',
        'Yield latency (mean)',
        'latency',
        'seconds',
        True,
        (0.0, None),
        'mean time before the agent yields the floor when it should',
    ),
    InteractionMetricSpec(
        'response_rate',
        'Response rate',
        'rate',
        'proportion',
        False,
        (0.0, 1.0),
        'share of turns the agent responded on when it should',
    ),
    InteractionMetricSpec(
        'yield_rate',
        'Yield rate',
        'rate',
        'proportion',
        False,
        (0.0, 1.0),
        'share of turns the agent yielded the floor on when it should',
    ),
    InteractionMetricSpec(
        'agent_interruption_rate',
        'Agent interruption rate',
        'rate',
        'proportion',
        True,
        (0.0, 1.0),
        'share of turns the agent interrupted the user',
    ),
    InteractionMetricSpec(
        'selectivity_backchannel',
        'Backchannel selectivity',
        'rate',
        'proportion',
        False,
        (0.0, 1.0),
        'share of user backchannels the agent correctly did not treat as a '
        'turn to respond to',
    ),
    InteractionMetricSpec(
        'selectivity_vocal_tic',
        'Vocal-tic selectivity',
        'rate',
        'proportion',
        False,
        (0.0, 1.0),
        'share of user vocal tics the agent correctly did not treat as a '
        'turn to respond to',
    ),
    InteractionMetricSpec(
        'selectivity_non_directed',
        'Non-directed-speech selectivity',
        'rate',
        'proportion',
        False,
        (0.0, 1.0),
        'share of non-directed user speech the agent correctly did not treat '
        'as a turn to respond to',
    ),
)

ORGANIZATION_SLUGS = {
    'Alibaba Cloud': 'alibaba',
    'Anthropic': 'anthropic',
    'DeepSeek': 'deepseek',
    'Distyl AI': 'distyl',
    'Google': 'google',
    'Moonshot AI': 'moonshot-ai',
    'Multiple providers': 'multiple',
    'NVIDIA': 'nvidia',
    'OpenAI': 'openai',
    'Qwen': 'qwen',
    'Sierra': 'sierra',
    'xAI': 'xai',
    'Z.ai': 'zhipu-ai',
    'Zhipu': 'zhipu-ai',
    'Zhipu AI': 'zhipu-ai',
}

# The leaderboard spells one provider several ways across submissions ('Z.ai'
# and 'Zhipu AI' are both present), and an unmapped spelling becomes its own
# developer slug and its own datastore directory. Matching on a normalized
# name keeps a new spelling of a provider already in the map from splitting
# it; a genuinely new provider still falls through to slugify.
_ORGANIZATION_SLUGS_BY_NORMALIZED_NAME = {
    name.strip().lower(): slug for name, slug in ORGANIZATION_SLUGS.items()
}


@dataclass(frozen=True)
class TauBenchSubmission:
    """One submission payload and the exact input it was read from.

    ``source_url`` is the immutable URL the bytes came from, and is ``None``
    for a local replay, where there is no upstream URL that is known to hold
    these bytes. ``content_sha256`` identifies the payload itself either way,
    so a record can name its input even when the input was a local file or a
    ``--base-url`` that is not the public leaderboard.
    """

    submission_id: str
    manifest_section: str
    submission: dict[str, Any]
    source_url: str | None
    content_sha256: str | None = None
    local_path: str | None = None
    source_commit: str | None = None

    @property
    def payload_sha256(self) -> str:
        """Hash the bytes when the loader kept them, the payload otherwise."""
        if self.content_sha256 is not None:
            return self.content_sha256
        return _canonical_sha256(self.submission)


@dataclass(frozen=True)
class EvaluationBundle:
    log: EvaluationLog
    developer: str
    model_name: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert tau-bench leaderboard JSON into EEE records.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(DEFAULT_OUTPUT_DIR),
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR}).',
    )
    parser.add_argument(
        '--base-url',
        default=RAW_SUBMISSIONS_BASE_URL,
        help='Base URL containing manifest.json and submission folders.',
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        help=(
            'Read a local tau-bench submissions directory instead of '
            'fetching from --base-url.'
        ),
    )
    parser.add_argument(
        '--sections',
        nargs='+',
        choices=MANIFEST_SECTIONS,
        default=list(MANIFEST_SECTIONS),
        help='Manifest sections to export.',
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Optional maximum number of submissions to fetch and export.',
    )
    parser.add_argument(
        '--allow-unpinned-source',
        action='store_true',
        help=(
            'Proceed when the base URL cannot be resolved to a commit, '
            'recording the mutable ref instead of a pinned one.'
        ),
    )
    parser.add_argument(
        '--failure-report',
        type=Path,
        help=(
            'Write rejected submissions and reasons here. Defaults beside '
            '--output-dir when any submission fails.'
        ),
    )
    parser.add_argument(
        '--emit-source-version',
        action='store_true',
        help='Print the upstream tau2-bench commit SHA and exit.',
    )
    return parser.parse_args(argv)


def _canonical_sha256(payload: Any) -> str:
    """Hash a payload by its canonical JSON form."""
    serialized = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), allow_nan=False
    )
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _git_ls_remote_commit(owner: str, repo: str, ref: str) -> str | None:
    """Resolve ``owner/repo@ref`` to a commit SHA via ``git ls-remote``."""
    env = dict(os.environ)
    env['GIT_TERMINAL_PROMPT'] = '0'
    query_refs = (
        ['HEAD']
        if ref == 'HEAD'
        else [f'refs/heads/{ref}', f'refs/tags/{ref}^{{}}', f'refs/tags/{ref}']
    )
    completed = subprocess.run(
        ['git', 'ls-remote', f'https://github.com/{owner}/{repo}.git', *query_refs],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        return None
    found: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and _COMMIT_SHA_RE.match(parts[0]):
            found[parts[1]] = parts[0]
    for target in query_refs:
        if target in found:
            return found[target]
    return None


def pin_base_url(base_url: str) -> tuple[str, str | None]:
    """Resolve a raw-GitHub base URL's ref to the commit it points at now.

    The default base URL names the ``main`` branch, and a record citing it
    cannot say which bytes produced its scores — the same URL serves different
    content next week. Resolving the ref once, before any submission is
    fetched, makes every URL in the run immutable and gives the batch one
    commit to record.

    Returns ``(base_url, ref)`` when ``ref`` is already a commit sha, or
    ``(base_url, None)`` when ``base_url`` is not a raw-GitHub URL; the caller
    is then relying on ``content_sha256`` alone.
    """
    match = _RAW_GITHUB_RE.match(base_url)
    if match is None:
        return base_url, None
    owner, repo, ref, path = match.group('owner', 'repo', 'ref', 'path')
    if _COMMIT_SHA_RE.match(ref):
        return base_url, ref
    sha: str | None = None
    try:
        commit = fetch_json(
            f'https://api.github.com/repos/{owner}/{repo}/commits/{ref}'
        )
        sha = (commit or {}).get('sha') if isinstance(commit, dict) else None
    except FetchError as exc:
        status = getattr(
            getattr(exc.__cause__, 'response', None), 'status_code', None
        )
        if status in (403, 429):
            sha = _git_ls_remote_commit(owner, repo, ref)
        if sha is None:
            raise
    if not isinstance(sha, str) or not _COMMIT_SHA_RE.match(sha):
        raise ValueError(
            f'Could not resolve {owner}/{repo}@{ref} to a commit; the '
            'response carried no usable sha. Pass --base-url with an explicit '
            'commit, or --allow-unpinned-source to record the mutable ref.'
        )
    return f'https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{path}', sha


def load_submissions(
    *,
    input_dir: Path | None = None,
    base_url: str = RAW_SUBMISSIONS_BASE_URL,
    sections: list[str] | tuple[str, ...] = MANIFEST_SECTIONS,
    limit: int | None = None,
    allow_unpinned_source: bool = False,
) -> list[TauBenchSubmission]:
    if input_dir is not None:
        return load_submissions_from_dir(input_dir, sections, limit=limit)
    return load_submissions_from_url(
        base_url,
        sections,
        limit=limit,
        allow_unpinned_source=allow_unpinned_source,
    )


def load_submissions_from_url(
    base_url: str,
    sections: list[str] | tuple[str, ...] = MANIFEST_SECTIONS,
    *,
    limit: int | None = None,
    allow_unpinned_source: bool = False,
) -> list[TauBenchSubmission]:
    """Fetch the manifest and the submissions it names.

    ``limit`` bounds what is downloaded rather than what is kept: slicing
    afterwards fetched every selected submission to throw most of them away,
    which for a quick check against the live leaderboard is one request per
    submission for nothing.
    """
    base_url = base_url.rstrip('/')
    commit: str | None = None
    try:
        base_url, commit = pin_base_url(base_url)
    except Exception:
        if not allow_unpinned_source:
            raise
    manifest = fetch_json(f'{base_url}/{MANIFEST_FILE_NAME}')
    records = []
    for section, submission_id in iter_manifest_ids(
        manifest, sections, limit=limit
    ):
        source_url = f'{base_url}/{submission_id}/{SUBMISSION_FILE_NAME}'
        submission = fetch_json(source_url)
        records.append(
            TauBenchSubmission(
                submission_id=submission_id,
                manifest_section=section,
                submission=submission,
                source_url=source_url,
                content_sha256=_canonical_sha256(submission),
                source_commit=commit,
            )
        )
    return records


def load_submissions_from_dir(
    input_dir: Path,
    sections: list[str] | tuple[str, ...] = MANIFEST_SECTIONS,
    *,
    limit: int | None = None,
) -> list[TauBenchSubmission]:
    manifest_path = input_dir / MANIFEST_FILE_NAME
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records = []
    for section, submission_id in iter_manifest_ids(
        manifest, sections, limit=limit
    ):
        path = input_dir / submission_id / SUBMISSION_FILE_NAME
        payload = path.read_bytes()
        submission = json.loads(payload.decode('utf-8'))
        records.append(
            TauBenchSubmission(
                submission_id=submission_id,
                manifest_section=section,
                submission=submission,
                # No upstream URL: these bytes came off disk, and the public
                # leaderboard is not known to hold them.
                source_url=None,
                content_sha256=hashlib.sha256(payload).hexdigest(),
                local_path=str(path),
            )
        )
    return records


def iter_manifest_ids(
    manifest: dict[str, Any],
    sections: list[str] | tuple[str, ...],
    *,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    pairs = []
    seen: set[str] = set()
    for section in sections:
        values = manifest.get(section) or []
        if not isinstance(values, list):
            raise ValueError(f'Manifest section {section!r} must be a list')
        for value in values:
            submission_id = str(value)
            if submission_id in seen:
                continue
            seen.add(submission_id)
            pairs.append((section, submission_id))
            if limit is not None and len(pairs) >= limit:
                return pairs
    return pairs


def convert_logs(
    records: list[TauBenchSubmission],
    *,
    retrieved_timestamp: str | None = None,
) -> SourceConversionResult[EvaluationBundle]:
    """Convert every submission, accounting for the ones that produce no record.

    A submission the leaderboard lists with no scores in any domain is an
    exclusion, not a failure: it is a real row that carries nothing to publish.
    A submission that raises is a failure, and the batch keeps going so one
    malformed payload does not cost the run.
    """
    retrieved_timestamp = retrieved_timestamp or str(time.time())
    bundles: list[EvaluationBundle] = []
    failures: list[SourceRecordFailure] = []
    exclusions: list[SourceRecordExclusion] = []
    for record in records:
        try:
            bundle = make_log(record, retrieved_timestamp)
        except Exception as exc:
            failures.append(
                SourceRecordFailure(
                    source_ref=record.submission_id,
                    reason=f'{type(exc).__name__}: {exc}',
                )
            )
            continue
        if bundle is None:
            exclusions.append(
                SourceRecordExclusion(
                    source_ref=record.submission_id,
                    reason=(
                        'submission reports no Pass^k or cost score in any '
                        'tau-bench domain'
                    ),
                )
            )
            continue
        bundles.append(bundle)
    return SourceConversionResult(
        source_name=SOURCE_NAME,
        total_records=len(records),
        records=bundles,
        failures=failures,
        exclusions=exclusions,
    )


def make_logs(
    records: list[TauBenchSubmission],
    *,
    retrieved_timestamp: str | None = None,
) -> list[EvaluationBundle]:
    result = convert_logs(records, retrieved_timestamp=retrieved_timestamp)
    result.raise_if_incomplete()
    return result.records


def make_log(
    record: TauBenchSubmission,
    retrieved_timestamp: str,
) -> EvaluationBundle | None:
    submission = record.submission
    model_name = required_str(submission, 'model_name', record.submission_id)
    model_org = required_str(
        submission, 'model_organization', record.submission_id
    )
    developer = organization_slug(model_org)
    model_slug = slugify(model_name)
    model_id = f'{developer}/{model_slug}'
    results = make_results(record, model_id)
    if not results:
        return None

    evaluation_timestamp = evaluation_date(submission)
    version = (
        _mapping(submission.get('methodology')).get('tau2_bench_version')
        or 'unknown'
    )
    sanitized_model_id = sanitize_filename(model_id)
    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=(
            f'tau-bench/{sanitized_model_id}/{record.submission_id}/'
            f'{retrieved_timestamp}'
        ),
        evaluation_timestamp=evaluation_timestamp,
        retrieved_timestamp=retrieved_timestamp,
        source_metadata=make_source_metadata(record),
        eval_library=EvalLibrary(
            name='tau2-bench',
            version=str(version),
            additional_details=_clean_details(
                {
                    'leaderboard_url': LEADERBOARD_URL,
                    'submissions_tree_url': SUBMISSIONS_TREE_URL,
                }
            ),
        ),
        model_info=ModelInfo(
            name=model_name,
            id=model_id,
            developer=developer,
            additional_details=make_model_details(record),
        ),
        evaluation_results=results,
    )
    return EvaluationBundle(log=log, developer=developer, model_name=model_slug)


def make_results(
    record: TauBenchSubmission,
    model_id: str,
) -> list[EvaluationResult]:
    submission = record.submission
    all_results = submission.get('results') or {}
    if not isinstance(all_results, dict):
        raise ValueError(
            f'{record.submission_id} has invalid results: expected object'
        )

    results = []
    for domain in DOMAINS:
        domain_results = all_results.get(domain)
        if domain_results is None:
            continue
        if not isinstance(domain_results, dict):
            raise ValueError(
                f'{record.submission_id}/{domain} results must be an object'
            )

        for metric in PASS_METRICS:
            score = parse_score(
                domain_results.get(metric),
                context=f'{record.submission_id}/{domain}/{metric}',
                bounds=PASS_SCORE_BOUNDS,
            )
            if score is None:
                continue
            results.append(
                make_result(
                    record,
                    model_id=model_id,
                    domain=domain,
                    metric=metric,
                    score=score,
                    domain_results=domain_results,
                )
            )

        cost = parse_score(
            domain_results.get('cost'),
            context=f'{record.submission_id}/{domain}/cost',
            bounds=COST_SCORE_BOUNDS,
        )
        if cost is not None:
            results.append(
                make_result(
                    record,
                    model_id=model_id,
                    domain=domain,
                    metric='cost',
                    score=cost,
                    domain_results=domain_results,
                )
            )
    results.extend(make_interaction_results(record, model_id))
    return results


def make_interaction_results(
    record: TauBenchSubmission,
    model_id: str,
) -> list[EvaluationResult]:
    """Emit the per-domain voice interaction panel, one result per metric.

    Voice submissions report an ``interaction_metrics`` block alongside their
    Pass^k results. It is absent for text submissions, so a missing block is
    nothing to publish rather than an error. The ``overall`` aggregate is kept
    under a synthetic ``overall`` domain so it is not dropped either.
    """
    interaction = _mapping(record.submission.get('interaction_metrics'))
    if not interaction:
        return []
    version = interaction.get('version')
    config = interaction.get('config')
    domain_panels = _mapping(interaction.get('domains'))

    panels: list[tuple[str, Any]] = [
        (domain, domain_panels[domain])
        for domain in DOMAINS
        if domain in domain_panels
    ]
    overall = interaction.get('overall')
    if overall is not None:
        panels.append(('overall', overall))

    results = []
    for domain, panel in panels:
        if not isinstance(panel, dict):
            raise ValueError(
                f'{record.submission_id} interaction_metrics/{domain} must '
                'be an object'
            )
        for spec in INTERACTION_METRICS:
            score = parse_score(
                panel.get(spec.key),
                context=f'{record.submission_id}/{domain}/{spec.key}',
                bounds=spec.bounds,
            )
            if score is None:
                continue
            results.append(
                make_interaction_result(
                    record,
                    model_id=model_id,
                    domain=domain,
                    spec=spec,
                    score=score,
                    panel=panel,
                    version=version,
                    config=config,
                )
            )
    return results


def make_interaction_result(
    record: TauBenchSubmission,
    *,
    model_id: str,
    domain: str,
    spec: InteractionMetricSpec,
    score: float,
    panel: dict[str, Any],
    version: Any,
    config: Any,
) -> EvaluationResult:
    submission = record.submission
    modality = str(submission.get('modality') or 'text')
    return EvaluationResult(
        evaluation_result_id=(
            f'tau_bench:{record.submission_id}:{domain}:{spec.key}'
        ),
        evaluation_name=f'tau_bench.{modality}.{domain}.{spec.key}',
        source_data=make_source_data(record, domain, panel),
        evaluation_timestamp=evaluation_date(submission),
        metric_config=make_interaction_metric_config(
            spec, domain=domain, version=version, config=config
        ),
        score_details=ScoreDetails(
            score=score,
            details=_clean_details(
                {
                    'submission_id': record.submission_id,
                    'model_id': model_id,
                    'domain': domain,
                    'metric': spec.key,
                    'raw_score': panel.get(spec.key),
                    'counts': panel.get('counts'),
                }
            ),
        ),
        generation_config=make_generation_config(submission, domain),
    )


def make_interaction_metric_config(
    spec: InteractionMetricSpec,
    *,
    domain: str,
    version: Any,
    config: Any,
) -> MetricConfig:
    minimum, maximum = spec.bounds
    return MetricConfig(
        evaluation_description=(
            f'tau-bench {domain} voice interaction metric: {spec.description}.'
        ),
        metric_id=f'tau_bench.interaction.{spec.key}',
        metric_name=spec.metric_name,
        metric_kind=spec.metric_kind,
        metric_unit=spec.metric_unit,
        lower_is_better=spec.lower_is_better,
        score_type=ScoreType.continuous,
        min_score=minimum,
        max_score=float('inf') if maximum is None else maximum,
        additional_details=_clean_details(
            {
                'domain': domain,
                'direction': (
                    'lower_is_better'
                    if spec.lower_is_better
                    else 'higher_is_better'
                ),
                'interaction_metrics_version': version,
                'interaction_metrics_config': config,
            }
        ),
    )


def make_result(
    record: TauBenchSubmission,
    *,
    model_id: str,
    domain: str,
    metric: str,
    score: float,
    domain_results: dict[str, Any],
) -> EvaluationResult:
    submission = record.submission
    modality = str(submission.get('modality') or 'text')
    metric_config = make_metric_config(domain=domain, metric=metric)
    evaluation_name = f'tau_bench.{modality}.{domain}.{metric}'

    return EvaluationResult(
        evaluation_result_id=(
            f'tau_bench:{record.submission_id}:{domain}:{metric}'
        ),
        evaluation_name=evaluation_name,
        source_data=make_source_data(record, domain, domain_results),
        evaluation_timestamp=evaluation_date(submission),
        metric_config=metric_config,
        score_details=ScoreDetails(
            score=score,
            details=_clean_details(
                {
                    'submission_id': record.submission_id,
                    'model_id': model_id,
                    'domain': domain,
                    'metric': metric,
                    'raw_score': domain_results.get(metric),
                    'retrieval_config': domain_results.get('retrieval_config'),
                }
            ),
        ),
        generation_config=make_generation_config(submission, domain),
    )


def make_metric_config(*, domain: str, metric: str) -> MetricConfig:
    if metric.startswith('pass_'):
        k = int(metric.split('_', 1)[1])
        return MetricConfig(
            evaluation_description=(
                f'tau-bench {domain} Pass^{k} success rate reported on the '
                f'public leaderboard: the share of tasks solved in all {k} '
                'independent trials.'
            ),
            metric_id='tau_bench.pass_hat_k',
            metric_name=f'Pass^{k}',
            metric_kind='pass_rate',
            metric_unit='percent',
            metric_parameters={'k': k},
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=100.0,
            additional_details=_clean_details(
                {
                    'domain': domain,
                    'score_scale': 'percent_0_to_100',
                    'metric_semantics': (
                        f'pass_hat_k: all {k} trials succeed. Not pass@k, '
                        'which counts a task solved when at least one of k '
                        'trials succeeds.'
                    ),
                }
            ),
        )

    if metric == 'cost':
        return MetricConfig(
            evaluation_description=(
                f'Average tau-bench cost per trajectory for {domain}, in USD, '
                'when reported by the submission.'
            ),
            metric_id='tau_bench.cost_per_trajectory',
            metric_name='Cost per trajectory',
            metric_kind='cost',
            metric_unit='usd_per_trajectory',
            lower_is_better=True,
            additional_details=_clean_details({'domain': domain}),
        )

    raise ValueError(f'Unsupported tau-bench metric: {metric}')


def make_source_data(
    record: TauBenchSubmission,
    domain: str,
    domain_results: dict[str, Any],
) -> SourceDataUrl:
    urls = [LEADERBOARD_URL]
    if record.source_url is not None:
        urls.append(record.source_url)
    return SourceDataUrl(
        dataset_name=f'tau-bench {domain}',
        source_type='url',
        url=urls,
        additional_details=_clean_details(
            {
                'domain': domain,
                'submission_id': record.submission_id,
                'manifest_section': record.manifest_section,
                'submission_sha256': record.payload_sha256,
                'submission_commit': record.source_commit,
                'local_input_path': record.local_path,
                'retrieval_config': domain_results.get('retrieval_config'),
                'trajectory_file': _mapping(
                    record.submission.get('trajectory_files')
                ).get(domain),
            }
        ),
    )


def make_source_metadata(record: TauBenchSubmission) -> SourceMetadata:
    submission = record.submission
    return SourceMetadata(
        source_name=SOURCE_NAME,
        source_type=SourceType.documentation,
        source_organization_name=SOURCE_ORGANIZATION,
        source_organization_url=SOURCE_ORGANIZATION_URL,
        evaluator_relationship=evaluator_relationship(submission),
        additional_details=_clean_details(
            {
                'leaderboard_url': LEADERBOARD_URL,
                'submissions_tree_url': SUBMISSIONS_TREE_URL,
                'submission_source_url': record.source_url,
                'submission_sha256': record.payload_sha256,
                'submission_commit': record.source_commit,
                'local_input_path': record.local_path,
                'submission_id': record.submission_id,
                'manifest_section': record.manifest_section,
                'submission_date': submission.get('submission_date'),
                'submission_type': submission.get('submission_type'),
                'modality': submission.get('modality') or 'text',
                'submitting_organization': submission.get(
                    'submitting_organization'
                ),
            }
        ),
    )


def make_model_details(
    record: TauBenchSubmission,
) -> dict[str, str] | None:
    submission = record.submission
    model_release = _mapping(submission.get('model_release'))
    references = submission.get('references') or []
    if not isinstance(references, list):
        references = [references]
    return _clean_details(
        {
            'raw_model_organization': submission.get('model_organization'),
            'submitting_organization': submission.get(
                'submitting_organization'
            ),
            'submission_id': record.submission_id,
            'submission_date': submission.get('submission_date'),
            'submission_type': submission.get('submission_type'),
            'modality': submission.get('modality') or 'text',
            'trajectories_available': submission.get('trajectories_available'),
            'reasoning_effort': submission.get('reasoning_effort'),
            'model_release_date': model_release.get('release_date'),
            'model_release_announcement_url': model_release.get(
                'announcement_url'
            ),
            'references': references or None,
        }
    )


def make_generation_config(
    submission: dict[str, Any],
    domain: str,
) -> GenerationConfig:
    methodology = _mapping(submission.get('methodology'))
    voice_config = _mapping(submission.get('voice_config'))
    pipeline = voice_config.get('pipeline')
    return GenerationConfig(
        generation_args=GenerationArgs(
            agentic_eval_config=AgenticEvalConfig(
                available_tools=[
                    AvailableTool(
                        name=f'tau-bench {domain} tools',
                        description=(
                            'Domain-specific customer service tools exposed '
                            'by the tau-bench environment.'
                        ),
                    )
                ],
                additional_details=_clean_details({'domain': domain}),
            )
        ),
        additional_details=_clean_details(
            {
                'evaluation_date': methodology.get('evaluation_date'),
                'tau2_bench_version': methodology.get('tau2_bench_version'),
                'user_simulator': methodology.get('user_simulator'),
                'methodology_notes': methodology.get('notes'),
                'verification': methodology.get('verification'),
                'submission_type': submission.get('submission_type'),
                'modality': submission.get('modality') or 'text',
                'reasoning_effort': submission.get('reasoning_effort'),
                'voice_provider': voice_config.get('provider'),
                'voice_model': voice_config.get('model'),
                'voice_tick_duration_seconds': voice_config.get(
                    'tick_duration_seconds'
                ),
                'voice_max_steps_seconds': voice_config.get(
                    'max_steps_seconds'
                ),
                'voice_user_tts_provider': voice_config.get(
                    'user_tts_provider'
                ),
                'voice_pipeline': pipeline,
            }
        ),
    )


def evaluator_relationship(
    submission: dict[str, Any],
) -> EvaluatorRelationship:
    model_org = slugify(str(submission.get('model_organization') or ''))
    submitter = slugify(str(submission.get('submitting_organization') or ''))
    if model_org and submitter and model_org == submitter:
        return EvaluatorRelationship.first_party
    return EvaluatorRelationship.third_party


def evaluation_date(submission: dict[str, Any]) -> str | None:
    methodology = _mapping(submission.get('methodology'))
    value = methodology.get('evaluation_date') or submission.get(
        'submission_date'
    )
    return str(value) if value else None


def required_str(
    payload: dict[str, Any],
    key: str,
    submission_id: str,
) -> str:
    value = payload.get(key)
    if value is None or str(value).strip() == '':
        raise ValueError(f'{submission_id} is missing required field {key}')
    return str(value)


def parse_score(
    raw: Any,
    *,
    context: str,
    bounds: tuple[float, float | None] | None = None,
) -> float | None:
    """Return one reported score, or ``None`` where the source reports none.

    ``bounds`` is checked because ``--input-dir`` and ``--base-url`` accept
    inputs the public leaderboard never serves: a replayed or hand-built
    submission can carry a percentage above 100, a negative cost, or a boolean,
    and a record built from one of those is invalid data rather than a record
    of what the source said.
    """
    if raw is None or raw == '':
        return None
    score = require_finite_number(raw, f'tau-bench score for {context}')
    if bounds is not None:
        minimum, maximum = bounds
        if score < minimum or (maximum is not None and score > maximum):
            expected = (
                f'{minimum} to {maximum}'
                if maximum is not None
                else f'{minimum} or greater'
            )
            raise ValueError(
                f'tau-bench score for {context} is outside the reported '
                f'scale ({expected}): {score!r}'
            )
    return score


def organization_slug(name: str) -> str:
    normalized = name.strip().lower()
    mapped = _ORGANIZATION_SLUGS_BY_NORMALIZED_NAME.get(normalized)
    return mapped if mapped is not None else slugify(name)


def slugify(value: str) -> str:
    base = re.sub(r'[^\w.\-]+', '-', value.strip().lower())
    base = re.sub(r'-{2,}', '-', base).strip('-')
    return sanitize_filename(base) or 'unknown'


def submission_source_url(submission_id: str) -> str:
    return f'{RAW_SUBMISSIONS_BASE_URL}/{submission_id}/{SUBMISSION_FILE_NAME}'


def export_logs(
    bundles: list[EvaluationBundle],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> list[Path]:
    """Publish the batch, or leave the tree as it was.

    ``save_evaluation_logs`` validates and serializes every record before it
    creates the first file, and removes what it created if a later write fails,
    so a run cannot leave half a leaderboard behind.
    """
    return save_evaluation_logs(
        EvaluationLogOutput(
            eval_log=bundle.log,
            base_dir=output_dir,
            developer=bundle.developer,
            model_name=bundle.model_name,
        )
        for bundle in bundles
    )


def _mapping(value: Any) -> dict[str, Any]:
    """Return *value* when it is an object, and an empty object otherwise.

    Every nested block in a submission (``methodology``, ``voice_config``,
    ``model_release``, ``trajectory_files``) is optional upstream, and a
    hand-built or older submission can carry a string or a list where the
    current schema has an object. ``x or {}`` only covers the absent case, so
    reading through it raised ``AttributeError`` on the malformed one.
    """
    return value if isinstance(value, dict) else {}


def _clean_details(values: dict[str, Any]) -> dict[str, str] | None:
    details = {
        key: _detail_value(value)
        for key, value in values.items()
        if value is not None
    }
    return details or None


def _detail_value(value: Any) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _probe_submissions_commit(
    owner: str, repo: str, ref: str, path: str
) -> str | None:
    """Return the latest commit touching ``path`` on ``ref``, without raw capture."""
    if _COMMIT_SHA_RE.match(ref):
        return ref
    from urllib.error import HTTPError, URLError
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen

    query = urlencode({'path': path, 'sha': ref, 'per_page': 1})
    url = f'https://api.github.com/repos/{owner}/{repo}/commits?{query}'
    req = Request(url, headers={'Accept': 'application/vnd.github+json'})
    try:
        with urlopen(req, timeout=30) as resp:  # noqa: S310
            commits = json.loads(resp.read().decode('utf-8'))
        if isinstance(commits, list) and commits:
            sha = commits[0].get('sha') if isinstance(commits[0], dict) else None
            if isinstance(sha, str) and _COMMIT_SHA_RE.match(sha):
                return sha
    except HTTPError as exc:
        if exc.code in (403, 429):
            return _git_ls_remote_commit(owner, repo, ref)
        return None
    except (URLError, TimeoutError, ValueError):
        return None
    return None


def run(args: argparse.Namespace) -> int:
    if getattr(args, 'emit_source_version', False):
        if args.input_dir is not None:
            raise SystemExit(1)
        match = _RAW_GITHUB_RE.match(args.base_url.rstrip('/'))
        if match is None:
            raise SystemExit(1)
        owner, repo, ref, path = match.group('owner', 'repo', 'ref', 'path')
        sha = _probe_submissions_commit(owner, repo, ref, path)
        if not sha:
            raise SystemExit(1)
        print(sha)
        return 0

    records = load_submissions(
        input_dir=args.input_dir,
        base_url=args.base_url,
        sections=args.sections,
        limit=args.limit,
        allow_unpinned_source=args.allow_unpinned_source,
    )
    result = convert_logs(records)
    paths = export_logs(result.records, args.output_dir)
    for path in paths:
        print(path)
    if result.failures or result.exclusions:
        report_path = save_failure_report(
            result,
            args.failure_report
            or default_failure_report_path(args.output_dir),
        )
        print(f'Conversion report: {report_path}')
    result.raise_if_incomplete()
    return len(paths)


if __name__ == '__main__':
    _args = parse_args()
    written = run(_args)
    if not _args.emit_source_version:
        print(f'Wrote {written} tau-bench submission log(s).')
