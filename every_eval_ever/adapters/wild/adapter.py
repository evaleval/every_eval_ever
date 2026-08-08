#!/usr/bin/env python3
"""Convert kensho/WILD-raw (arXiv:2604.01418) into Every Eval Ever records.

WILD-raw is item-level evaluation data: ~7.5M (model, item) rows for 65 models
across 27 benchmarks, run by Kensho with Inspect AI. One aggregate log per
(model, benchmark) holds the overall accuracy plus one result per subtask;
`--include-instances` also writes the per-item instance sidecar. See README.md.

Run:
    uv run python -m every_eval_ever.adapters.wild.adapter --output-dir /tmp/eee-wild/data/wild --limit-shards 1
    uv run python -m every_eval_ever validate '/tmp/eee-wild/data/wild/*/*/*.json*'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from every_eval_ever.converters.common.publication import (
    publish_evaluation_logs,
)
from every_eval_ever.eval_types import (
    DetailedEvaluationResults,
    EvalLibrary,
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    Format,
    HashAlgorithm,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceDataHf,
    SourceDataPrivate,
    SourceMetadata,
)
from every_eval_ever.helpers import SCHEMA_VERSION
from every_eval_ever.helpers.io import (
    SourceConversionResult,
    SourceRecordFailure,
    datastore_output_dir,
    datastore_repo_file_path,
    default_failure_report_path,
    save_failure_report,
)
from every_eval_ever.instance_level_types import (
    AnswerAttributionItem,
    Evaluation,
    Input,
    InstanceLevelEvaluationLog,
    InteractionType,
    Output,
    TokenUsage,
)

HF_REPO_ID = 'kensho/WILD-raw'
HF_REVISION = 'main'
N_SHARDS = 15
BATCH_ROWS = 20_000     # rows held in memory per read; see iter_batches
COLLECTION = 'wild'
DEFAULT_OUTPUT_DIR = f'data/{COLLECTION}'
SOURCE_NAME = 'WILD-raw'
SOURCE_ORGANIZATION = 'Kensho'
HF_DATASET_URL = f'https://huggingface.co/datasets/{HF_REPO_ID}'
PAPER_URL = 'https://arxiv.org/abs/2604.01418'

# WILD task -> the dataset the eval ran on (each verified to exist on HF), NOT
# kensho/WILD-raw, which holds the results. Tasks with no clean public repo use the
# `other` variant. Canonicalizing the benchmark id is the eval-card-registry's job.
WILD_DATASET_REPO = {
    'arc_easy': 'allenai/ai2_arc', 'arc_challenge': 'allenai/ai2_arc',
    'bbh': 'lukaemon/bbh', 'bigcodebench': 'bigcode/bigcodebench',
    'boolq': 'google/boolq', 'chembench': 'jablonkagroup/ChemBench',
    'commonsense_qa': 'tau/commonsense_qa', 'drop': 'ucinlp/drop',
    'gsm8k': 'openai/gsm8k', 'gsm_symbolic': 'apple/GSM-Symbolic',
    'hellaswag': 'Rowan/hellaswag', 'ifeval': 'google/IFEval',
    'math': 'hendrycks/competition_math', 'medqa': 'bigbio/med_qa',
    'mmlu': 'cais/mmlu', 'mmlu_pro': 'TIGER-Lab/MMLU-Pro', 'musr': 'TAUR-Lab/MuSR',
    'paws': 'google-research-datasets/paws', 'piqa': 'ybisk/piqa',
    'race_h': 'ehovy/race', 'squad': 'rajpurkar/squad',
    'truthfulqa': 'truthfulqa/truthful_qa', 'winogrande': 'allenai/winogrande',
    # provenance resolved from the WILD paper + Inspect Evals loaders:
    'finance_fundamentals': 'kensho/bizbench', 'pre_flight': 'AirsideLabs/pre-flight-06',
    'bbeh': 'BBEH/bbeh',
}
# aime's two subtasks come from different repos (the exact ones Inspect Evals loads).
AIME_REPO_BY_SUBTASK = {'2024': 'Maxwell-Jia/AIME_2024', '2025': 'math-ai/aime25'}

# item_id is read in the aggregate pass too, to name an unusable row in the report.
AGG_COLUMNS = ['model', 'task', 'subtask', 'item_id', 'score',
               'input_tokens', 'output_tokens']
INSTANCE_COLUMNS = AGG_COLUMNS + ['conversation', 'target', 'answer',
                                  'scores', 'stop_reason']


# parquet streaming (HF or local), batched, column-projected

def _shard_handles(parquet: list[str] | None, limit_shards: int | None,
                   revision: str | None = None):
    """Yield (label, opener) for each parquet source. opener() -> file-like.
    `revision` pins the HF commit for remote reads (see resolve_source_revision)."""
    if parquet:
        sources = parquet
    else:
        sources = [
            f'datasets/{HF_REPO_ID}/data-{i:05d}-of-{N_SHARDS:05d}.parquet'
            for i in range(N_SHARDS)
        ]
    if limit_shards is not None:
        sources = sources[:limit_shards]
    for src in sources:
        if parquet:  # local path
            yield src, (lambda s=src: open(s, 'rb'))
        else:        # HuggingFace
            from huggingface_hub import HfFileSystem
            fs = HfFileSystem()
            rev = revision or HF_REVISION
            yield src, (lambda s=src: fs.open(s, revision=rev))


def iter_batches(parquet: list[str] | None, columns: list[str],
                 limit_shards: int | None = None,
                 revision: str | None = None,
                 batch_size: int = BATCH_ROWS) -> Iterator[tuple[str, dict[str, list]]]:
    """Yield ``(shard_label, {col: [values]})`` in batches of ``batch_size`` rows.

    A WILD shard is a single row group of 500,000 rows, so reading whole row groups
    would hold every selected column for all of them at once — several GB for the
    instance columns, even when ``--max-instances`` stops after the first few.
    """
    for label, opener in _shard_handles(parquet, limit_shards, revision):
        with opener() as fh:
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
                yield label, {c: batch.column(c).to_pylist() for c in columns}


# aggregation

@dataclass
class Agg:
    n: int = 0
    correct: float = 0.0
    in_tok: int = 0
    out_tok: int = 0
    tok_n: int = 0          # rows with complete token usage — the token-mean divisor

    def add(self, score: float, in_t, out_t):
        self.n += 1
        self.correct += score
        if in_t is not None and out_t is not None:
            self.in_tok += int(in_t)
            self.out_tok += int(out_t)
            self.tok_n += 1


def item_score(raw) -> float | None:
    """The row's binary correctness, or ``None`` when it carries no usable one — a
    missing score is not a wrong answer, so it must not be counted as 0."""
    if raw is None:
        return None
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or score not in (0.0, 1.0):
        return None
    return score


def aggregate(parquet, limit_shards, models: set[str] | None,
              revision: str | None = None):
    """Aggregate item rows into ``{(model, task): {subtask|None: Agg}}``, the ``None``
    key being the benchmark overall. Returns ``(groups, total_rows, failures)``: a row
    without a usable 0/1 score is reported, never counted into a denominator."""
    groups: dict[tuple[str, str], dict[str | None, Agg]] = defaultdict(
        lambda: defaultdict(Agg))
    failures: list[SourceRecordFailure] = []
    total = 0
    for label, batch in iter_batches(parquet, AGG_COLUMNS, limit_shards, revision):
        for model, task, subtask, item_id, raw, in_t, out_t in zip(
                batch['model'], batch['task'], batch['subtask'],
                batch['item_id'], batch['score'],
                batch['input_tokens'], batch['output_tokens']):
            if models and model not in models:
                continue
            total += 1
            score = item_score(raw)
            if score is None:
                failures.append(SourceRecordFailure(
                    source_ref=f'{label}#{model}/{task}/{item_id}',
                    reason=f'score {raw!r} is not a usable binary correctness value',
                ))
                continue
            g = groups[(model, task)]
            g[None].add(score, in_t, out_t)                       # benchmark overall
            g[subtask if subtask not in (None, '') else '_'].add(score, in_t, out_t)
    return groups, total, failures


# record construction

def _source_data(task: str, n: int, subtask: str | None = None):
    """The dataset the eval ran on (not WILD-raw, which holds the results)."""
    repo = WILD_DATASET_REPO.get(task)
    if task == 'aime' and subtask in AIME_REPO_BY_SUBTASK:
        repo = AIME_REPO_BY_SUBTASK[subtask]
    if repo:
        return SourceDataHf(dataset_name=task, source_type='hf_dataset',
                            hf_repo=repo, samples_number=n)
    return SourceDataPrivate(
        dataset_name=task, source_type='other',
        additional_details={'note': 'no single public HF dataset repo for this WILD '
                                    'task/subtask; results are in ' + HF_REPO_ID})


def metric_details(agg: Agg) -> dict[str, str]:
    """Item counts, plus token means over the rows that carried token usage — a row
    without token counts leaves the mean rather than averaging in as a zero."""
    details = {'n_items': str(agg.n), 'n_correct': str(int(agg.correct))}
    if agg.tok_n:
        details['n_items_with_token_usage'] = str(agg.tok_n)
        details['mean_input_tokens'] = f'{agg.in_tok / agg.tok_n:.1f}'
        details['mean_output_tokens'] = f'{agg.out_tok / agg.tok_n:.1f}'
    return details


def _result(task: str, subtask: str | None, agg: Agg) -> EvaluationResult:
    name = f'wild.{task}' if subtask is None else f'wild.{task}.{subtask}'
    rid = task if subtask is None else f'{task}::{subtask}'
    accuracy = agg.correct / agg.n if agg.n else 0.0
    # score is binary per item (verified), so accuracy = mean and the analytic
    # standard error of a proportion is sqrt(p(1-p)/n).
    se = math.sqrt(accuracy * (1 - accuracy) / agg.n) if agg.n else 0.0
    level = 'overall' if subtask is None else 'subtask'
    return EvaluationResult(
        evaluation_result_id=rid,
        evaluation_name=name,
        source_data=_source_data(task, agg.n, subtask),
        metric_config=MetricConfig(
            evaluation_description=(
                f'Mean binary item correctness on {name} (WILD-raw).'),
            # The registry's canonical global metric: `evaluation_name` keeps the
            # tasks apart, so the cross-source accuracy join stays whole.
            metric_id='accuracy',
            metric_name='accuracy',
            metric_kind='accuracy',
            metric_unit='proportion',
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=1.0,
            metric_parameters={'aggregation_level': level, 'aggregation': 'micro'},
            additional_details=metric_details(agg),
        ),
        score_details=ScoreDetails(
            score=accuracy,
            details={'n_items': str(agg.n), 'n_correct': str(int(agg.correct))},
            uncertainty={'standard_error': {'value': se, 'method': 'analytic'},
                         'num_samples': agg.n},
        ),
    )


def build_log(model: str, task: str, subs: dict[str | None, Agg],
              eval_ts: str, retrieved_ts: str,
              revision: str | None = None) -> tuple[EvaluationLog, str, str]:
    developer = model.split('/')[0] if '/' in model else 'unknown'
    model_slug = model.split('/')[-1]
    sanitized = model.replace('/', '_')
    real_subs = sorted(k for k in subs if k is not None)
    # With ≤1 distinct subtask the overall IS that subtask, so emit only the overall
    # (17 WILD tasks have "general" as their only subtask).
    results = [_result(task, None, subs[None])]
    if len(real_subs) > 1:
        for sub in real_subs:
            results.append(_result(task, sub, subs[sub]))
    # Only claim a dataset_revision for remote reads (a pinned commit). A local
    # --parquet run's revision is unknown, so it gets a local marker instead of a
    # false remote-provenance claim.
    source_details = {
        'dataset_url': HF_DATASET_URL,
        'paper_url': PAPER_URL,
        'note': 'Item-level evals run by Kensho with the Inspect AI framework (WILD paper).',
    }
    if revision:
        source_details['dataset_revision'] = revision
    else:
        source_details['dataset_source'] = (
            'local parquet file(s); source WILD-raw revision unknown (not stamped)')
    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        # keyed on the stable evaluation time, so reruns are idempotent
        evaluation_id=f'wild/{sanitized}/{task}/{eval_ts}',
        retrieved_timestamp=retrieved_ts,
        evaluation_timestamp=eval_ts,
        source_metadata=SourceMetadata(
            source_name=SOURCE_NAME,
            source_type='evaluation_run',
            source_organization_name=SOURCE_ORGANIZATION,
            source_organization_url=HF_DATASET_URL,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details=source_details,
        ),
        eval_library=EvalLibrary(
            name='inspect_ai', version='unknown',
            additional_details={'note': 'Run with the Inspect AI framework (WILD paper).'},
        ),
        model_info=ModelInfo(
            name=model, id=model, developer=developer,
            additional_details={'wild_model_id': model},
        ),
        evaluation_results=results,
    )
    return log, developer, model_slug


# instance-level (--include-instances)

def _sample_hash(raw: str, reference: list[str]) -> str:
    """Canonical cross-adapter sample hash: sha256 over canonical JSON of
    {"raw", "reference"}. Any other spelling stops joining with the other adapters'
    instances for the same item (`every_eval_ever/adapters/openeval`)."""
    payload = json.dumps({'raw': raw, 'reference': reference},
                         sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _split_conversation(raw: str | None) -> tuple[str, list[str]]:
    """Split the `conversation` column into (prompt, generation_turns).

    prompt = the user + system turns only -> input.raw: answer-free and
    model-independent, so it hashes identically across models. generation_turns = the
    assistant turn content(s) -> output.raw, the model's full generation."""
    if not raw:
        return '', []
    try:
        msgs = json.loads(raw)
    except (ValueError, TypeError):
        return str(raw), []
    if not isinstance(msgs, list):
        return str(raw), []
    prompt_parts: list[str] = []
    gen_parts: list[str] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        content = m.get('content')
        if not content:
            continue
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        role = m.get('role')
        if role in ('user', 'system'):
            prompt_parts.append(text)
        elif role == 'assistant':
            gen_parts.append(text)
    return '\n\n'.join(prompt_parts), gen_parts


def _primary_scorer(scores_json: str | None) -> tuple[str, str]:
    """Return (scorer_name, scored_answer) from the SAME scorer entry, so the name
    and value can't point at different scorers. The Inspect `scores` map is keyed by
    scorer name ('match', 'choice', 'model_graded_qa', …); WILD emits one scorer per
    item, and the first key is taken if several ever appear. `scored_answer` is the
    scorer's parsed answer, not the model's generation."""
    if scores_json:
        try:
            scores = json.loads(scores_json)
            if scores:
                name = str(next(iter(scores)))          # the scorer we attribute to
                val = scores[name]
                ans = val.get('answer') if isinstance(val, dict) else None
                return name, (str(ans) if ans else '')
        except (ValueError, TypeError, AttributeError, KeyError):
            pass
    return 'unknown', ''


def make_instance(row: dict, evaluation_id: str, model: str,
                  multi_subtask: bool) -> InstanceLevelEvaluationLog | None:
    """One instance record, or ``None`` for a row the aggregate also excluded."""
    task = row['task']
    subtask = row['subtask'] if row['subtask'] not in (None, '') else '_'
    # Attach to the finest-grain result: the leaf subtask when the benchmark is
    # split, else the lone overall (matching build_log's dedup, so the FK resolves).
    # Leaf-only is intentional: every item belongs to exactly one subtask, so linking
    # the overall as well would duplicate all ~7.5M rows for no new information.
    if multi_subtask:
        name, rid = f'wild.{task}.{subtask}', f'{task}::{subtask}'
    else:
        name, rid = f'wild.{task}', task
    score = item_score(row['score'])
    if score is None:
        return None            # reported by the aggregate pass; not scored here
    in_t, out_t = row['input_tokens'], row['output_tokens']
    scorer, scored_answer = _primary_scorer(row.get('scores'))
    # `source` names where the parsed answer came from, so it is never attributed
    # to output.raw, which holds the full generation.
    column_answer = str(row.get('answer') or '')
    extracted = column_answer or scored_answer
    if column_answer:
        attribution_source = 'answer'
    elif scored_answer:
        attribution_source = f'scores.{scorer}.answer'
    else:
        attribution_source = 'unavailable'
    prompt, generation = _split_conversation(row.get('conversation'))
    reference = [str(row.get('target') or '')]
    # A row with no assistant turn gets an empty list: substituting the parsed
    # answer would label scorer data as the model's output.
    output_raw = generation
    sample_hash = _sample_hash(prompt, reference)
    return InstanceLevelEvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=evaluation_id,
        model_id=model,
        evaluation_name=name,
        evaluation_result_id=rid,
        sample_id=str(row['item_id']),
        sample_hash=sample_hash,
        interaction_type=InteractionType.single_turn,
        input=Input(raw=prompt, reference=reference),
        output=Output(raw=output_raw),
        answer_attribution=[AnswerAttributionItem(
            turn_idx=0, source=attribution_source,
            extracted_value=extracted,
            extraction_method=scorer, is_terminal=True)],
        evaluation=Evaluation(score=score, is_correct=score == 1.0),
        # Omitted rather than zeroed when the row carries no usage.
        token_usage=(
            TokenUsage(input_tokens=int(in_t), output_tokens=int(out_t),
                       total_tokens=int(in_t) + int(out_t))
            if in_t is not None and out_t is not None else None),
        metadata={'stop_reason': str(row.get('stop_reason') or ''),
                  'subtask': str(subtask), 'scorer': scorer},
    )


def write_instances(parquet, limit_shards, models, staged_paths: dict,
                    eval_ids: dict, multi: set, max_instances: int | None,
                    revision: str | None = None
                    ) -> dict[tuple[str, str], tuple[str, int]]:
    """Stream item rows into the staged sidecars. Returns ``{key: (sha256, rows)}``.

    The digest is accumulated as the bytes are appended, so no sidecar is ever
    re-read or held whole in memory — WILD writes ~7.5M instance rows."""
    digests: dict[tuple[str, str], object] = {}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    written = 0
    reached_cap = False
    for _label, batch in iter_batches(parquet, INSTANCE_COLUMNS, limit_shards,
                                      revision):
        # group this batch's rows by (model, task) to bound open handles
        buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
        for i in range(len(batch['model'])):
            key = (batch['model'][i], batch['task'][i])
            if models and key[0] not in models:
                continue
            if key not in staged_paths:
                continue
            if max_instances is not None and written >= max_instances:
                reached_cap = True
                break
            row = {c: batch[c][i] for c in INSTANCE_COLUMNS}
            inst = make_instance(row, eval_ids[key], key[0], key in multi)
            if inst is None:      # unusable score: the aggregate skipped it too
                continue
            buckets[key].append(
                json.dumps(inst.model_dump(mode='json', exclude_none=True),
                           ensure_ascii=False))
            written += 1
        for key, lines in buckets.items():
            payload = ('\n'.join(lines) + '\n').encode('utf-8')
            staged_paths[key].parent.mkdir(parents=True, exist_ok=True)
            with staged_paths[key].open('ab') as fh:
                fh.write(payload)
            digests.setdefault(key, hashlib.sha256()).update(payload)
            counts[key] += len(lines)
        if reached_cap:
            break
    return {key: (digest.hexdigest(), counts[key])
            for key, digest in digests.items()}


# driver

FULL_SHA_RE = re.compile(r'[0-9a-f]{40}')


def resolve_base_output_dir(output_dir: Path) -> Path:
    """The datastore root above ``output_dir``, which has to be the collection dir.

    Publication derives ``<root>/wild/<developer>/<model>/`` itself, so it takes the
    root rather than the leaf. A path whose last component is not the collection
    would silently write beside the one asked for, and the replacement scan would
    read that other directory too."""
    if output_dir.name != COLLECTION:
        raise SystemExit(
            f'--output-dir must end in {COLLECTION!r}, the collection directory '
            f'publication writes into; got {output_dir}. Pass <root>/{COLLECTION} '
            f'(default {DEFAULT_OUTPUT_DIR}).'
        )
    return output_dir.parent


def resolve_source_revision(override: str | None,
                            parquet: list[str] | None) -> tuple[str | None, str | None]:
    """Pin a concrete commit as ``(revision, commit_timestamp)``, so both passes and
    any rerun read one snapshot. Local `--parquet` runs have no remote revision."""
    if parquet:                       # local files: no remote revision to pin
        return None, None
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(HF_REPO_ID, revision=override or HF_REVISION)
        commit_ts = None
        if info.lastModified:
            dt = info.lastModified
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            commit_ts = repr(dt.timestamp())
        return info.sha, commit_ts    # info.sha is the concrete commit SHA
    except Exception as exc:  # noqa: BLE001
        if override and FULL_SHA_RE.fullmatch(override):
            # A commit SHA is already the pin; the lookup only adds the commit
            # date, so losing it costs provenance, not reproducibility. Without
            # that date resolve_eval_timestamp requires --evaluation-timestamp.
            print(f'WARNING: could not read metadata for {HF_REPO_ID}@{override} '
                  f'({exc!r}); using it as given, with no commit date.')
            return override, None
        if override:
            raise SystemExit(
                f'could not resolve {HF_REPO_ID}@{override} to a commit ({exc!r}), '
                f'and {override!r} is not a commit SHA. A branch or tag can move '
                'between the aggregate pass and the instance pass, so it cannot '
                'stand in for the pin the lookup failed to produce. Pass the '
                '40-character commit SHA.'
            )
        raise SystemExit(
            f'could not resolve {HF_REPO_ID}@{HF_REVISION} to a concrete commit '
            f'({exc!r}). Reading the mutable {HF_REVISION!r} ref would make both '
            'passes and any rerun read possibly different data. Pass --revision '
            '<sha> to pin a snapshot explicitly.'
        )


def resolve_eval_timestamp(override: str | None,
                           commit_ts: str | None = None) -> str:
    """When the evaluation was RUN: an explicit override, else the pinned commit date.

    ``evaluation_id`` is keyed on this, so there is no now() fallback — it would give
    the same data a different identity on every run."""
    if override:
        return str(override)
    if commit_ts:
        return commit_ts
    raise SystemExit(
        'no evaluation timestamp is available: local --parquet runs carry no source '
        'commit date, and evaluation_id is keyed on this value, so falling back to '
        'now() would give identical reruns different logical identities. Pass '
        '--evaluation-timestamp <epoch>.'
    )


def logical_identity(evaluation_id: str) -> str:
    """The (model, benchmark) an ``evaluation_id`` is about, without its timestamp.

    ``evaluation_id`` ends in the source commit date, so re-pinning the dataset gives
    the same model and benchmark a new id. Replacement keys on this prefix instead, so
    a refresh supersedes its own earlier copy however the snapshot moved."""
    return evaluation_id.rsplit('/', 1)[0]


def superseded_records(base_output_dir: Path,
                       logs: list[EvaluationLog]) -> list[Path]:
    """Files a previous run published for the (model, benchmark) pairs in ``logs``.

    Filenames are fresh uuid4s, so publishing into a populated target adds a second
    copy of a record rather than replacing it. Each candidate is read for its own
    ``evaluation_id`` rather than matched on its path, because a path names only the
    model: the same directory holds the benchmarks this run does not cover, and a
    partial run must leave those alone. A sidecar travels with its aggregate."""
    wanted = {logical_identity(log.evaluation_id) for log in logs}
    directories = {
        datastore_output_dir(base_output_dir, COLLECTION, log.model_info.id,
                             log.model_info.developer)
        for log in logs
    }
    found: set[Path] = set()
    for directory in sorted(directories):
        for path in sorted(directory.glob('*.json')):
            try:
                published = json.loads(path.read_text())['evaluation_id']
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f'WARNING: {path} carries no readable evaluation_id '
                      f'({exc!r}); leaving it in place. If this run writes the same '
                      'model and benchmark, the directory will hold both.')
                continue
            if logical_identity(str(published)) not in wanted:
                continue
            found.add(path)
            sidecar = path.with_name(f'{path.stem}_samples.jsonl')
            if sidecar.exists():
                found.add(sidecar)
    return sorted(found)


def publish(logs: list[EvaluationLog], file_uuids: list[str],
            base_output_dir: Path, staging_root: Path) -> list[Path]:
    """Publish each aggregate together with its sidecar, one log at a time.

    The shared publisher buffers a batch's bytes before creating any file, so one call
    over WILD's ~1,700 logs would hold the whole sidecar corpus in memory. Anything
    already created is removed if a later log fails."""
    published: list[Path] = []
    try:
        for log, file_uuid in zip(logs, file_uuids):
            published.extend(publish_evaluation_logs(
                [log], base_output_dir, [file_uuid],
                staged_output_dir=staging_root, collection_override=COLLECTION))
    except Exception:
        for path in reversed(published):
            path.with_name(f'{path.stem}_samples.jsonl').unlink(missing_ok=True)
            path.unlink(missing_ok=True)
        raise
    return published


def run(args: argparse.Namespace) -> int:
    models = set(args.models) if args.models else None
    # Checked before any lookup or read, so a mistyped destination costs nothing.
    base_output_dir = resolve_base_output_dir(args.output_dir)
    revision, commit_ts = resolve_source_revision(args.revision, args.parquet)
    # retrieved = when this record was created (now); evaluation = when WILD ran it.
    eval_ts = resolve_eval_timestamp(args.evaluation_timestamp, commit_ts)
    retrieved_ts = str(args.retrieved_timestamp) if args.retrieved_timestamp else str(time.time())
    print(f'dataset_revision = {revision} | evaluation_timestamp = {eval_ts} '
          f'| retrieved_timestamp = {retrieved_ts}')

    groups, total_rows, failures = aggregate(args.parquet, args.limit_shards,
                                             models, revision)
    print(f'aggregated {len(groups)} (model, benchmark) groups '
          f'from {total_rows} item rows')
    if models:
        matched = {model for model, _task in groups}
        if not matched:
            raise SystemExit(
                f'--models selected {len(models)} model(s) and the source has none '
                f'of them: {", ".join(sorted(models))}. Nothing would be published, '
                'so the selection is treated as a mistake rather than an empty '
                'refresh.'
            )
        if missing := models - matched:
            print(f'WARNING: no source rows for {len(missing)} selected model(s): '
                  f'{", ".join(sorted(missing))}')

    keys = sorted(groups)
    logs: list[EvaluationLog] = []
    file_uuids: list[str] = []
    for model, task in keys:
        log, _developer, _model_slug = build_log(model, task, groups[(model, task)],
                                                eval_ts, retrieved_ts, revision)
        logs.append(log)
        file_uuids.append(str(uuid.uuid4()))

    # Checked before the instance pass so a rejected rerun costs nothing.
    superseded = superseded_records(base_output_dir, logs)
    if superseded and not args.replace_existing:
        raise SystemExit(
            f'{len(superseded)} file(s) under {args.output_dir} already hold the '
            f'model and benchmark pairs this run writes, e.g. {superseded[0]}. '
            'Filenames are fresh uuid4s, so writing now would add a second copy of '
            'each rather than replace it. Pass --replace-existing to replace them.'
        )

    with tempfile.TemporaryDirectory(prefix='eee-wild-publication-') as staging:
        staging_root = Path(staging)
        if args.include_instances:
            print('staging instance sidecars…')
            multi = {k for k, subs in groups.items()
                     if len([s for s in subs if s is not None]) > 1}
            staged_paths = {
                key: datastore_output_dir(staging_root, COLLECTION,
                                          log.model_info.id,
                                          log.model_info.developer)
                / f'{file_uuid}_samples.jsonl'
                for key, log, file_uuid in zip(keys, logs, file_uuids)}
            eval_ids = {key: log.evaluation_id for key, log in zip(keys, logs)}
            staged = write_instances(args.parquet, args.limit_shards, models,
                                     staged_paths, eval_ids, multi,
                                     args.max_instances, revision)
            for key, log, file_uuid in zip(keys, logs, file_uuids):
                if key not in staged:
                    continue
                checksum, rows = staged[key]
                log.detailed_evaluation_results = DetailedEvaluationResults(
                    format=Format.jsonl,
                    # The full repository-relative path, not the basename: it is what
                    # the schema, the publisher and the datastore gate all check.
                    file_path=datastore_repo_file_path(
                        COLLECTION, log.model_info.id, log.model_info.developer,
                        f'{file_uuid}_samples.jsonl'),
                    hash_algorithm=HashAlgorithm.sha256, checksum=checksum,
                    total_rows=rows)
            print(f'staged {sum(rows for _, rows in staged.values())} '
                  'instance records')

        result = SourceConversionResult(
            source_name=SOURCE_NAME, total_records=total_rows,
            records=logs, failures=failures)
        # Written before publication: it accounts for the conversion, so a
        # publication that raises must not take the record of what failed with it.
        if failures:
            print('Unconverted source rows: '
                  f'{save_failure_report(result, default_failure_report_path(args.output_dir))}')

        published = publish(logs, file_uuids, base_output_dir, staging_root)
        # Removed only once the replacement is in place, so an aborted run leaves
        # the previous refresh whole rather than a hole where it used to be.
        for path in superseded:
            path.unlink(missing_ok=True)

    print(f'wrote {len(published)} aggregate EvaluationLog(s) -> {args.output_dir}')
    result.raise_if_incomplete()
    return len(published)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Convert kensho/WILD-raw to Every Eval Ever.')
    # nargs='+' so a bare --parquet errors instead of silently converting all 15
    # remote shards.
    p.add_argument('--parquet', nargs='+', default=None,
                   help='Local parquet path(s); default fetches the HF shards.')
    p.add_argument('--output-dir', type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    p.add_argument('--limit-shards', type=int, default=None,
                   help='Only read the first N shards (for smoke runs).')
    # nargs='+' for the same reason as --parquet: a bare --models must error, not
    # parse to [] and quietly convert every model.
    p.add_argument('--models', nargs='+', default=None,
                   help='Filter to these model ids.')
    p.add_argument('--include-instances', action='store_true',
                   help='Also write per-item `<uuid>_samples.jsonl` instance sidecars.')
    p.add_argument('--max-instances', type=int, default=None,
                   help='Cap total instance rows written (smoke runs).')
    p.add_argument('--retrieved-timestamp', default=None,
                   help='Override the record-creation epoch (default: now).')
    p.add_argument('--evaluation-timestamp', default=None,
                   help='Override when the eval ran (default: the pinned commit date).')
    p.add_argument('--replace-existing', action='store_true',
                   help='Replace the files already published for the model and '
                        'benchmark pairs this run writes; anything else in the '
                        'output directory is left alone. Without it their presence '
                        'is an error, because a rerun would otherwise add a second '
                        'copy of each record rather than replace it.')
    p.add_argument('--revision', default=None,
                   help='Pin a specific kensho/WILD-raw commit SHA/tag for reproducible '
                        'reruns (default: resolve the current main commit and pin that).')
    return p.parse_args()


if __name__ == '__main__':
    written = run(parse_args())
    print(f'Wrote {written} WILD model×benchmark log(s).')
