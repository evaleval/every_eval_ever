#!/usr/bin/env python3
"""Convert aims-foundations/measurement-db into Every Eval Ever records."""

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
from dataclasses import dataclass, field
from datetime import timezone
from pathlib import Path
from typing import Any, Iterator

import requests

from every_eval_ever import eval_types as et
from every_eval_ever import instance_level_types as ilt
from every_eval_ever.converters.common.publication import (
    publish_evaluation_logs,
)
from every_eval_ever.helpers import SCHEMA_VERSION, raw_capture
from every_eval_ever.helpers.developer import get_developer
from every_eval_ever.helpers.io import (
    SourceConversionResult,
    SourceRecordExclusion,
    SourceRecordFailure,
    datastore_output_dir,
    datastore_repo_file_path,
    default_failure_report_path,
    save_failure_report,
)

HF_REPO_ID = 'aims-foundations/measurement-db'
HF_REVISION = 'main'
COLLECTION = 'measurement-db'
DEFAULT_OUTPUT_DIR = f'data/{COLLECTION}'
SOURCE_NAME = 'AI Measurement Data Bank'
SOURCE_ORGANIZATION = 'Stanford AIMS Lab'
HF_DATASET_URL = f'https://huggingface.co/datasets/{HF_REPO_ID}'
RESOLVER_URL = 'https://evaleval-entity-registry.hf.space/api/v1/resolve'
PARQUET_BATCH_ROWS = 65_536
TRACE_OUTPUT_CHAR_LIMIT = 10_000
FULL_SHA_RE = re.compile(r'[0-9a-f]{40}')
_SAFE_SLUG_RE = re.compile(r'[^A-Za-z0-9._-]+')

OVERLAPPING_BENCHMARKS = frozenset(
    'helm_afr helm_airbench helm_anthropic_redteam helm_bbq helm_bold '
    'helm_cleva helm_harmbench helm_real_toxicity_prompts '
    'helm_simple_safety_tests helm_thaiexam helm_xstest image2struct '
    'legalbench medhelm_ehr_sql medhelm_head_qa medhelm_med_mcqa '
    'medhelm_med_qa medhelm_medbullets medhelm_medcalc_bench medhelm_medec '
    'medhelm_medhallu medhelm_pubmed_qa medhelm_race_based_med mmlu '
    'truthfulqa_mc vhelm'.split()
)
WITHHELD_BENCHMARKS = frozenset(
    'arena_140k bountybench cybench imagenet_hard medarabiq '
    'mvt_image_difficulty osworld prediction_arena scivisagentbench '
    'situat3dchange swe_together visual_memory'.split()
)
PROVIDER_NAMESPACE_MAP: dict[str, str] = dict(
    item.split('=')
    for item in (
        '01.ai=01-ai|ai2=allenai|alibaba=Qwen|anthropic=anthropic|'
        'baichuan=baichuan-inc|bigcode=bigcode|cohere=CohereForAI|'
        'databricks=databricks|deepseek=deepseek-ai|deepseek-ai=deepseek-ai|'
        'eleutherai=EleutherAI|gemini=google|google=google|'
        'google deepmind=google|huggingface=HuggingFaceH4|ibm=ibm|lmsys=lmsys|'
        'meta=meta-llama|microsoft=microsoft|minimax=MiniMaxAI|'
        'ministral=mistralai|mistral=mistralai|mistral ai=mistralai|'
        'moonshot=moonshotai|moonshot ai=moonshotai|'
        'nous research=NousResearch|nvidia=nvidia|openai=openai|qwen=Qwen|'
        'snowflake=Snowflake|stability ai=stabilityai|'
        'technology innovation institute=tiiuae|thudm=zai-org|tii=tiiuae|'
        'together=togethercomputer|upstage=upstage|x-ai=xai|xai=xai|'
        'z-ai=zai-org|zhipu=zai-org|zhipu ai=zai-org'
    ).split('|')
)
MODEL_PREFIX_NAMESPACE_MAP: dict[str, str] = dict(
    item.split('=')
    for item in (
        'baize=project-baize|chatglm=zai-org|dolly=databricks|'
        'fastchat-t5=lmsys|guanaco=timdettmers|h2ogpt=h2oai|k2-think=LLM360|'
        'koala=bair|limo=GAIR|nous-hermes=NousResearch|oasst=OpenAssistant|'
        'openthinker=open-thoughts|qed-nano=Qwen|qwq=Qwen|rwkv=RWKV|'
        'stablelm=stabilityai|tulu=allenai|wizardlm=WizardLM'
    ).split('|')
)
_REDUNDANT_VENDOR_PREFIXES: tuple[tuple[str, str], ...] = (
    ('moonshotai', 'moonshot-'),
    ('zai-org', 'zhipu-'),
    ('amazon', 'amazon-'),
    ('openai', 'openai-'),
    ('mistralai', 'mistral-codestral'),
    ('mistralai', 'mistral-magistral'),
)
_ROUTING_PREFIXES: tuple[tuple[str, str], ...] = (
    ('accounts/fireworks/models/', 'fireworks'),
    ('fireworks_ai/', 'fireworks'),
    ('openrouter/', 'openrouter'),
    ('together_ai/', 'together'),
    ('vertex_ai/', 'vertex_ai'),
    ('bedrock/', 'bedrock'),
    ('models/', 'google'),
)
_CLOSED_API_FAMILIES = frozenset(
    'openai/gpt-3.5 openai/gpt-4 openai/gpt-4.1 openai/gpt-4o openai/gpt-5 '
    'openai/o1 openai/o3 openai/o4 anthropic/claude- google/gemini- '
    'google/palm- xai/grok-2 xai/grok-3 xai/grok-4'.split()
)
_VALID_DEPLOYMENT_TYPES = {'self_deployed', 'externally_managed', 'unknown'}
_VALID_AVAILABILITY_TYPES = {'open_weights', 'closed_weights', 'unknown'}
_RESOLVER_CACHE: dict[str, tuple[str | None, dict[str, str]]] = {}
_DEV_CASING_CACHE: dict[tuple[str, str], str] = {}
_MODEL_CASING_CACHE: dict[tuple[str, str, str], str] = {}


def _safe_component(value: str, fallback: str = 'item') -> str:
    cleaned = _SAFE_SLUG_RE.sub('-', str(value).strip()).strip('._-')
    if not cleaned or cleaned.lower() == 'unknown':
        return fallback
    return cleaned if cleaned[0].isalnum() else f'm-{cleaned}'


def _truncate_trace(text: str) -> str:
    if len(text) <= TRACE_OUTPUT_CHAR_LIMIT:
        return text
    dropped = len(text) - TRACE_OUTPUT_CHAR_LIMIT
    return f'{text[:TRACE_OUTPUT_CHAR_LIMIT]}\n[truncated {dropped} characters]'


def _extract_final_output(raw_trace: str) -> str:
    """Extract the final model output/answer from a trace or JSON trajectory wrapper."""
    s = raw_trace.strip()
    if s.startswith('{') and s.endswith('}'):
        try:
            d = json.loads(s)
        except ValueError:
            d = None
        if isinstance(d, dict):
            for k in ('final_output', 'published_output', 'output', 'response', 'model_patch', 'patch'):
                if d.get(k) is not None and str(d[k]).strip():
                    return _truncate_trace(str(d[k]).strip())
            rec = d.get('record')
            if isinstance(rec, dict):
                side = d.get('side')
                if side in ('modelA', 'modelB'):
                    val = rec.get('responseA' if side == 'modelA' else 'responseB')
                    if val is not None and str(val).strip():
                        return _truncate_trace(str(val).strip())
                for k in ('output', 'response', 'final_output', 'answer'):
                    if rec.get(k) is not None and str(rec[k]).strip():
                        return _truncate_trace(str(rec[k]).strip())
            elif isinstance(rec, list) and rec:
                ans = [
                    {k: v.get('answer') for k, v in item.items() if isinstance(v, dict) and 'answer' in v}
                    for item in rec if isinstance(item, dict)
                ]
                if any(ans):
                    return _truncate_trace(json.dumps(ans, separators=(',', ':')))
            ans_rec = d.get('answer', {}).get('record') if isinstance(d.get('answer'), dict) else None
            if isinstance(ans_rec, dict) and isinstance(ans_rec.get('choices'), list) and ans_rec['choices']:
                turns = ans_rec['choices'][0].get('turns') if isinstance(ans_rec['choices'][0], dict) else None
                if isinstance(turns, list) and turns:
                    return _truncate_trace('\n\n---\n\n'.join(str(t) for t in turns))
            res = d.get('result')
            if isinstance(res, dict):
                for k in ('final_output', 'output', 'response', 'taken_actions'):
                    if res.get(k) is not None and str(res[k]).strip():
                        val = res[k][-1] if isinstance(res[k], list) and res[k] else res[k]
                        return _truncate_trace(
                            json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list)) else str(val)
                        )
            for mkey in ('messages', 'traj', 'trajectory'):
                msgs = d.get(mkey)
                if isinstance(msgs, list) and msgs:
                    for m in reversed(msgs):
                        if isinstance(m, dict) and m.get('role') in ('assistant', 'model', None) and m.get('content'):
                            return _truncate_trace(str(m['content']).strip())
            judg = d.get('judgment')
            if isinstance(judg, dict) and judg.get('model_answer') is not None:
                return _truncate_trace(str(judg['model_answer']).strip())
    return _truncate_trace(raw_trace)


def _normalize_subject_candidate(raw: str) -> tuple[str, str | None]:
    cand = raw.strip().split(' / ', 1)[-1].strip() if ' / ' in raw else raw.strip()
    lower = cand.lower()
    for prefix, platform in _ROUTING_PREFIXES:
        if lower.startswith(prefix):
            return cand[len(prefix) :].strip(), platform
    return cand, None


def resolve_model_id(
    raw_value: str, *, enabled: bool = True, timeout: float = 10.0
) -> tuple[str | None, dict[str, str]]:
    if not enabled:
        return None, {'model_id_resolution': 'offline'}
    if raw_value in _RESOLVER_CACHE:
        return _RESOLVER_CACHE[raw_value]
    try:
        resp = requests.post(
            RESOLVER_URL, json={'raw_value': raw_value, 'entity_type': 'model'}, timeout=timeout
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        res = (None, {'model_id_resolution': 'unreachable', 'model_id_resolution_error': str(exc)[:160]})
        _RESOLVER_CACHE[raw_value] = res
        return res
    canonical = data.get('canonical_id')
    prov: dict[str, str] = {'model_id_resolution': 'registry'}
    for key in ('strategy', 'confidence', 'created_new', 'review_status'):
        if data.get(key) is not None:
            prov[f'model_id_{key}'] = str(data[key])
    detail = data.get('resolution_detail')
    if isinstance(detail, dict) and detail.get('hf_repo_id'):
        prov['hf_repo_id'] = str(detail['hf_repo_id'])
    res = (str(canonical) if canonical else None, prov)
    _RESOLVER_CACHE[raw_value] = res
    return res


def _canonicalize_path_casing(
    model_id: str, developer: str, *collection_dirs: Path
) -> tuple[str, str]:
    root_key = '|'.join(str(d) for d in collection_dirs)
    model_slug = model_id.split('/', 1)[1] if '/' in model_id else model_id
    for collection_dir in collection_dirs:
        if collection_dir.is_dir():
            for child in collection_dir.iterdir():
                if child.is_dir() and child.name.lower() == developer.lower():
                    developer = child.name
                    break
    developer = _DEV_CASING_CACHE.setdefault((root_key, developer.lower()), developer)
    for collection_dir in collection_dirs:
        dev_dir = collection_dir / developer
        if dev_dir.is_dir():
            for child in dev_dir.iterdir():
                if child.is_dir() and child.name.lower() == model_slug.lower():
                    model_slug = child.name
                    break
    model_slug = _MODEL_CASING_CACHE.setdefault(
        (root_key, developer.lower(), model_slug.lower()), model_slug
    )
    return f'{developer}/{model_slug}', developer


def resolve_subject_identity(
    subject_row: dict[str, Any], benchmark_id: str, *, registry_resolve: bool = True
) -> tuple[str, str, str, str | None, dict[str, str]] | None:
    subject_id = str(subject_row.get('subject_id') or '')
    display_name = str(
        subject_row.get('display_name') or subject_row.get('normalized_name') or subject_id
    ).strip()
    normalized_name = str(subject_row.get('normalized_name') or '').strip()
    provider_raw = str(subject_row.get('provider') or '').strip()

    candidate, platform = _normalize_subject_candidate(normalized_name or display_name)
    alt_display, alt_platform = _normalize_subject_candidate(display_name)
    platform = platform or alt_platform

    reg_id, reg_prov = resolve_model_id(candidate, enabled=registry_resolve)
    if reg_id is None and alt_display != candidate:
        reg_id, reg_prov = resolve_model_id(alt_display, enabled=registry_resolve)

    def _lookup_dev(name: str) -> str:
        if '/' in name and any(ch.isspace() or ch in '()' for ch in name.split('/', 1)[0]):
            return get_developer(name.replace('/', '-'))
        return get_developer(name)

    developer, model_slug = 'unknown', ''
    if reg_id and '/' in reg_id:
        reg_dev, reg_slug = reg_id.split('/', 1)
        hf_dev = _lookup_dev(reg_slug)
        use_hf = reg_prov.get('model_id_review_status') == 'draft' and hf_dev != 'unknown'
        raw_ns = hf_dev if use_hf else reg_dev
        developer = PROVIDER_NAMESPACE_MAP.get(raw_ns.lower(), raw_ns)
        model_slug = _safe_component(reg_slug, fallback=f'subject-{subject_id[:8]}')

    for prefix, ns in MODEL_PREFIX_NAMESPACE_MAP.items():
        if developer == 'unknown' and candidate.lower().startswith(prefix):
            developer = ns
            break
    for cand in (candidate, alt_display):
        if developer == 'unknown':
            developer = _lookup_dev(cand)
    if developer == 'unknown' and provider_raw:
        mapped = PROVIDER_NAMESPACE_MAP.get(provider_raw.lower())
        if mapped:
            developer = mapped
        elif provider_raw.lower() not in ('unknown', 'none', 'other', '(agent scaffold)', 'sup ai'):
            developer = _safe_component(provider_raw.lower(), 'unknown')
    if developer != 'unknown':
        developer = PROVIDER_NAMESPACE_MAP.get(developer.lower(), developer)
    if developer == 'unknown':
        return None

    if not model_slug:
        raw_slug = candidate
        if '/' in raw_slug and not any(ch.isspace() or ch in '()' for ch in raw_slug.split('/', 1)[0]):
            raw_slug = raw_slug.split('/', 1)[1]
        model_slug = _safe_component(raw_slug, fallback=f'subject-{subject_id[:8]}')
    for dev_match, v_pref in _REDUNDANT_VENDOR_PREFIXES:
        if developer == dev_match and model_slug.lower().startswith(v_pref) and len(model_slug) > len(v_pref):
            if v_pref in ('mistral-codestral', 'mistral-magistral'):
                model_slug = model_slug[len('mistral-') :]
            else:
                model_slug = model_slug[len(v_pref) :]
            break
    model_id = f'{developer}/{model_slug}'

    extra_raw = subject_row.get('subject_features_extra')
    extra_dict: dict[str, Any] = {}
    if isinstance(extra_raw, str) and extra_raw.strip():
        try:
            parsed = json.loads(extra_raw)
            if isinstance(parsed, dict):
                extra_dict = parsed
        except ValueError:
            pass

    dep = str(extra_dict.get('deployment_type') or 'unknown')
    deployment_type = dep if dep in _VALID_DEPLOYMENT_TYPES else 'unknown'
    avail = str(extra_dict.get('model_availability') or 'unknown')
    model_availability = avail if avail in _VALID_AVAILABILITY_TYPES else 'unknown'

    if any(model_id.lower().startswith(pref) for pref in _CLOSED_API_FAMILIES):
        model_availability = 'closed_weights' if model_availability == 'unknown' else model_availability
        deployment_type = 'externally_managed' if deployment_type == 'unknown' else deployment_type
    elif reg_prov.get('hf_repo_id') and model_availability == 'unknown':
        model_availability = 'open_weights'
    if platform and deployment_type == 'unknown':
        deployment_type = 'externally_managed'

    details: dict[str, str] = {
        'deployment_type': deployment_type,
        'model_availability': model_availability,
        'mdb_subject_id': subject_id,
        'mdb_benchmark_id': benchmark_id,
        'mdb_display_name': display_name,
        **reg_prov,
    }
    if normalized_name:
        details['mdb_normalized_name'] = normalized_name
    if provider_raw:
        details['mdb_provider'] = provider_raw
    for col in ('release_date', 'access_date', 'harness', 'reasoning_effort', 'harness_version'):
        val = subject_row.get(col)
        if val is not None and str(val).strip():
            details[f'mdb_{col}'] = str(val).strip()
    if isinstance(extra_raw, str) and extra_raw.strip():
        details['mdb_subject_features_extra'] = extra_raw.strip()
    return model_id, display_name, developer, platform, details


@dataclass
class ScaleSpec:
    kind: str
    min_score: float
    max_score: float
    lower_is_better: bool
    is_binary: bool
    raw_json: str


def parse_response_scale(raw_scale: str | dict[str, Any] | None) -> ScaleSpec:
    if raw_scale is None:
        return ScaleSpec('discrete', 0.0, 1.0, False, True, '{"kind":"discrete","values":[0,1]}')
    if isinstance(raw_scale, str):
        try:
            scale = json.loads(raw_scale)
        except ValueError:
            scale = {'kind': 'mixed'}
        raw_str = raw_scale
    elif isinstance(raw_scale, dict):
        scale, raw_str = raw_scale, json.dumps(raw_scale, sort_keys=True, separators=(',', ':'))
    else:
        scale, raw_str = {'kind': 'mixed'}, str(raw_scale)
    if not isinstance(scale, dict):
        scale = {'kind': 'mixed'}
    kind = str(scale.get('kind') or 'mixed')
    lower = str(scale.get('direction') or 'higher_is_better') == 'lower_is_better'
    for key in ('values', 'levels'):
        if isinstance(scale.get(key), list) and scale[key]:
            vals = [float(v) for v in scale[key]]
            return ScaleSpec(kind, min(vals), max(vals), lower, sorted(set(vals)) == [0.0, 1.0], raw_str)
    return ScaleSpec(kind, float(scale.get('min', 0.0)), float(scale.get('max', 1.0)), lower, False, raw_str)


@dataclass
class ConditionAgg:
    n_scored: int = 0
    n_ungraded: int = 0
    sum_score: float = 0.0
    sum_sq_score: float = 0.0
    min_observed: float | None = None
    max_observed: float | None = None
    trials: set[int] = field(default_factory=set)

    def add(self, value: float | None, trial: int | None) -> None:
        if trial is not None:
            self.trials.add(int(trial))
        if value is None or (isinstance(value, float) and math.isnan(value)):
            self.n_ungraded += 1
            return
        s = float(value)
        self.n_scored += 1
        self.sum_score += s
        self.sum_sq_score += s * s
        self.min_observed = s if self.min_observed is None else min(self.min_observed, s)
        self.max_observed = s if self.max_observed is None else max(self.max_observed, s)

    @property
    def mean(self) -> float:
        return self.sum_score / self.n_scored if self.n_scored else 0.0

    def standard_error(self, is_binary: bool) -> float:
        if self.n_scored <= 0:
            return 0.0
        p = self.mean
        if is_binary and 0.0 <= p <= 1.0:
            return math.sqrt(p * (1.0 - p) / self.n_scored)
        if self.n_scored == 1:
            return 0.0
        var = max(0.0, (self.sum_sq_score - (self.sum_score * self.sum_score) / self.n_scored) / (self.n_scored - 1))
        return math.sqrt(var / self.n_scored)


def _build_source_data(benchmark_row: dict[str, Any], n_samples: int):
    b_id = str(benchmark_row.get('benchmark_id') or 'unknown')
    b_name = str(benchmark_row.get('name') or b_id)
    source_url = str(benchmark_row.get('source_url') or '').strip()
    if 'huggingface.co/datasets/' in source_url:
        parts = source_url.split('huggingface.co/datasets/', 1)[1].strip('/').split('/')
        if len(parts) >= 2:
            return et.SourceDataHf(
                dataset_name=b_name, source_type='hf_dataset',
                hf_repo=f'{parts[0]}/{parts[1]}', samples_number=n_samples,
            )
    if source_url.startswith(('http://', 'https://')):
        return et.SourceDataUrl(
            dataset_name=b_name, source_type='url', url=[source_url], samples_number=n_samples
        )
    return et.SourceDataPrivate(
        dataset_name=b_name, source_type='other', samples_number=n_samples,
        additional_details={'mdb_benchmark_id': b_id},
    )


def _metric_id_and_kind(
    benchmark_id: str, response_type: str | None, is_binary: bool
) -> tuple[str, str, str, str]:
    rt = (response_type or '').strip().lower()
    if is_binary or rt == 'binary':
        return 'accuracy', 'accuracy', 'accuracy', 'proportion'
    rt_slug = _safe_component(rt or 'score', 'score')
    metric_id = f'measurement_db.{benchmark_id}.{rt_slug}'
    unit = (
        'proportion' if rt in ('win_rate', 'rate', 'fraction', 'error_present')
        else ('rating' if rt in ('likert', 'ordinal') else 'score')
    )
    return metric_id, (rt or 'score'), (rt or 'score'), unit


def _build_evaluation_result(
    benchmark_row: dict[str, Any], scale: ScaleSpec, condition_label: str | None, agg: ConditionAgg
) -> et.EvaluationResult:
    b_id = str(benchmark_row['benchmark_id'])
    b_name = str(benchmark_row.get('name') or b_id)
    response_type = str(benchmark_row.get('response_type') or 'score')
    metric_id, metric_name, metric_kind, metric_unit = _metric_id_and_kind(b_id, response_type, scale.is_binary)
    if condition_label is None:
        eval_name, result_id, level = f'measurement_db.{b_id}', b_id, 'overall'
    else:
        cond_slug = _safe_component(condition_label, fallback='condition')
        eval_name, result_id, level = f'measurement_db.{b_id}.{cond_slug}', f'{b_id}::{cond_slug}', 'condition'

    min_s = min(scale.min_score, agg.min_observed) if agg.min_observed is not None else scale.min_score
    max_s = max(scale.max_score, agg.max_observed) if agg.max_observed is not None else scale.max_score
    metric_details: dict[str, str] = {
        'n_scored': str(agg.n_scored), 'n_ungraded': str(agg.n_ungraded),
        'n_trials': str(len(agg.trials) or 1), 'response_type': response_type,
        'response_scale': scale.raw_json,
    }
    if condition_label is not None:
        metric_details['test_condition'] = condition_label
    score_kv: dict[str, str] = {'n_scored': str(agg.n_scored), 'n_ungraded': str(agg.n_ungraded)}
    if scale.is_binary:
        score_kv['n_correct'] = str(int(round(agg.sum_score)))

    return et.EvaluationResult(
        evaluation_result_id=result_id, evaluation_name=eval_name,
        source_data=_build_source_data(benchmark_row, agg.n_scored),
        metric_config=et.MetricConfig(
            evaluation_description=f'{b_name} ({eval_name}) curated in AI Measurement Data Bank.',
            metric_id=metric_id, metric_name=metric_name, metric_kind=metric_kind,
            metric_unit=metric_unit, lower_is_better=scale.lower_is_better,
            score_type=et.ScoreType.continuous, min_score=min_s, max_score=max_s,
            metric_parameters={'aggregation_level': level, 'aggregation': 'micro'},
            additional_details=metric_details,
        ),
        score_details=et.ScoreDetails(
            score=agg.mean, details=score_kv,
            uncertainty={
                'standard_error': {'value': agg.standard_error(scale.is_binary), 'method': 'analytic'},
                'num_samples': agg.n_scored,
            },
        ),
    )


def _sample_hash(raw: str, reference: list[str]) -> str:
    payload = json.dumps({'raw': raw, 'reference': reference}, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _parse_item_grading(grading_criterion_raw: str | None) -> tuple[list[str], str | None]:
    refs: list[str] = []
    rule: str | None = None
    if grading_criterion_raw:
        try:
            crit = json.loads(grading_criterion_raw)
            if isinstance(crit, dict):
                if crit.get('reference_answer') is not None and str(crit['reference_answer']).strip():
                    refs.append(str(crit['reference_answer']))
                if crit.get('rule') is not None and str(crit['rule']).strip():
                    rule = str(crit['rule'])
        except ValueError:
            pass
    return refs, rule


def _read_parquet_table(opener) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    with opener() as fh:
        return pq.read_table(fh).to_pylist()


def _iter_parquet_batches(
    opener, columns: list[str] | None = None, batch_size: int = PARQUET_BATCH_ROWS
) -> Iterator[list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    with opener() as fh:
        pf = pq.ParquetFile(fh)
        avail = set(pf.schema_arrow.names)
        cols = [c for c in columns if c in avail] if columns else None
        for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
            yield batch.to_pylist()


def _benchmark_table_openers(local_dir: Path | None, slug: str, revision: str | None) -> dict[str, Any]:
    if local_dir is not None:
        tdir = local_dir / 'formatted_tables' if (local_dir / 'formatted_tables').is_dir() else local_dir
        openers: dict[str, Any] = {}
        for name in ('benchmarks', 'subjects', 'items', 'responses', 'traces', 'assets'):
            p = tdir / f'{name}.parquet'
            if not p.is_file() and name == 'responses':
                p = tdir / 'response.parquet'
            if p.is_file():
                openers[name] = lambda path=p: path.open('rb')
        return openers

    from huggingface_hub import HfApi, HfFileSystem

    api, fs, rev = HfApi(), HfFileSystem(), revision or HF_REVISION
    entries = list(api.list_repo_tree(HF_REPO_ID, repo_type='dataset', revision=rev, path_in_repo=slug))
    folder = f'{slug}/formatted_tables'
    modern = any(e.path == folder for e in entries)
    if modern:
        entries = list(api.list_repo_tree(HF_REPO_ID, repo_type='dataset', revision=rev, path_in_repo=folder, recursive=True))
    prefix = (folder if modern else slug) + '/'
    openers = {}
    for entry in entries:
        if hasattr(entry, 'blob_id') and entry.path.endswith('.parquet'):
            stem = Path(entry.path[len(prefix) :]).stem
            stem = 'responses' if stem == 'response' else stem
            if stem in ('benchmarks', 'subjects', 'items', 'responses', 'traces', 'assets'):
                hf_path = f'datasets/{HF_REPO_ID}/{entry.path}'
                openers[stem] = lambda p=hf_path, r=rev: fs.open(p, revision=r)
    return openers


def convert_one_benchmark(
    slug: str, openers: dict[str, Any], *, models_filter: set[str] | None,
    eval_ts: str, retrieved_ts: str, revision: str | None,
    include_instances: bool, max_instances: int | None,
    registry_resolve: bool, staging_root: Path, base_output_dir: Path | None = None,
) -> tuple[list[et.EvaluationLog], list[str], int, list[SourceRecordFailure], list[SourceRecordExclusion]]:
    for req in ('benchmarks', 'subjects', 'items', 'responses'):
        if req not in openers:
            raise ValueError(f'benchmark {slug!r} is missing required {req}.parquet')
    benchmarks_rows = _read_parquet_table(openers['benchmarks'])
    if not benchmarks_rows:
        raise ValueError(f'benchmark {slug!r} has an empty benchmarks.parquet')
    benchmark_row = benchmarks_rows[0]
    benchmark_id = str(benchmark_row.get('benchmark_id') or slug)

    if str(benchmark_row.get('release') or 'public') == 'withheld':
        return ([], [], 1, [], [SourceRecordExclusion(
            source_ref=f'{benchmark_id}#benchmarks.parquet',
            reason='benchmark marked release=withheld in measurement-db',
        )])

    scale = parse_response_scale(benchmark_row.get('response_scale'))
    subjects_by_id = {str(r['subject_id']): r for r in _read_parquet_table(openers['subjects']) if r.get('subject_id')}
    items_by_id = (
        {str(r['item_id']): r for r in _read_parquet_table(openers['items']) if r.get('item_id')}
        if include_instances else {}
    )

    subject_aggs: dict[str, dict[str | None, ConditionAgg]] = defaultdict(lambda: defaultdict(ConditionAgg))
    subject_rows_for_instances: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_instance_keys: set[tuple[str, str, str | None, str | None]] = set()
    wanted_response_ids: set[str] = set()
    failures: list[SourceRecordFailure] = []
    exclusions: list[SourceRecordExclusion] = []
    total_rows = 0

    resp_cols = ['response_id', 'subject_id', 'item_id', 'response', 'trial', 'test_condition']
    if include_instances:
        resp_cols.append('interactors')

    for batch in _iter_parquet_batches(openers['responses'], columns=resp_cols):
        for row in batch:
            total_rows += 1
            subject_id = str(row.get('subject_id') or '')
            if not subject_id or subject_id not in subjects_by_id:
                failures.append(SourceRecordFailure(
                    source_ref=f'{benchmark_id}#response_id={row.get("response_id")}',
                    reason=f'response references unknown subject_id {subject_id!r}',
                ))
                continue
            subj = subjects_by_id[subject_id]
            if models_filter and not ({subject_id, str(subj.get('display_name') or ''), str(subj.get('normalized_name') or '')} & models_filter):
                continue

            val, trial, cond_raw = row.get('response'), row.get('trial'), row.get('test_condition')
            cond = str(cond_raw).strip() if cond_raw is not None and str(cond_raw).strip() not in ('', '_') else None
            subject_aggs[subject_id][None].add(val, trial)
            if cond is not None:
                subject_aggs[subject_id][cond].add(val, trial)

            if val is None or (isinstance(val, float) and math.isnan(val)):
                exclusions.append(SourceRecordExclusion(
                    source_ref=f'{benchmark_id}#{subject_id}/{row.get("item_id")}/{row.get("response_id")}',
                    reason='ungraded response (response is null in responses.parquet)',
                ))
            elif include_instances and (max_instances is None or len(subject_rows_for_instances[subject_id]) < max_instances):
                inst_key = (
                    subject_id,
                    str(row.get('item_id') or ''),
                    cond,
                    str(row.get('interactors') or '') or None,
                )
                if inst_key in seen_instance_keys:
                    continue
                seen_instance_keys.add(inst_key)
                subject_rows_for_instances[subject_id].append(row)
                if row.get('response_id'):
                    wanted_response_ids.add(str(row['response_id']))

    traces_by_response_id: dict[str, str] = {}
    if include_instances and 'traces' in openers and wanted_response_ids:
        for tbatch in _iter_parquet_batches(openers['traces'], columns=['response_id', 'trace']):
            for r in tbatch:
                rid = str(r.get('response_id') or '')
                if rid in wanted_response_ids and r.get('trace') is not None:
                    traces_by_response_id[rid] = _extract_final_output(str(r['trace']))

    logs: list[et.EvaluationLog] = []
    file_uuids: list[str] = []
    for subject_id in sorted(subject_aggs):
        aggs = subject_aggs[subject_id]
        overall = aggs[None]
        if overall.n_scored == 0:
            exclusions.append(SourceRecordExclusion(
                source_ref=f'{benchmark_id}#subject_id={subject_id}',
                reason='subject has 0 scored responses in responses.parquet',
            ))
            continue

        subj = subjects_by_id[subject_id]
        resolved = resolve_subject_identity(subj, benchmark_id, registry_resolve=registry_resolve)
        if resolved is None:
            prov_low = str(subj.get('provider') or '').strip().lower()
            norm_low = str(subj.get('normalized_name') or '').strip().lower()
            disp_low = str(subj.get('display_name') or '').strip().lower()
            if prov_low == '(agent scaffold)' or norm_low.startswith('agent:') or 'ensemble' in disp_low:
                exclusions.append(SourceRecordExclusion(
                    source_ref=f'{benchmark_id}#subject_id={subject_id}',
                    reason='agent scaffold or ensemble entry with no disclosed base model',
                ))
            else:
                failures.append(SourceRecordFailure(
                    source_ref=f'{benchmark_id}#subject_id={subject_id}',
                    reason=f'cannot resolve developer namespace for subject {subj.get("display_name")!r}',
                ))
            continue

        model_id, display_name, developer, platform, model_details = resolved
        c_dirs = (
            (base_output_dir / COLLECTION, staging_root / COLLECTION)
            if base_output_dir is not None else (staging_root / COLLECTION,)
        )
        model_id, developer = _canonicalize_path_casing(model_id, developer, *c_dirs)
        real_conds = sorted(k for k in aggs if k is not None and aggs[k].n_scored > 0)
        multi_condition = len(real_conds) > 1
        results = [_build_evaluation_result(benchmark_row, scale, None, overall)]
        if multi_condition:
            results.extend(_build_evaluation_result(benchmark_row, scale, c, aggs[c]) for c in real_conds)

        source_details: dict[str, str] = {
            'dataset_url': HF_DATASET_URL, 'mdb_benchmark_id': benchmark_id, 'mdb_schema_version': '3',
        }
        if benchmark_row.get('paper_url'):
            source_details['paper_url'] = str(benchmark_row['paper_url'])
        if benchmark_row.get('source_url'):
            source_details['upstream_source_url'] = str(benchmark_row['source_url'])
        source_details['dataset_revision' if revision else 'dataset_source'] = (
            revision or 'local measurement-db benchmark tables'
        )

        evaluation_id = f'measurement-db/{benchmark_id}/{subject_id}/{eval_ts}'
        file_uuid = str(uuid.uuid4())
        log = et.EvaluationLog(
            schema_version=SCHEMA_VERSION, evaluation_id=evaluation_id,
            retrieved_timestamp=retrieved_ts, evaluation_timestamp=eval_ts,
            source_metadata=et.SourceMetadata(
                source_name=SOURCE_NAME, source_type='evaluation_run',
                source_organization_name=SOURCE_ORGANIZATION, source_organization_url=HF_DATASET_URL,
                evaluator_relationship=et.EvaluatorRelationship.third_party, additional_details=source_details,
            ),
            eval_library=et.EvalLibrary(
                name=str(subj.get('harness') or 'unknown').strip() or 'unknown',
                version=str(subj.get('harness_version') or 'unknown').strip() or 'unknown',
                additional_details={'note': 'Curated in aims-foundations/measurement-db.'},
            ),
            model_info=et.ModelInfo(
                name=display_name, id=model_id, developer=developer,
                inference_platform=platform, additional_details=model_details,
            ),
            evaluation_results=results,
        )

        if include_instances:
            sidecar_path = datastore_output_dir(staging_root, COLLECTION, model_id, developer) / f'{file_uuid}_samples.jsonl'
            checksum, n_written = _write_subject_instances(
                sidecar_path=sidecar_path, response_rows=subject_rows_for_instances[subject_id],
                items_by_id=items_by_id, traces_by_response_id=traces_by_response_id,
                benchmark_id=benchmark_id, scale=scale, evaluation_id=evaluation_id,
                model_id=model_id, multi_condition=multi_condition, max_instances=max_instances,
            )
            if n_written > 0:
                log.detailed_evaluation_results = et.DetailedEvaluationResults(
                    format=et.Format.jsonl,
                    file_path=datastore_repo_file_path(COLLECTION, model_id, developer, f'{file_uuid}_samples.jsonl'),
                    hash_algorithm=et.HashAlgorithm.sha256, checksum=checksum, total_rows=n_written,
                )
        logs.append(log)
        file_uuids.append(file_uuid)

    return logs, file_uuids, total_rows, failures, exclusions


def _write_subject_instances(
    *, sidecar_path: Path, response_rows: list[dict[str, Any]], items_by_id: dict[str, dict[str, Any]],
    traces_by_response_id: dict[str, str], benchmark_id: str, scale: ScaleSpec,
    evaluation_id: str, model_id: str, multi_condition: bool, max_instances: int | None,
) -> tuple[str, int]:
    lines: list[str] = []
    for row in response_rows:
        if max_instances is not None and len(lines) >= max_instances:
            break
        val = row.get('response')
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        score = float(val)
        item_id = str(row.get('item_id') or '')
        item = items_by_id.get(item_id, {})
        raw_item_id = str(item.get('raw_item_id') or item_id)
        full_content = str(item.get('content') or raw_item_id or item_id)
        references, rule_text = _parse_item_grading(item.get('grading_criterion'))

        cond_raw = row.get('test_condition')
        cond_key = str(cond_raw).strip() if cond_raw is not None and str(cond_raw).strip() not in ('', '_') else None
        if multi_condition and cond_key is not None:
            cond_slug = _safe_component(cond_key, fallback='condition')
            eval_name, result_id = f'measurement_db.{benchmark_id}.{cond_slug}', f'{benchmark_id}::{cond_slug}'
        else:
            eval_name, result_id = f'measurement_db.{benchmark_id}', benchmark_id

        response_id = str(row.get('response_id') or '')
        trace_text = traces_by_response_id.get(response_id)
        output_raw = [_truncate_trace(trace_text)] if trace_text and trace_text.strip() else [str(val)]

        meta: dict[str, str] = {
            'mdb_item_id': item_id, 'mdb_response_id': response_id,
            'mdb_trial': str(row.get('trial') if row.get('trial') is not None else 1),
        }
        if cond_key is not None:
            meta['test_condition'] = cond_key
        if row.get('interactors') is not None and str(row.get('interactors')).strip():
            meta['interactors'] = str(row.get('interactors')).strip()
        if rule_text:
            meta['grading_rule'] = rule_text[:500]

        is_correct = math.isclose(score, scale.min_score if scale.lower_is_better else scale.max_score)
        inst = ilt.InstanceLevelEvaluationLog(
            schema_version=SCHEMA_VERSION, evaluation_id=evaluation_id, model_id=model_id,
            evaluation_name=eval_name, evaluation_result_id=result_id, sample_id=raw_item_id,
            sample_hash=_sample_hash(full_content, references),
            interaction_type=ilt.InteractionType.single_turn,
            input=ilt.Input(raw=full_content, reference=references),
            output=ilt.Output(raw=output_raw), answer_attribution=[],
            evaluation=ilt.Evaluation(score=score, is_correct=is_correct), metadata=meta,
        )
        lines.append(
            json.dumps(inst.model_dump(mode='json', exclude_none=True), ensure_ascii=False)
            .replace('\x85', '\\u0085').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
        )

    if not lines:
        return '', 0
    payload = ('\n'.join(lines) + '\n').encode('utf-8')
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(lines)


def discover_local_benchmark_dirs(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_dir():
            raise SystemExit(f'--benchmark-dir does not exist: {path}')
        if (path / 'formatted_tables').is_dir() or (path / 'benchmarks.parquet').is_file():
            found.append(path)
        else:
            children = sorted(
                c for c in path.iterdir()
                if c.is_dir() and not c.name.startswith('_')
                and ((c / 'formatted_tables').is_dir() or (c / 'benchmarks.parquet').is_file())
            )
            if not children:
                raise SystemExit(f'no benchmark tables found under --benchmark-dir {path}')
            found.extend(children)
    return found


def selected_benchmarks(
    candidates: list[str], requested: list[str] | None, include_overlapping: bool
) -> list[str]:
    pool = requested if requested else candidates
    out: list[str] = []
    for slug in pool:
        if slug in WITHHELD_BENCHMARKS:
            if requested:
                raise SystemExit(f'benchmark {slug!r} is marked release=withheld in measurement-db')
            continue
        if not include_overlapping and slug in OVERLAPPING_BENCHMARKS:
            if requested:
                raise SystemExit(
                    f'benchmark {slug!r} is in OVERLAPPING_BENCHMARKS; pass --include-overlapping to convert it'
                )
            continue
        out.append(slug)
    return out


def list_remote_benchmarks(
    requested: list[str] | None, include_overlapping: bool, revision: str | None
) -> list[str]:
    if requested:
        return selected_benchmarks(requested, requested, include_overlapping)
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(
        HF_REPO_ID, repo_type='dataset', revision=revision or HF_REVISION, recursive=False
    )
    candidates = sorted(
        e.path for e in entries if not hasattr(e, 'blob_id') and not e.path.startswith(('.', '_'))
    )
    return selected_benchmarks(candidates, None, include_overlapping)


def resolve_base_output_dir(output_dir: Path) -> Path:
    if output_dir.name != COLLECTION:
        raise SystemExit(f'--output-dir must end in {COLLECTION!r}, got {output_dir}.')
    return output_dir.parent


def resolve_source_revision(override: str | None, local_mode: bool) -> tuple[str | None, str | None]:
    if local_mode:
        return None, None
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(HF_REPO_ID, revision=override or HF_REVISION)
        dt = info.lastModified
        commit_ts = repr((dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()) if dt else None
        return info.sha, commit_ts
    except Exception as exc:  # noqa: BLE001
        if override and FULL_SHA_RE.fullmatch(override):
            return override, None
        raise SystemExit(f'could not resolve {HF_REPO_ID}@{override or HF_REVISION} ({exc!r}).')


def resolve_eval_timestamp(override: str | None, commit_ts: str | None = None) -> str:
    if override or commit_ts:
        return str(override or commit_ts)
    raise SystemExit('no evaluation timestamp available; pass --evaluation-timestamp <epoch>.')


def emit_source_version(args: argparse.Namespace) -> str:
    if args.benchmark_dir:
        hasher = hashlib.sha256()
        for bdir in discover_local_benchmark_dirs(args.benchmark_dir):
            tdir = bdir / 'formatted_tables' if (bdir / 'formatted_tables').is_dir() else bdir
            parts = [f'{p.name}:{p.stat().st_size}' for p in sorted(tdir.glob('*.parquet'))]
            hasher.update(f'{bdir.name}:{"|".join(parts)}\n'.encode('utf-8'))
        return hasher.hexdigest()
    revision, _ = resolve_source_revision(args.revision, local_mode=False)
    return str(revision)


def logical_identity(evaluation_id: str) -> str:
    return evaluation_id.rsplit('/', 1)[0]


def superseded_records(base_output_dir: Path, logs: list[et.EvaluationLog]) -> list[Path]:
    wanted = {logical_identity(log.evaluation_id) for log in logs}
    coll_dir = base_output_dir / COLLECTION
    if not coll_dir.is_dir():
        return []
    found: set[Path] = set()
    for path in sorted(coll_dir.rglob('*.json')):
        if path.name.startswith('_'):
            continue
        try:
            published = json.loads(path.read_text(encoding='utf-8'))['evaluation_id']
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if logical_identity(str(published)) in wanted:
            found.add(path)
            sidecar = path.with_name(f'{path.stem}_samples.jsonl')
            if sidecar.exists():
                found.add(sidecar)
    return sorted(found)


def publish(
    logs: list[et.EvaluationLog], file_uuids: list[str], base_output_dir: Path, staging_root: Path
) -> list[Path]:
    published: list[Path] = []
    try:
        for log, file_uuid in zip(logs, file_uuids):
            published.extend(publish_evaluation_logs(
                [log], base_output_dir, [file_uuid], staged_output_dir=staging_root, collection_override=COLLECTION
            ))
    except Exception:
        for path in reversed(published):
            path.with_name(f'{path.stem}_samples.jsonl').unlink(missing_ok=True)
            path.unlink(missing_ok=True)
        raise
    return published


def run(args: argparse.Namespace) -> int:
    if args.emit_source_version:
        print(emit_source_version(args))
        return 0
    base_output_dir = resolve_base_output_dir(args.output_dir)
    local_mode = bool(args.benchmark_dir)
    revision, commit_ts = resolve_source_revision(args.revision, local_mode=local_mode)
    eval_ts = resolve_eval_timestamp(args.evaluation_timestamp, commit_ts)
    retrieved_ts = str(args.retrieved_timestamp) if args.retrieved_timestamp else str(time.time())
    models_filter = set(args.models) if args.models else None

    if raw_capture.active_sink() is not None and revision:
        raw_capture.record_pointer(
            kind='hf_dataset', reference=HF_REPO_ID, revision=revision,
            url=HF_DATASET_URL, label=SOURCE_NAME, revision_required=True,
        )

    if local_mode:
        local_dirs = discover_local_benchmark_dirs(args.benchmark_dir)
        if args.benchmarks:
            local_dirs = [d for d in local_dirs if d.name in set(args.benchmarks)]
        work_items: list[tuple[str, Path | None]] = [(d.name, d) for d in local_dirs]
    else:
        work_items = [(s, None) for s in list_remote_benchmarks(args.benchmarks, args.include_overlapping, revision)]

    all_logs: list[et.EvaluationLog] = []
    all_uuids: list[str] = []
    all_failures: list[SourceRecordFailure] = []
    all_exclusions: list[SourceRecordExclusion] = []
    total_rows = 0

    with tempfile.TemporaryDirectory(prefix='eee-mdb-publication-') as staging:
        staging_root = Path(staging)
        for slug, local_dir in work_items:
            openers = _benchmark_table_openers(local_dir, slug, revision)
            logs, file_uuids, rows, failures, exclusions = convert_one_benchmark(
                slug, openers, models_filter=models_filter, eval_ts=eval_ts,
                retrieved_ts=retrieved_ts, revision=revision,
                include_instances=args.include_instances, max_instances=args.max_instances,
                registry_resolve=not args.no_registry_resolve, staging_root=staging_root,
                base_output_dir=base_output_dir,
            )
            all_logs.extend(logs)
            all_uuids.extend(file_uuids)
            all_failures.extend(failures)
            all_exclusions.extend(exclusions)
            total_rows += rows

        superseded = superseded_records(base_output_dir, all_logs)
        if superseded and not args.replace_existing:
            raise SystemExit(
                f'{len(superseded)} file(s) under {args.output_dir} already hold '
                f'records this run writes, e.g. {superseded[0]}. Pass --replace-existing to replace them.'
            )
        result = SourceConversionResult(
            source_name=SOURCE_NAME, total_records=total_rows, records=all_logs,
            failures=all_failures, exclusions=all_exclusions,
        )
        if all_failures or all_exclusions:
            save_failure_report(result, default_failure_report_path(args.output_dir))
        published = publish(all_logs, all_uuids, base_output_dir, staging_root)
        for path in superseded:
            path.unlink(missing_ok=True)

    print(f'wrote {len(published)} aggregate EvaluationLog(s) -> {args.output_dir}')
    result.raise_if_incomplete()
    return len(published)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Convert aims-foundations/measurement-db to Every Eval Ever.')
    p.add_argument('--output-dir', type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    p.add_argument('--benchmark-dir', nargs='+', type=Path, default=None)
    p.add_argument('--benchmarks', nargs='+', default=None)
    p.add_argument('--include-overlapping', action='store_true')
    p.add_argument('--models', nargs='+', default=None)
    p.add_argument('--include-instances', action='store_true')
    p.add_argument('--max-instances', type=int, default=None)
    p.add_argument('--no-registry-resolve', action='store_true')
    p.add_argument('--emit-source-version', action='store_true')
    p.add_argument('--retrieved-timestamp', default=None)
    p.add_argument('--evaluation-timestamp', default=None)
    p.add_argument('--replace-existing', action='store_true')
    p.add_argument('--revision', default=None)
    return p.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
