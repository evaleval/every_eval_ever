#!/usr/bin/env python3
"""Convert the Open Medical-LLM Leaderboard results into Every Eval Ever aggregate logs.

Data source (HuggingFace dataset): openlifescienceai/results
  Layout: <developer>/<model>/results_*.json  (lm-evaluation-harness output format)
  Backs the Space: openlifescienceai/open_medical_llm_leaderboard

One EvaluationLog per model (developer/model), with one EvaluationResult per medical
benchmark (accuracy, proportion 0-1, higher is better). Aggregates only; no per-item data.

Run from the EEE repo dir:
    uv run python -m every_eval_ever.adapters.open_medical_llm.adapter --output-dir /tmp/eee-omll [--limit N]
    uv run python -m every_eval_ever validate /tmp/eee-omll
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

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
    require_finite_number,
    save_evaluation_logs,
    save_failure_report,
)
from every_eval_ever.helpers.io import datastore_path_components

REPO = "openlifescienceai/results"
RESOLVE = "https://huggingface.co/datasets/openlifescienceai/results/resolve/main/"
TREE = "https://huggingface.co/api/datasets/openlifescienceai/results/tree/main?recursive=true"
LEADERBOARD_SPACE = "https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard"
SRC = "open-medical-llm-leaderboard"

# Hosted eval-card-registry resolver (public HF Space, no auth). Maps a raw HF
# ``developer/model`` id to the shared canonical id. See resolve_model_id.
RESOLVER_URL = "https://evaleval-entity-registry.hf.space/api/v1/resolve"
# below this the resolver's alias is treated as unverified (flag for review):
RESOLVE_CONFIDENCE_FLOOR = 0.9

# HF resolves a renamed/aliased repo id to its current one on GET. Used only to
# adjudicate a path/config disagreement — see evaluated_model_repo.
HF_MODEL_API = "https://huggingface.co/api/models/"

# task name -> (human display name, verified HF dataset repo)
TASKS = {
    "medmcqa": ("MedMCQA", "openlifescienceai/medmcqa"),
    "medqa_4options": ("MedQA (USMLE, 4 options)", "openlifescienceai/MedQA-USMLE-4-options-hf"),
    "pubmedqa": ("PubMedQA", "openlifescienceai/pubmedqa"),
    "mmlu_anatomy": ("MMLU: Anatomy", "openlifescienceai/mmlu_anatomy"),
    "mmlu_clinical_knowledge": ("MMLU: Clinical Knowledge", "openlifescienceai/mmlu_clinical_knowledge"),
    "mmlu_college_biology": ("MMLU: College Biology", "openlifescienceai/mmlu_college_biology"),
    "mmlu_college_medicine": ("MMLU: College Medicine", "openlifescienceai/mmlu_college_medicine"),
    "mmlu_medical_genetics": ("MMLU: Medical Genetics", "openlifescienceai/mmlu_medical_genetics"),
    "mmlu_professional_medicine": ("MMLU: Professional Medicine", "openlifescienceai/mmlu_professional_medicine"),
}

# Capture optional fractional seconds so two runs of the same model in the same
# whole second stay distinguishable (see parse_ts / make_log ts_token). Also allow
# an ISO 'T' separator alongside the leaderboard's space/underscore form.
TS_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[ _T](\d{2})[:_-](\d{2})[:_-](\d{2})(?:[.,](\d{1,6}))?"
)


def stringify(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, separators=(",", ":"))
    return str(v)


def clean_details(d: dict) -> dict:
    return {k: stringify(v) for k, v in d.items() if v is not None}


def parse_ts(path: str):
    m = TS_RE.search(path)
    if not m:
        return None
    y, mo, da, h, mi, s = (int(g) for g in m.groups()[:6])
    frac = m.group(7)
    micros = int(frac.ljust(6, "0")[:6]) if frac else 0  # right-pad ms->us, cap at us
    try:
        return datetime(y, mo, da, h, mi, s, micros, tzinfo=timezone.utc)
    except ValueError:
        return None


def fetch_json(path: str) -> dict:
    url = RESOLVE + urllib.parse.quote(path)
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read())


def _next_link(link_header: str | None) -> str | None:
    """Extract the ``rel="next"`` URL from an RFC-5988 ``Link`` header (HF tree
    pagination), or ``None`` when there is no next page."""
    if not link_header:
        return None
    for part in link_header.split(","):
        seg = part.strip()
        if 'rel="next"' not in seg:
            continue
        lt, gt = seg.find("<"), seg.find(">")
        if lt != -1 and gt != -1:
            return seg[lt + 1 : gt]
    return None


def _iter_tree_pages(url: str):
    """Yield tree entries across ALL pages. The HF tree API caps a page at ~1000
    entries and points at the next via a ``Link: <...>; rel="next"`` header — a
    single unpaginated GET silently truncates large repos."""
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "eee-omll-adapter"})
        with urllib.request.urlopen(req, timeout=60) as r:
            yield from json.loads(r.read())
            url = _next_link(r.headers.get("Link"))


def list_result_files() -> list[str]:
    out = []
    for x in _iter_tree_pages(TREE):
        if x.get("type") != "file":
            continue
        p = x["path"]
        segs = p.split("/")
        # skip hidden dirs (e.g. .ipynb_checkpoints) and jupyter checkpoint files
        if any(s.startswith(".") for s in segs) or segs[-1].endswith("-checkpoint.json"):
            continue
        if p.endswith(".json") and segs[-1].startswith("results_"):
            out.append(p)
    return out


def resolve_model_id(raw_repo: str, *, enabled: bool = True, timeout: float = 15.0) -> tuple[str, dict]:
    """Canonicalize an HF ``developer/model`` id via the hosted eval-card-registry
    resolver, for use as ``model_info.id``. Returns ``(model_id, provenance)``.

    Never fatal: the opt-out and any network error fall back to the path id and
    record which (``offline`` / ``unreachable``). The resolver's last strategy is
    to auto-create a draft, so ``created_new`` / ``review_status`` / ``confidence``
    come back in the provenance for ``_needs_registry_review`` to judge.
    """
    if not enabled:
        return raw_repo, {"model_id_resolution": "offline"}
    try:
        resp = requests.post(
            RESOLVER_URL,
            json={"raw_value": raw_repo, "entity_type": "model"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001 — resolution is best-effort, never fatal
        return raw_repo, {"model_id_resolution": "unreachable",
                          "model_id_resolution_error": str(e)[:200]}
    canonical = data.get("canonical_id") or raw_repo
    return canonical, {
        "model_id_resolution": "registry",
        "model_id_resolution_strategy": data.get("strategy"),
        "model_id_resolution_confidence": data.get("confidence"),
        "model_id_created_new": data.get("created_new"),
        "model_id_review_status": data.get("review_status"),
    }


def _needs_registry_review(prov: dict | None) -> bool:
    """True when a resolved id is not a confident, already-reviewed canonical:
    unreachable resolver, a freshly auto-created draft, a non-``reviewed`` status,
    or confidence below the floor. (``offline`` is reported once, not per model.)"""
    if not prov:
        return False
    if prov.get("model_id_resolution") == "unreachable":
        return True
    if prov.get("model_id_created_new"):
        return True
    status = prov.get("model_id_review_status")
    if status not in (None, "reviewed"):
        return True
    conf = prov.get("model_id_resolution_confidence")
    return isinstance(conf, (int, float)) and conf < RESOLVE_CONFIDENCE_FLOOR


def pretrained_repo(config: dict) -> str | None:
    """``model_args.pretrained`` — the checkpoint lm-evaluation-harness loaded."""
    args = config.get("model_args")
    if isinstance(args, dict):
        value = args.get("pretrained")
    else:
        match = re.search(r"pretrained=([^,]+)", str(args or ""))
        value = match.group(1) if match else None
    if isinstance(value, str) and value.strip(" \"'"):
        return value.strip(" \"'")
    return None


def config_model_repo(config: dict) -> str | None:
    """The model id the run itself recorded: ``pretrained``, else ``model_name``."""
    name = config.get("model_name")
    fallback = name.strip(" \"'") if isinstance(name, str) else ""
    return pretrained_repo(config) or fallback or None


def canonical_hf_repo(repo: str, *, timeout: float = 15.0) -> str | None:
    """The repo id HF's alias redirect lands on, or ``None`` if it cannot be read."""
    try:
        resp = requests.get(f"{HF_MODEL_API}{repo}", timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("id") or None
    except Exception:  # noqa: BLE001 — an unreadable alias is reported, not guessed
        return None


def evaluated_model_repo(
    model_repo: str, config: dict, *, check_aliases: bool = True
) -> tuple[str | None, dict]:
    """Which model the scores belong to, as ``(repo, provenance)``.

    Uses the dataset path when it agrees with the run config. When they disagree,
    both are resolved through HuggingFace's alias redirect and the shared target
    wins; two genuinely different repos give ``(None, ...)``, because picking one by
    preference would attribute one model's scores to another.
    """
    config_repo = config_model_repo(config)
    if not config_repo or config_repo == model_repo:
        return model_repo, {"model_identity_source": "dataset_path"}

    provenance = {
        "model_identity_dataset_path": model_repo,
        "model_identity_run_config": config_repo,
        "name_path_divergence": True,
    }
    if not check_aliases:
        return None, {**provenance, "model_identity_source": "unresolved_offline"}
    resolved = {repo: canonical_hf_repo(repo) for repo in (model_repo, config_repo)}
    if None in resolved.values():
        return None, {**provenance, "model_identity_source": "unresolved_unreachable"}
    if resolved[model_repo] != resolved[config_repo]:
        return None, {**provenance, "model_identity_source": "conflicting_repos"}
    return resolved[model_repo], {**provenance, "model_identity_source": "hf_alias"}


def latest_per_model(paths: list[str]) -> tuple[dict[str, str], list[str]]:
    """Group 3-segment ``developer/model/results_*.json`` paths, latest file per model.

    Root-level 2-segment baselines (e.g. ``GPT-4/results_*.json``) are returned
    separately and skipped: they are hand-curated closed-model paper numbers
    (bare ``acc`` only, no ``acc_stderr``/``model_args``/``bootstrap_iters``),
    NOT lm-evaluation-harness runs, so their provenance differs from the rest.
    """
    by_model: dict[str, list[str]] = defaultdict(list)
    baselines: list[str] = []
    for p in paths:
        parts = p.split("/")
        if len(parts) < 3:
            baselines.append(p)
            continue
        by_model["/".join(parts[:2])].append(p)
    chosen = {}
    for model, files in by_model.items():
        files.sort(key=lambda p: (parse_ts(p) or datetime.min.replace(tzinfo=timezone.utc), p))
        chosen[model] = files[-1]
    return chosen, baselines


def make_result(task: str, metrics: dict, eval_ts_iso: str | None) -> EvaluationResult | None:
    acc = metrics.get("acc,none")
    if acc is None:
        return None
    display, hf_repo = TASKS[task]

    # Checked here, where the task and the source file are still known: the batch
    # serializer also rejects NaN/infinity, but it does so after every worker has
    # reported, so one unusable number would fail the whole export instead of being
    # attributed to the file it came from.
    score = require_finite_number(acc, f"{task} acc,none")

    stderr = metrics.get("acc_stderr,none")
    uncertainty = None
    if stderr is not None:
        uncertainty = Uncertainty(standard_error=StandardError(
            value=require_finite_number(stderr, f"{task} acc_stderr,none")))

    score_details = ScoreDetails(
        score=score,
        details=clean_details(
            {
                "raw_metric_key": "acc,none",
                "acc_norm": metrics.get("acc_norm,none"),
                "acc_norm_stderr": metrics.get("acc_norm_stderr,none"),
                "harness_alias": metrics.get("alias"),
            }
        ),
        uncertainty=uncertainty,
    )

    return EvaluationResult(
        evaluation_result_id=f"{SRC}.{task}",
        evaluation_name=f"{SRC}.{task}",
        evaluation_timestamp=eval_ts_iso,
        source_data=SourceDataHf(
            dataset_name=display,
            source_type="hf_dataset",
            hf_repo=hf_repo,
        ),
        metric_config=MetricConfig(
            evaluation_description=(
                f"Accuracy on the {display} medical QA benchmark as reported by the "
                "Open Medical-LLM Leaderboard."
            ),
            # The registry's canonical global metric: accuracy on a 4-choice MCQ
            # set is `accuracy`. The benchmark is kept apart by evaluation_name.
            metric_id="accuracy",
            metric_name="accuracy",
            metric_kind="accuracy",
            metric_unit="proportion",
            lower_is_better=False,
            score_type=ScoreType.continuous,
            min_score=0.0,
            max_score=1.0,
        ),
        score_details=score_details,
    )


def make_log(
    model_repo: str,
    obj: dict,
    path: str,
    retrieved_ts: str,
    *,
    model_id: str | None = None,
    dataset_repo: str | None = None,
    resolution_details: dict | None = None,
) -> tuple[EvaluationLog, str, str] | None:
    """Build one aggregate log for ``developer/model``.

    ``model_repo`` is the evaluated model as established by evaluated_model_repo, and
    it drives the developer/model routing and the model metadata. ``model_id`` is the
    registry-canonical id for ``model_info.id`` (the join key); pass ``None`` for
    path-mode (id == source repo). ``dataset_repo`` is the ``developer/model`` path the
    result file sits under, and it alone keys ``evaluation_id``: an HF alias redirect
    can move ``model_repo`` and the registry can re-map a draft ``model_id``, so either
    would hand the same source file a second identity on re-ingest. Offline unit tests
    call this directly without a resolver, so it never touches the network.
    """
    developer, model = model_repo.split("/", 1)
    config = obj.get("config", {}) or {}
    results = obj.get("results", {}) or {}

    ev_results = []
    for task in TASKS:
        md = results.get(task)
        if isinstance(md, dict):
            r = make_result(task, md, None)  # per-result ts filled below
            if r is not None:
                ev_results.append(r)
    if not ev_results:
        return None

    eval_dt = parse_ts(path)
    eval_ts_iso = eval_dt.isoformat() if eval_dt else None
    for r in ev_results:
        r.evaluation_timestamp = eval_ts_iso

    if eval_dt:
        base = int(eval_dt.timestamp())
        # keep sub-second precision so two runs in the same whole second differ
        ts_token = f"{base}.{eval_dt.microsecond:06d}" if eval_dt.microsecond else str(base)
    else:
        # No parseable timestamp in the filename: derive a STABLE token from the
        # result file path (never `now`), so evaluation_id stays idempotent across
        # re-runs. evaluation_timestamp is left None (the run time is unknown).
        ts_token = re.sub(r"[^0-9A-Za-z._-]+", "-", path.rsplit("/", 1)[-1].removesuffix(".json"))

    resolved_id = model_id or model_repo  # join key; raw_slug keys evaluation_id
    raw_slug = (dataset_repo or model_repo).replace("/", "_")
    # Route by the registry-canonical model_info.id (as mmlu_pro / openeval / hal
    # do), so a canonical-id consumer finds the published record even when the
    # resolver remapped the id away from the HF repo path. model_repo stays as
    # provenance (source_model_repo, above) and raw_slug still keys evaluation_id.
    _, route_developer, route_model = datastore_path_components(SRC, resolved_id, developer)

    md_details: dict = {
        # `pretrained=<hf repo>` is lm-evaluation-harness loading a checkpoint
        # locally, so both axes are evidenced by the run config. Without it the
        # placeholders stand rather than being asserted from the repo name.
        "deployment_type": "self_deployed" if pretrained_repo(config) else None,
        "model_availability": "open_weights" if pretrained_repo(config) else None,
        "model_sha": config.get("model_sha"),
        "model_dtype": config.get("model_dtype"),
        "model_args": config.get("model_args"),
        "model_num_parameters": config.get("model_num_parameters"),
        "model_revision": config.get("model_revision"),
    }
    if resolution_details:
        md_details.update(resolution_details)
    if resolved_id != model_repo:
        md_details["source_model_repo"] = model_repo  # keep the raw->canonical mapping visible

    model_info = ModelInfo(
        name=model_repo,
        id=resolved_id,
        developer=developer,
        additional_details=clean_details(md_details),
    )

    log = EvaluationLog(
        schema_version=SCHEMA_VERSION,
        evaluation_id=f"{SRC}/{raw_slug}/{ts_token}",
        evaluation_timestamp=eval_ts_iso,
        retrieved_timestamp=retrieved_ts,
        source_metadata=SourceMetadata(
            source_name="Open Medical-LLM Leaderboard",
            source_type="documentation",
            source_organization_name="Open Life Science AI",
            source_organization_url=LEADERBOARD_SPACE,
            evaluator_relationship=EvaluatorRelationship.third_party,
            additional_details={
                "source_role": "aggregator",
                "results_dataset": REPO,
                "source_result_file": path,
            },
        ),
        eval_library=EvalLibrary(
            name="lm-evaluation-harness",
            version="unknown",
            additional_details={
                "inferred_from": "result format (acc,none / acc_stderr,none / bootstrap_iters / fewshot_seed)"
            },
        ),
        model_info=model_info,
        evaluation_results=sorted(ev_results, key=lambda r: r.evaluation_name),
    )
    return log, route_developer, route_model


def convert(
    chosen: dict[str, str],
    baselines: list[str],
    retrieved_ts: str,
    *,
    resolve_enabled: bool = True,
    workers: int = 8,
) -> tuple[SourceConversionResult[tuple[EvaluationLog, str, str]], list[tuple[str, dict]]]:
    """Convert every selected result file, accounting for each one.

    Also returns the models whose registry id needs review. A selected file that
    yields no record is a failure, not a skip: it was chosen as this
    leaderboard's latest result for a model, so an empty conversion means the
    export is incomplete.
    """
    flagged: list[tuple[str, dict]] = []
    records: list[tuple[EvaluationLog, str, str]] = []
    failures: list[SourceRecordFailure] = []

    def worker(model_repo: str):
        # make_log is INSIDE the try: a malformed record must not escape ex.map()
        # and abort the whole run — it becomes a per-model failure instead.
        try:
            obj = fetch_json(chosen[model_repo])
            evaluated, identity = evaluated_model_repo(
                model_repo, obj.get("config") or {}, check_aliases=resolve_enabled
            )
            if evaluated is None:
                return ("ERR", model_repo,
                        f'the dataset path and the run config name different models '
                        f'({identity["model_identity_dataset_path"]} vs '
                        f'{identity["model_identity_run_config"]}) and they could not '
                        f'be reconciled ({identity["model_identity_source"]}), so '
                        f'which model was evaluated is unknown', None, None)
            model_id, prov = resolve_model_id(evaluated, enabled=resolve_enabled)
            built = make_log(evaluated, obj, chosen[model_repo], retrieved_ts,
                             model_id=model_id, dataset_repo=model_repo,
                             resolution_details={**identity, **prov})
        except Exception as e:  # noqa: BLE001
            return ("ERR", model_repo, str(e), None, None)
        if built is None:
            return ("ERR", model_repo,
                    'none of the leaderboard tasks carry an `acc,none` score', None, None)
        # Every chosen file is a model's latest leaderboard result and is expected
        # to carry all nine benchmarks. A shortfall means a task failed to run or
        # report upstream, so keep the converted record BUT report it, so the run
        # exits non-zero rather than publishing a silent partial.
        covered = {r.evaluation_name for r in built[0].evaluation_results}
        missing = [t for t in TASKS if f"{SRC}.{t}" not in covered]
        return ("OK", model_repo, built, prov, missing)

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for status, model_repo, payload, prov, missing in ex.map(worker, sorted(chosen)):
            if status == "OK":
                records.append(payload)
                if missing:
                    failures.append(SourceRecordFailure(
                        source_ref=chosen[model_repo],
                        reason=(f'{len(missing)} of {len(TASKS)} leaderboard '
                                f'benchmarks have no `acc,none` score '
                                f'({", ".join(missing)}); the converted record is '
                                f'kept, but the source file is incomplete'),
                    ))
                    print(f"  PARTIAL {model_repo}: missing {missing}")
                if resolve_enabled and _needs_registry_review(prov):
                    flagged.append((model_repo, prov))
            else:
                failures.append(SourceRecordFailure(
                    source_ref=chosen[model_repo], reason=payload,
                ))
                print(f"  ERROR {model_repo}: {payload}")

    exclusions = [
        SourceRecordExclusion(
            source_ref=path,
            reason=('a hand-curated closed-model baseline (bare `acc`, no '
                    '`acc_stderr`/`model_args`), not an lm-evaluation-harness run'),
        )
        for path in sorted(baselines)
    ]
    return SourceConversionResult(
        source_name='Open Medical-LLM Leaderboard results',
        total_records=len(chosen) + len(baselines),
        records=records,
        failures=failures,
        exclusions=exclusions,
    ), flagged


def write_conversion_report(
    result: SourceConversionResult[tuple[EvaluationLog, str, str]],
    output_dir: str,
) -> Path:
    """Persist this run's accounting, replacing any previous run's copy in one step.

    Called on every run, a clean one included, and before publication.
    """
    final = default_failure_report_path(output_dir)
    staged = save_failure_report(result, final.with_name(final.name + ".tmp"))
    # Staged then renamed: an interrupted write must not truncate the report
    # a complete run left behind.
    os.replace(staged, final)
    return final


def existing_records(
    output_dir: str, routes: list[tuple[str, str]]
) -> list[Path]:
    """Records already published for the models this run is about to write.

    Filenames are fresh uuid4s, so publishing over a populated target would add
    a second copy of the same evaluation_id rather than replace it.
    """
    return sorted(
        path
        for developer, model in routes
        for path in Path(output_dir).joinpath(developer, model).glob('*.json')
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/open-medical-llm")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N models.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--no-registry-resolve",
        action="store_true",
        help="Skip the eval-card-registry lookup and use the path-derived HF id as "
             "model_info.id (faster / offline / deterministic, but NOT canonicalized). "
             "Also skips the HuggingFace alias check used to reconcile a "
             "path/config model-name disagreement.",
    )
    ap.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace the records already published for these models; they are "
             "removed only once this run's records are written. Without it a "
             "populated output directory is an error, because a re-run would add "
             "a second copy of every record.",
    )
    return ap.parse_args(argv)


def main() -> dict:
    args = parse_args()
    resolve_enabled = not args.no_registry_resolve
    # Checked before the source listing, so a mistyped limit costs no requests.
    if args.limit is not None and args.limit < 0:
        raise SystemExit(
            f"--limit must not be negative; got {args.limit}. A negative slice "
            "drops models off the end of the selection instead of taking the "
            "first N."
        )

    retrieved_ts = str(time.time())
    # Snapshot the source for the scheduled runner's provenance gate. Every
    # per-model result file read below lives in the one openlifescienceai/results
    # dataset, so a pointer at its resolved commit covers them all; re-storing
    # the files buys nothing over the commit. No-op unless capture is active.
    raw_capture.record_hf_dataset(REPO, label="Open Medical-LLM results")
    paths = list_result_files()
    chosen, baselines = latest_per_model(paths)
    if args.limit is not None:
        chosen = {model: chosen[model] for model in sorted(chosen)[: args.limit]}
    if not chosen:
        limit_note = "" if args.limit is None else f" after --limit {args.limit}"
        raise SystemExit(
            f"no models to convert{limit_note}. {len(paths)} source file(s) were "
            f"listed, {len(baselines)} of them hand-curated baselines. Nothing "
            "would be published, so this exits rather than reporting a successful "
            "refresh that wrote nothing."
        )
    print(f"Models to process: {len(chosen)}")
    print(
        f"Skipped {len(baselines)} hand-curated baseline entries (different provenance): "
        + ", ".join(sorted(p.split('/')[0] for p in baselines))
    )
    if not resolve_enabled:
        print("  NOTE: --no-registry-resolve set; model_info.id is path-derived and NOT registry-verified.")

    result, flagged = convert(
        chosen, baselines, retrieved_ts,
        resolve_enabled=resolve_enabled, workers=args.workers,
    )

    # Written before publication, and on every run: it accounts for the conversion,
    # so a publication that raises must not take the record of what failed with it,
    # and a run with nothing to report has to say so rather than leave an earlier
    # run's report standing as if it were this run's.
    report = write_conversion_report(result, args.output_dir)
    print(f"Conversion accounting: {report} ({len(result.failures)} unconverted, "
          f"{len(result.exclusions)} excluded)")

    stale = existing_records(
        args.output_dir, [(developer, model) for _, developer, model in result.records]
    )
    if stale and not args.replace_existing:
        raise SystemExit(
            f'{len(stale)} record(s) are already published under '
            f'{args.output_dir} for these models, e.g. {stale[0]}. Record '
            'filenames are fresh uuid4s, so writing now would add a second copy '
            'of each evaluation_id. Pass --replace-existing to replace them.'
        )

    written = save_evaluation_logs([
        EvaluationLogOutput(
            eval_log=log,
            base_dir=args.output_dir,
            developer=developer,
            model_name=model,
        )
        for log, developer, model in result.records
    ])
    # Removed only once the replacements are on disk: save_evaluation_logs preflights
    # the whole batch and rolls back what it created, so a failure leaves the previous
    # publication whole rather than a gap where it used to be. missing_ok because the
    # replacements are already published — a file that vanished meanwhile is no reason
    # to fail a refresh that succeeded.
    for path in stale:
        path.unlink(missing_ok=True)
    print(f"Wrote {len(written)} logs; {len(result.exclusions)} excluded; "
          f"{len(result.failures)} failed. -> {args.output_dir}")
    if flagged:
        print(f"\n  {len(flagged)} model id(s) need registry review "
              "(unresolved / auto-created draft / low-confidence / unreviewed):")
        for mr, prov in flagged:
            print(f"    - {mr}: resolution={prov.get('model_id_resolution')} "
                  f"strategy={prov.get('model_id_resolution_strategy')} "
                  f"confidence={prov.get('model_id_resolution_confidence')} "
                  f"created_new={prov.get('model_id_created_new')} "
                  f"review_status={prov.get('model_id_review_status')}")
        print("  -> record these in the PR decision log and prepare a registry alias PR.")
    result.raise_if_incomplete()
    return {
        "written": len(written),
        "failures": len(result.failures),
        "exclusions": len(result.exclusions),
        "flagged": len(flagged),
    }


if __name__ == "__main__":            # run:  uv run python -m every_eval_ever.adapters.open_medical_llm.adapter
    main()
