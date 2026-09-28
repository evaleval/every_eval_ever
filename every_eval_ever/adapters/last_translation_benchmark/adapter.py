"""Last Translation Benchmark (LTBv1) -> Every Eval Ever converter.

LTB is a live, human-authored benchmark of hard-to-translate inputs. Each example
carries verification rules, and each translation (by a human or an MT system) is
judged against every rule by an LLM judge (Gemini 3.1 Pro). A translation passes
an example only if it satisfies *all* of that example's rules -- this is exactly
how the LTB leaderboard score is computed (scripts/41-score_leaderboard.py).

This converter turns that into EEE records: one aggregate JSON per translation
system, plus a companion instance-level JSONL with one row per example.

    data/last-translation-benchmark/
      google/
        gemini-3.1-pro-preview/
          {uuid}.json           # aggregate: overall + per-language-pair scores
          {uuid}_samples.jsonl  # one row per example
      facebook/
        nllb-200-3.3B/
          ...

Input:
    v1.json from the LTB release (scripts/03a-prepare_release.py), available at
      https://hf.co/datasets/zouhar/last-translation-benchmark  (data/v1.json)
      http://last-translation-benchmark.vilda.net/LTBv1.json

Usage:
    Run as a module; ADAPTER is
    ``python -m every_eval_ever.adapters.last_translation_benchmark.adapter``.

    ADAPTER --input-path path/to/v1.json
    ADAPTER --input-path path/to/v1.json --aggregate-only
    ADAPTER --input-path path/to/v1.json --include-human
    ADAPTER --input-path path/to/v1.json --validate-with /path/to/every_eval_ever
"""

import argparse
import collections
import hashlib
import json
import logging
import math
import re
import sys
import time
import uuid as uuid_mod
from pathlib import Path

from .ltb_constants import (
    BENCHMARK_NAME,
    DATASET_VERSION,
    EVAL_LIBRARY,
    LLM_SCORING,
    LTB_HF_REPO,
    METRIC_PASS_RATE,
    METRIC_RULE_PASS_RATE,
    MODEL_ALIASES,
    MODEL_PREFIX_BLOCKLIST,
    MODEL_REGISTRY,
    NEEDS_REVIEW_KEY,
    PROMPT_NOT_REPRODUCIBLE,
    SCHEMA_VERSION,
    SOURCE_METADATA,
    TAGS,
    TRANSLATE_PROMPT_TEMPLATE,
)
from .ltb_instance_level import build_instance_record
from .ltb_types import ModelInfo, Outcome, Stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ltb2eee")


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")
    return slug or "unknown"


def resolve_model(display_name: str, stats: Stats) -> ModelInfo:
    """Map an LTB display name onto an EEE model identity."""
    entry = MODEL_REGISTRY.get(display_name)
    if entry is None:
        if display_name not in stats.models_unknown:
            stats.models_unknown.append(display_name)
        return ModelInfo(
            display_name=display_name,
            model_id=f"unknown/{slugify(display_name)}",
            developer="unknown",
            platform=None,
            deployment_type="unknown",
            availability="unknown",
            needs_review="display name not in MODEL_REGISTRY; id was auto-generated",
        )
    return ModelInfo(
        display_name=display_name,
        model_id=entry["id"],
        developer=entry.get("developer", entry["id"].split("/")[0]),
        platform=entry.get("platform"),
        deployment_type=entry.get("deployment", "unknown"),
        availability=entry.get("availability", "unknown"),
        needs_review=entry.get(NEEDS_REVIEW_KEY),
        engine=entry.get("engine"),
        prompt=entry.get("prompt"),
        prompt_note=entry.get("prompt_note"),
        gen_args=entry.get("gen_args"),
    )


def deterministic_uuid(*parts: str) -> str:
    """Stable UUID-v4-shaped id, so re-running does not churn filenames.

    The EEE schema requires the v4 version/variant nibbles, so they are forced.
    """
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    chars = list(digest[:32])
    chars[12] = "4"
    chars[16] = "89ab"[int(digest[16], 16) % 4]
    h = "".join(chars)
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def language_pair(example: dict) -> str:
    src = example.get("source_lang_iso")
    tgt = example.get("target_lang_iso")
    if src and tgt:
        return f"{src}-{tgt}"
    src_name = (example.get("source_lang") or "unknown").split("(")[0].strip()
    tgt_name = (example.get("target_lang") or "unknown").split("(")[0].strip()
    return f"{slugify(src_name)}-{slugify(tgt_name)}".lower()


# ---------------------------------------------------------------------------
# Reading LTB outcomes
# ---------------------------------------------------------------------------

def collect_outcomes(
    examples: list[dict], stats: Stats, include_human: bool, missing_translation: str = "fail"
) -> dict[str, list[Outcome]]:
    """Group every scored (example, system) pair by system display name."""
    by_model: dict[str, list[Outcome]] = collections.defaultdict(list)
    stats.examples_total = len(examples)

    for example in examples:
        tags = tuple(example.get("tags") or [])
        pair = language_pair(example)
        rules_total_expected = len(example.get("verification_rules") or [])

        for mt in example.get("translations") or []:
            name = mt.get("model")
            if not name:
                continue
            if name.startswith(MODEL_PREFIX_BLOCKLIST):
                stats.skipped_blocklisted += 1
                continue
            if name == "human" and not include_human:
                continue

            verified = mt.get("verified")
            if not verified or any(v is None for v in verified):
                # No usable judge verdict -> not scored. This is the ONLY filter
                # LTB itself applies (scripts/03b-bake_results_paper.py and
                # scripts/41-score_leaderboard.py both key off verified_extra).
                stats.skipped_unverified += 1
                continue

            translation = mt.get("translation")
            if translation is None or translation == "":
                # A missing/empty translation that still carries a verdict is a
                # *recorded failure to produce output*, and LTB keeps it in the
                # denominator (the verdict is already all-False). Matching that is
                # what makes these scores identical to the published leaderboard
                # and to the paper's computed/baked.json.
                stats.missing_translations += 1
                if missing_translation == "skip":
                    stats.skipped_no_translation += 1
                    continue
                translation = ""

            rules_passed = sum(1 for v in verified if v)
            by_model[name].append(
                Outcome(
                    example_id=example.get("id"),
                    passed=all(verified),
                    rules_passed=rules_passed,
                    rules_total=len(verified) or rules_total_expected,
                    translation=translation,
                    tags=tags,
                    lang_pair=pair,
                    example=example,
                )
            )

    merge_aliases(by_model, stats)
    stats.models_seen = len(by_model)
    return by_model


def merge_aliases(by_model: dict[str, list[Outcome]], stats: Stats) -> None:
    """Fold aliased display names into their canonical system, in place.

    Some systems appear under two LTB display names because they were run twice
    through different providers. MODEL_ALIASES says which name is canonical; on
    an example both scored, the canonical run wins. Emitting them as two records
    would make one model look like two to every consumer of the datastore.
    """
    for alias, canonical in MODEL_ALIASES.items():
        alias_outcomes = by_model.pop(alias, None)
        if not alias_outcomes:
            continue
        canonical_outcomes = by_model.setdefault(canonical, [])
        already = {o.example_id for o in canonical_outcomes}
        added = [o for o in alias_outcomes if o.example_id not in already]
        canonical_outcomes.extend(added)
        stats.aliases_merged[alias] = {
            "canonical": canonical,
            "alias_examples": len(alias_outcomes),
            "contributed": len(added),
            "overridden_by_canonical": len(alias_outcomes) - len(added),
        }
        logger.info(
            "merged %r into %r: %d of its %d examples were already covered by "
            "%r and kept from there, %d added",
            alias, canonical, len(alias_outcomes) - len(added), len(alias_outcomes),
            canonical, len(added),
        )


# ---------------------------------------------------------------------------
# Aggregate record
# ---------------------------------------------------------------------------

def _uncertainty(score: float, n: int) -> dict:
    se = math.sqrt(max(score * (1.0 - score), 0.0) / n) if n else 0.0
    return {
        "standard_error": {"value": se, "method": "analytic"},
        "num_samples": n,
    }


def _source_data(subset: str, pair: str | None, outcomes: list[Outcome], max_sample_ids: int) -> dict:
    details = {"subset": subset, "dataset_version": DATASET_VERSION}
    if pair:
        details["language_pair"] = pair
    data = {
        "dataset_name": "Last Translation Benchmark",
        "source_type": "hf_dataset",
        "hf_repo": LTB_HF_REPO,
        "hf_split": "train",
        "samples_number": len(outcomes),
        "additional_details": details,
    }
    if max_sample_ids and len(outcomes) <= max_sample_ids:
        data["sample_ids"] = [str(o.example_id) for o in outcomes]
    return data


def _result(
    *,
    subset: str,
    pair: str | None,
    metric: dict,
    outcomes: list[Outcome],
    rule_level: bool,
    max_sample_ids: int,
    suffix: str = "",
    model: ModelInfo | None = None,
) -> dict:
    if rule_level:
        total = sum(o.rules_total for o in outcomes)
        passed = sum(o.rules_passed for o in outcomes)
        score = passed / total if total else 0.0
        n = total
    else:
        n = len(outcomes)
        score = sum(1 for o in outcomes if o.passed) / n if n else 0.0

    scope = pair or "overall"
    name = f"{subset} {pair}" if pair else subset
    description = (
        f"Fraction of individual verification rules passed on {subset}"
        if rule_level
        else f"Fraction of {subset} examples whose translation passed all verification rules"
    )
    if pair:
        description += f" for language pair {pair}"

    metric_config = dict(metric)
    metric_config["evaluation_description"] = description
    metric_config["llm_scoring"] = LLM_SCORING

    return {
        "evaluation_result_id": f"{BENCHMARK_NAME}/{subset}/{scope}/{metric['metric_id']}{suffix}",
        "evaluation_name": name,
        "source_data": _source_data(subset, pair, outcomes, max_sample_ids),
        "metric_config": metric_config,
        "score_details": {
            "score": score,
            "uncertainty": _uncertainty(score, n),
            "details": {
                "examples_scored": str(len(outcomes)),
                "examples_passed": str(sum(1 for o in outcomes if o.passed)),
                "rules_scored": str(sum(o.rules_total for o in outcomes)),
                "rules_passed": str(sum(o.rules_passed for o in outcomes)),
            },
        },
        "generation_config": _generation_config(model),
    }


def _generation_config(model: ModelInfo | None) -> dict:
    """Decoding parameters and prompt format actually used for this system.

    The OpenRouter-served LLMs (scripts/20b) were queried with defaults and a
    single generic prompt. The dedicated MT systems (scripts/22) each have real,
    published decoding parameters and, in several cases, their own prompt format.
    """
    gen_args: dict = {}
    details: dict[str, str] = {}

    if model is not None and model.gen_args:
        gen_args.update(model.gen_args)
    else:
        details["decoding"] = (
            "LTB does not publish decoding parameters for the API-served systems; "
            "they were queried with provider defaults"
        )

    prompt = model.prompt if model is not None else None
    if prompt is PROMPT_NOT_REPRODUCIBLE:
        details["prompt_format"] = model.prompt_note or "not reproducible from the public release"
    elif isinstance(prompt, str):
        gen_args["prompt_template"] = prompt
    else:
        gen_args["prompt_template"] = TRANSLATE_PROMPT_TEMPLATE

    config: dict = {"generation_args": gen_args}
    if details:
        config["additional_details"] = details
    return config


def build_aggregate(
    model: ModelInfo,
    outcomes: list[Outcome],
    retrieved_timestamp: str,
    min_pair_size: int,
    max_sample_ids: int,
    per_language: bool,
    disambiguate: bool = False,
) -> tuple[dict, list[dict]]:
    """Return (aggregate record, results list) for one system.

    `disambiguate` is set when several LTB display names resolve to the same
    model_id. Two distinct runs must not share an evaluation_id, and
    evaluation_result_id is the documented foreign key for instance rows, so
    both get the display name appended. Every colliding entry is suffixed (not
    just the later one), so the ids do not depend on iteration order.

    MODEL_ALIASES already merges the display names that are known to be one
    model run twice, so in practice this is a guard for a future collision.
    """
    model_key = model.model_id.replace("/", "_")
    suffix = ""
    if disambiguate:
        model_key = f"{model_key}__{slugify(model.display_name).lower()}"
        suffix = f"@{slugify(model.display_name).lower()}"
    evaluation_id = f"{BENCHMARK_NAME}/{model_key}/{retrieved_timestamp}"

    results: list[dict] = []
    for subset in TAGS:
        subset_outcomes = [o for o in outcomes if subset in o.tags]
        if not subset_outcomes:
            continue
        results.append(
            _result(subset=subset, pair=None, metric=METRIC_PASS_RATE, outcomes=subset_outcomes,
                    rule_level=False, max_sample_ids=max_sample_ids, suffix=suffix, model=model)
        )
        results.append(
            _result(subset=subset, pair=None, metric=METRIC_RULE_PASS_RATE, outcomes=subset_outcomes,
                    rule_level=True, max_sample_ids=max_sample_ids, suffix=suffix, model=model)
        )

    primary = TAGS[0]
    if per_language:
        primary_outcomes = [o for o in outcomes if primary in o.tags]
        by_pair: dict[str, list[Outcome]] = collections.defaultdict(list)
        for o in primary_outcomes:
            by_pair[o.lang_pair].append(o)
        for pair in sorted(by_pair):
            pair_outcomes = by_pair[pair]
            if len(pair_outcomes) < min_pair_size:
                continue
            results.append(
                _result(subset=primary, pair=pair, metric=METRIC_PASS_RATE, outcomes=pair_outcomes,
                        rule_level=False, max_sample_ids=max_sample_ids, suffix=suffix, model=model)
            )

    model_details = {
        "deployment_type": model.deployment_type,
        "model_availability": model.availability,
        "ltb_display_name": model.display_name,
    }
    merged_from = sorted(a for a, c in MODEL_ALIASES.items() if c == model.display_name)
    if merged_from:
        model_details["ltb_merged_display_names"] = "; ".join(merged_from)
        model_details["ltb_merge_note"] = (
            "the same model run twice under two LTB display names; this record is the "
            f"{model.display_name!r} run, which already covers every example the other "
            "one scored"
        )
    if model.needs_review:
        model_details["id_provenance"] = model.needs_review

    model_info = {
        "name": model.display_name,
        "id": model.model_id,
        "developer": model.developer,
        "additional_details": model_details,
    }
    if model.platform:
        model_info["inference_platform"] = model.platform
    if model.engine:
        model_info["inference_engine"] = {"name": model.engine}

    record = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "retrieved_timestamp": retrieved_timestamp,
        "source_metadata": SOURCE_METADATA,
        "eval_library": EVAL_LIBRARY,
        "model_info": model_info,
        "evaluation_results": results,
    }
    return record, results


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_model(
    model: ModelInfo,
    outcomes: list[Outcome],
    output_dir: Path,
    args,
    stats: Stats,
    retrieved_timestamp: str,
    disambiguate: bool = False,
) -> dict:
    record, results = build_aggregate(
        model,
        outcomes,
        retrieved_timestamp,
        min_pair_size=args.min_pair_size,
        max_sample_ids=args.max_sample_ids,
        per_language=not args.no_per_language,
        disambiguate=disambiguate,
    )
    if not results:
        logger.warning("no results for %s (no examples in tags %s) - skipped", model.display_name, TAGS)
        return {}

    # The display name is part of the derivation so that two LTB display names
    # mapping to the same model_id produce two files in the same folder rather
    # than silently overwriting each other. EEE allows multiple result files per
    # model; `model_info.additional_details.ltb_display_name` tells them apart.
    file_uuid = (
        str(uuid_mod.uuid4())
        if args.random_uuid
        else deterministic_uuid(BENCHMARK_NAME, DATASET_VERSION, model.model_id, model.display_name)
    )

    model_dir = output_dir / model.developer_dir / model.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    json_path = model_dir / f"{file_uuid}.json"
    jsonl_path = model_dir / f"{file_uuid}_samples.jsonl"

    if not args.aggregate_only:
        # Instance rows are attached to the primary subset's example-level
        # pass-rate result. Per-language-pair results reuse the same instances;
        # metadata.language_pair lets consumers regroup them.
        primary = TAGS[0]
        primary_result_id = next(
            r["evaluation_result_id"] for r in results
            if r["evaluation_result_id"].startswith(f"{BENCHMARK_NAME}/{primary}/overall/")
            and METRIC_PASS_RATE["metric_id"] in r["evaluation_result_id"]
        )
        primary_outcomes = [o for o in outcomes if primary in o.tags]

        rows = 0
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for o in sorted(primary_outcomes, key=lambda x: x.example_id):
                row = build_instance_record(
                    o, model,
                    evaluation_id=record["evaluation_id"],
                    evaluation_name=primary,
                    evaluation_result_id=primary_result_id,
                )
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                rows += 1
        stats.instance_rows += rows

        checksum = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
        record["detailed_evaluation_results"] = {
            "format": "jsonl",
            "file_path": (
                f"data/{BENCHMARK_NAME}/{model.developer_dir}/{model.model_dir}/{file_uuid}_samples.jsonl"
            ),
            "hash_algorithm": "sha256",
            "checksum": checksum,
            "total_rows": rows,
            "additional_details": {
                "linked_result": primary_result_id,
                "note": "one row per LTBv1 example; per-language-pair aggregate results "
                "reuse these rows, grouped by metadata.language_pair",
            },
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=4)
        f.write("\n")
    stats.files_written += 1

    overall = next(
        (r["score_details"]["score"] for r in results
         if r["evaluation_result_id"].endswith(f"/{TAGS[0]}/overall/{METRIC_PASS_RATE['metric_id']}")),
        None,
    )
    return {
        "display_name": model.display_name,
        "model_id": model.model_id,
        "path": f"data/{BENCHMARK_NAME}/{model.developer_dir}/{model.model_dir}/{file_uuid}.json",
        "results": len(results),
        "examples_scored": len(outcomes),
        f"{TAGS[0]}_pass_rate": overall,
        "needs_review": model.needs_review,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def schema_checks(output_dir: Path, eee_repo: Path) -> int:
    """Validate every generated file against the EEE JSON schemas."""
    try:
        import jsonschema
    except ImportError:
        logger.error("jsonschema is not installed: pip install jsonschema")
        return 1

    agg_schema_path = next(
        (p for p in [eee_repo / "eval.schema.json",
                     eee_repo / "every_eval_ever" / "schemas" / "eval.schema.json"] if p.exists()),
        None,
    )
    inst_schema_path = next(
        (p for p in [eee_repo / "instance_level_eval.schema.json",
                     eee_repo / "every_eval_ever" / "schemas" / "instance_level_eval.schema.json"]
         if p.exists()),
        None,
    )
    if not agg_schema_path or not inst_schema_path:
        logger.error("could not find EEE schemas under %s", eee_repo)
        return 1

    agg_schema = json.loads(agg_schema_path.read_text())
    inst_schema = json.loads(inst_schema_path.read_text())
    logger.info(
        "validating against %s (agg v%s, instance v%s)",
        eee_repo, agg_schema.get("version"), inst_schema.get("version"),
    )

    # Compile once. jsonschema.validate() re-builds the validator on every call,
    # which turns ~100k instance rows into a multi-minute run.
    agg_validator = jsonschema.validators.validator_for(agg_schema)(agg_schema)
    inst_validator = jsonschema.validators.validator_for(inst_schema)(inst_schema)

    failures = 0
    for path in sorted(output_dir.rglob("*.json")):
        if path.name == "conversion_report.json":
            continue
        for e in agg_validator.iter_errors(json.loads(path.read_text(encoding="utf-8"))):
            failures += 1
            logger.error("%s: %s (at %s)", path.name, e.message, "/".join(str(p) for p in e.absolute_path))

    for path in sorted(output_dir.rglob("*_samples.jsonl")):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                error = next(inst_validator.iter_errors(json.loads(line)), None)
                if error is not None:
                    failures += 1
                    logger.error("%s:%d: %s (at %s)", path.name, i, error.message,
                                 "/".join(str(p) for p in error.absolute_path))
                    break  # one error per file is enough to act on

    if failures:
        logger.error("validation FAILED with %d error(s)", failures)
    else:
        logger.info("validation passed")
    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-path", required=True,
                    help="path to LTB v1.json (list of examples)")
    ap.add_argument("--output-dir", default=f"data/{BENCHMARK_NAME}",
                    help=f"output root (default: data/{BENCHMARK_NAME})")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip the instance-level {uuid}_samples.jsonl files")
    ap.add_argument("--include-human", action="store_true",
                    help="also emit the human contributor translations as a reference row")
    ap.add_argument("--no-per-language", action="store_true",
                    help="omit per-language-pair results, keep overall only")
    ap.add_argument("--missing-translation", choices=["fail", "skip"], default="fail",
                    help="how to treat a null/empty translation that still carries a judge "
                         "verdict: 'fail' keeps it in the denominator, reproducing LTB's "
                         "published numbers exactly (default); 'skip' excludes it")
    ap.add_argument("--min-examples", type=int, default=100,
                    help="skip systems scored on fewer than this many examples; LTBv1 has a "
                         "long tail of systems run on <50 examples (default: 100)")
    ap.add_argument("--min-pair-size", type=int, default=5,
                    help="minimum examples for a per-language-pair result (default: 5)")
    ap.add_argument("--max-sample-ids", type=int, default=200,
                    help="include source_data.sample_ids when a result has at most this many "
                         "examples; 0 disables (default: 200)")
    ap.add_argument("--only-model", action="append", default=None,
                    help="restrict to these LTB display names (repeatable)")
    ap.add_argument("--random-uuid", action="store_true",
                    help="use a fresh uuid4 per file instead of a deterministic one")
    ap.add_argument("--retrieved-timestamp", default=None,
                    help="Unix epoch string; defaults to now")
    ap.add_argument("--validate-with", default=None,
                    help="path to a clone of evaleval/every_eval_ever; validates the output "
                         "against its bundled schemas after conversion")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip all post-conversion validation")
    ap.add_argument("--strict", action="store_true",
                    help="fail if any LTB system is missing from MODEL_REGISTRY")
    args = ap.parse_args()

    input_path = Path(args.input_path).expanduser()
    if not input_path.exists():
        logger.error("input not found: %s", input_path)
        logger.error("download it from https://hf.co/datasets/%s (data/v1.json)", LTB_HF_REPO)
        return 1

    examples = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(examples, list):
        logger.error("expected v1.json to be a list of examples, got %s", type(examples).__name__)
        return 1

    retrieved_timestamp = args.retrieved_timestamp or f"{time.time():.6f}"
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = Stats()
    by_model = collect_outcomes(
        examples, stats,
        include_human=args.include_human,
        missing_translation=args.missing_translation,
    )

    if args.only_model:
        wanted = set(args.only_model)
        by_model = {k: v for k, v in by_model.items() if k in wanted}

    # Systems that were only run on a handful of examples produce a score that
    # is not comparable to the rest of the table; drop them by default.
    thin = {name: len(o) for name, o in by_model.items() if len(o) < args.min_examples}
    if thin:
        logger.warning(
            "%d system(s) below --min-examples=%d, skipped: %s",
            len(thin), args.min_examples,
            ", ".join(f"{n} ({c})" for n, c in sorted(thin.items(), key=lambda kv: -kv[1])),
        )
        by_model = {k: v for k, v in by_model.items() if k not in thin}
        stats.models_thin = thin

    resolved = {name: resolve_model(name, stats) for name in sorted(by_model)}

    collisions: dict[str, list[str]] = collections.defaultdict(list)
    for name, model in resolved.items():
        collisions[model.model_id].append(name)
    shared_ids = {mid for mid, names in collisions.items() if len(names) > 1}
    for model_id, names in sorted(collisions.items()):
        if len(names) > 1:
            logger.warning(
                "model id %s is shared by %d LTB display names (%s); writing one result "
                "file per display name in the same folder, with the display name appended to "
                "evaluation_id and evaluation_result_id - add a MODEL_ALIASES entry if they "
                "are one model run twice",
                model_id, len(names), ", ".join(names),
            )

    summaries = []
    for display_name, model in resolved.items():
        summary = write_model(model, by_model[display_name], output_dir, args, stats,
                              retrieved_timestamp,
                              disambiguate=model.model_id in shared_ids)
        if summary:
            summaries.append(summary)

    logger.info("---")
    logger.info("examples in release:      %d", stats.examples_total)
    logger.info("systems converted:        %d", stats.files_written)
    logger.info("instance-level rows:      %d", stats.instance_rows)
    logger.info("missing translations:     %d (policy: %s)", stats.missing_translations, args.missing_translation)
    logger.info("  of those, skipped:      %d", stats.skipped_no_translation)
    logger.info("skipped, no judge verdict:%d", stats.skipped_unverified)
    logger.info("skipped, blocklisted:     %d", stats.skipped_blocklisted)

    if stats.models_unknown:
        logger.warning("%d system(s) missing from MODEL_REGISTRY (auto-id'd as unknown/*): %s",
                       len(stats.models_unknown), ", ".join(sorted(stats.models_unknown)))
        logger.warning("add them to ltb_constants.MODEL_REGISTRY before submitting")

    review = [s for s in summaries if s.get("needs_review")]
    if review:
        logger.warning("%d system(s) have an unverified model id:", len(review))
        for s in review:
            logger.warning("  %-24s -> %-34s (%s)", s["display_name"], s["model_id"], s["needs_review"])

    report_path = output_dir / "conversion_report.json"
    report_path.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "retrieved_timestamp": retrieved_timestamp,
                "schema_version": SCHEMA_VERSION,
                "stats": stats.__dict__,
                "systems": sorted(summaries, key=lambda s: -(s.get(f"{TAGS[0]}_pass_rate") or 0)),
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("report: %s", report_path)
    logger.info("NOTE: delete %s before submitting; only data files belong in the PR", report_path.name)

    if args.strict and stats.models_unknown:
        return 1

    if args.no_validate or not args.validate_with:
        return 0
    return 1 if schema_checks(output_dir, Path(args.validate_with).expanduser()) else 0


if __name__ == "__main__":
    sys.exit(main())
