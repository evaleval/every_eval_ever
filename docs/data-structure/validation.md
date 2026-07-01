---
layout: default
title: Validation
parent: Data Structure
nav_order: 2
---

# Validation

Validation uses Pydantic models generated from the JSON schemas. This validates aggregate `.json` files against `EvaluationLog` and instance-level `_samples.jsonl` files line-by-line against `InstanceLevelEvaluationLog`. Requires [uv](https://docs.astral.sh/uv/).

## Validate files with the package CLI

```sh
# Single aggregate file
uv run python -m every_eval_ever validate data/benchmark/dev/model/uuid.json

# Instance-level JSONL
uv run python -m every_eval_ever validate data/benchmark/dev/model/uuid_samples.jsonl

# Entire directory (recurses into subdirectories)
uv run python -m every_eval_ever validate data/benchmark/dev/model/

# Multiple paths
uv run python -m every_eval_ever validate file1.json file2_samples.jsonl data/
```

File type is determined by extension: `.json` validates against `EvaluationLog`, `.jsonl` validates each line against `InstanceLevelEvaluationLog`.

### Output formats

```sh
# Rich terminal output (default)
uv run python -m every_eval_ever validate data/

# Machine-readable JSON
uv run python -m every_eval_ever validate --format json data/

# GitHub Actions annotations
uv run python -m every_eval_ever validate --format github data/
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--format {rich,json,github}` | `rich` | Output format |
| `--max-errors N` | `50` | Maximum errors reported per JSONL file |

Exit code is `0` if all files pass and `1` if any fail.

## Duplicate Check

Run duplicate detection separately for aggregate JSON records:

```sh
uv run every_eval_ever check-duplicates data/benchmark/
```

This command uses the same semantic fingerprint as the validator Space. It
ignores non-identity fields such as UUIDs, timestamps, free-form details, paths,
and source metadata, then compares model, evaluation library, dataset identity,
metric identity, score, and generation config within each `data/<collection>`.

## Semantic Warnings

The CLI and Space share the same non-blocking semantic warnings:

- Datastore path hierarchy and UUID4 filename checks.
- Missing aggregate `.jsonl` companions when detailed results reference them.
- Missing `score_type`, `min_score`, or `max_score`, and scores outside bounds.
- Non-integer count fields such as `num_samples`.
- Model deployment metadata under `model_info.additional_details`:
  `deployment_type` is `api`, `local`, or `unknown`; `api` models use
  `model_availability` values `closed_source`, `open_weights_deployment`, or
  `other`; `local` models use `hf`, `unavailable`, or `other`.
- Required Hugging Face model checks when `model_availability` is `hf`.
- Required Hugging Face dataset checks when `source_data.source_type` is
  `hf_dataset`, plus warnings for weak `other` dataset provenance.

## PR Bot

The Hugging Face datastore PR bot validates changed `data/**/*.json` and
`data/**/*.jsonl` files through the package validation core, checks paths,
compares aggregate candidates against accepted records, and posts a visible PR
report.
