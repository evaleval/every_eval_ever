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
uv run every_eval_ever validate data/benchmark/dev/model/uuid.json

# Instance-level JSONL
uv run every_eval_ever validate data/benchmark/dev/model/uuid_samples.jsonl

# Entire directory (recurses into subdirectories)
uv run every_eval_ever validate data/benchmark/dev/model/

# Multiple paths
uv run every_eval_ever validate file1.json file2_samples.jsonl data/
```

File type is determined by extension: `.json` validates against `EvaluationLog`, `.jsonl` validates each line against `InstanceLevelEvaluationLog`.

### Output formats

```sh
# Machine-readable JSON output (default)
uv run every_eval_ever validate data/

# Machine-readable JSON
uv run every_eval_ever validate --format json data/

# GitHub Actions annotations
uv run every_eval_ever validate --format github data/
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--format {json,github}` | `json` | Output format |
| `--max-errors N` | `50` | Maximum errors reported per JSONL file |

Exit code is `0` if all files pass and `1` if any fail.

## Duplicate Check

Run duplicate detection separately for aggregate JSON records:

```sh
uv run every_eval_ever check-duplicates data/benchmark/

# Explicit same-batch-only mode; does not check accepted datastore records
uv run every_eval_ever check-duplicates --local-only data/benchmark/

# Use a downloaded or test manifest without network access
uv run every_eval_ever check-duplicates --manifest manifest.json data/benchmark/
```

This command uses the same semantic fingerprint as the validator Space. It
ignores non-identity fields such as UUIDs, timestamps, free-form details, paths,
and source metadata, then compares model, evaluation library, dataset identity,
metric identity, score, and generation config within each `data/<collection>`.
By default it downloads the accepted datastore manifest. Manifest download,
candidate parsing, and candidate fingerprint failures are blocking; the command
does not silently fall back to local-only comparison.

Dataset identity includes URL artifacts for URL sources and repository,
configuration, revision, split, and sample IDs when supplied for Hugging Face
sources. Private/custom sources require a stable source ID, source version, or
source URL before they can be fingerprinted safely.

### Skip unchanged adapter sources before download

Semantic fingerprints are computed from downloaded aggregate records, so they
cannot by themselves prevent repeated downloads. Incremental adapter runs use a
separate source index stored in `manifest.json`:

```json
{
  "fingerprint_version": "eee-semantic-v2",
  "files": {
    "data/alpaca_eval/dev/model/result.json": {
      "fingerprint": "<semantic-fingerprint>"
    }
  },
  "sources": {
    "<sha256-of-adapter-and-source-id>": {
      "adapter": "alpaca_eval",
      "source_id": "leaderboard:v2",
      "revision": "etag:\"abc123\"",
      "files": ["data/alpaca_eval/dev/model/result.json"]
    }
  }
}
```

An adapter first discovers lightweight `SourceCandidate` values containing a
stable adapter name, source ID, and immutable revision. `SourceIndex.plan()`
then returns one of `download_new`, `download_changed`, or `skip_unchanged`.
`execute_download_plan()` never invokes the adapter download callback for an
exact source/revision match. After a successful download and export,
`source_manifest_entry()` produces the entry to persist in the next manifest.

For HTTP artifacts, `fetch_http_revision()` performs a HEAD request and uses an
ETag or Last-Modified value. An endpoint without revision metadata is rejected
for incremental operation; it is not assumed unchanged. Hugging Face adapters
should use a commit SHA or blob revision, and API adapters should use an
upstream update/version identifier. The semantic fingerprint check still runs
after changed/new content is downloaded, because source identity and semantic
identity solve different problems.

`fingerprint_version` is mandatory. A manifest produced by another fingerprint
algorithm is rejected and must be rebuilt; fingerprints from different
algorithms are never compared implicitly.

## Semantic Warnings

The CLI and Space share the same non-blocking semantic warnings:

- Datastore path hierarchy and UUID4 filename checks.
- Missing aggregate `.jsonl` companions at the exact relative path declared by
  `detailed_evaluation_results.file_path`. Absolute and parent-traversing paths
  are rejected.
- Missing, null, or non-finite score metadata; reversed bounds; and scores
  outside valid bounds. Nonstandard JSON `NaN`/`Infinity` and duplicate object
  keys are blocking parse errors.
- Non-integer count fields such as `num_samples`.
- Model deployment metadata under `model_info.additional_details`:
  `deployment_type` is `api`, `local`, or `unknown`; `api` models use
  `model_availability` values `closed_source`, `open_weights_deployment`, or
  `other`; `local` models use `hf`, `unavailable`, or `other`.
- Required Hugging Face model checks when `model_availability` is `hf`.
- Required Hugging Face dataset checks when `source_data.source_type` is
  `hf_dataset`, plus warnings for weak `other` dataset provenance.

The aggregate schema requires `hf_repo` for Hugging Face dataset sources and
requires `format: jsonl` plus `file_path` whenever detailed results are present.
Multi-turn and agentic instance records require at least one message and an
`evaluation.num_turns` value.

## PR Bot

The Hugging Face datastore PR bot validates changed `data/**/*.json` and
`data/**/*.jsonl` files through the package validation core, checks paths,
compares aggregate candidates against accepted records, and posts a visible PR
report.
