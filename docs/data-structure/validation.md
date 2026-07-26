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

### Output

```sh
# Validation reports are emitted as machine-readable JSON
uv run every_eval_ever validate data/
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-errors N` | `50` | Maximum errors reported per JSONL file |

Exit code is `0` if all files pass and `1` if any fail.

## Duplicate Check

Run duplicate detection separately for aggregate JSON records:

```sh
uv run every_eval_ever check-duplicates data/benchmark/

# Use a downloaded or test manifest without network access
uv run every_eval_ever check-duplicates --manifest manifest.json data/benchmark/
```

This command uses the same semantic fingerprint as the validator Space. It
ignores non-identity fields such as UUIDs, timestamps, free-form details, paths,
and source metadata. Fingerprint v3 compares model ID, model revision,
precision, deployment type, inference platform and engine,
evaluation-library name and version, plus the complete order-independent set
of evaluation name, dataset identity (including URL locations), metric
identity, exact score, and complete generation configuration. Comparison is
global across the datastore; a different `data/<collection>` path cannot
bypass deduplication.
By default it downloads the accepted datastore manifest. Manifest download,
candidate parsing, and candidate fingerprint failures are blocking; the command
does not silently fall back to local-only comparison.

URL source locations participate in semantic identity. Hugging Face dataset
identity includes repository, configuration, revision, split, and sample IDs
when supplied. Private/custom sources require a stable source ID, source
version, or source URL.

The manifest contains only the accepted semantic fingerprints:

```json
{
  "fingerprint_version": "eee-semantic-v3",
  "files": {
    "data/alpaca_eval/dev/model/result.json": {
      "fingerprint": "<semantic-fingerprint>"
    }
  }
}
```

Cron adapters download and validate their current output before using the
canonical semantic duplicate check. There is no separate source-revision
shortcut: heterogeneous upstream metadata is not trusted to prove that record
contents are unchanged. Exact semantic duplicates are omitted from upload. If
the same logical evaluation later has a different score, its fingerprint
changes and both immutable records are retained.

Each scheduled adapter also writes one deterministic, replayable input snapshot
to the private ingestion dataset before any datastore upload. Raw blobs are
gzip-compressed and addressed by their SHA-256:

```text
raw/<adapter>/<sha-prefix>/<sha256>/<input-name>.gz
```

The SHA is an archive identity and storage optimization only. It does not skip
the upstream download and does not decide whether a generated evaluation is a
duplicate. Every run appends immutable `raw_archived` and `completed` or
`failed` events under
`ledger/events/<year>/<month>/<day>/<run-id>/`. Those events link raw hashes to
adapter results, selected datastore paths, duplicate matches, and PR URLs.
`EEE_INGESTION_REPO_ID` selects the private dataset and
`EEE_INGESTION_HF_TOKEN` supplies its token. Both are required; the cron fails
before running adapters when either is absent.

Scheduled runs publish collection-scoped PRs with
`EEE_CRON_DEDUP_MODE=deferred` while the datastore migration is under review.
Deferred mode uses the canonical dedup pathway against the current run and
files already present in open adapter-cron PRs, but it does not read or claim
coverage of datastore `main`. Every deferred PR says this explicitly and must
not be merged until the complete accepted fingerprint-v3 manifest exists.

Set `EEE_CRON_DEDUP_MODE=enforced` after that manifest is published. Enforced
mode adds the accepted datastore manifest to the same canonical comparison.
Unknown mode values fail rather than silently selecting a scope. The
`--archive-only` CLI option remains available for explicit fetch/validate/raw
archive runs that should not inspect the datastore or open PRs.

PR ownership is per collection. Most scheduled adapters therefore maintain one
open PR, while a multi-collection adapter maintains one PR for each collection.
Later runs update the matching open `[adapter-cron] <collection>` PR.

The GitHub cron runs only adapters in its explicit allowlist. Larger adapters
can still be selected manually, but are not included in scheduled or
`--force-all` runs until their resource use is known to fit the runner.

`fingerprint_version` is mandatory. A manifest produced by another fingerprint
algorithm is rejected and must be rebuilt; fingerprints from different
algorithms are never compared implicitly.

## Enforcement Rules And Warnings

The CLI and Space classify every registered check explicitly. The following
rules are blocking errors:

- Datastore path hierarchy and UUID4 filename checks.
- Missing aggregate `.jsonl` companions at the exact relative path declared by
  `detailed_evaluation_results.file_path`. Absolute and parent-traversing paths
  are rejected.
- Missing, null, or non-finite score metadata; reversed bounds; and scores
  outside valid bounds. Nonstandard JSON `NaN`/`Infinity` and duplicate object
  keys are blocking parse errors.
- Non-integer count fields such as `num_samples`.
- Model deployment metadata under `model_info.additional_details`:
  `deployment_type` is `self_deployed`, `externally_managed`, or `unknown`,
  while the independent `model_availability` axis is `open_weights`,
  `closed_weights`, or `unknown`. No combination is implicitly forbidden.
- Required Hugging Face dataset checks when `source_data.source_type` is
  `hf_dataset`.

Allowed `source_type: other` records with weak provenance remain advisory
warnings. Aggregates with no evaluation results and empty instance JSONL files
are blocking errors.

The aggregate schema requires `hf_repo` for Hugging Face dataset sources and
requires `format: jsonl` plus `file_path` whenever detailed results are present.
Multi-turn and agentic instance records require at least one message and an
`evaluation.num_turns` value.

## PR Bot

The Hugging Face datastore PR bot validates changed `data/**/*.json` and
`data/**/*.jsonl` files through the package validation core, checks paths,
compares aggregate candidates against accepted records, and posts a visible PR
report.
