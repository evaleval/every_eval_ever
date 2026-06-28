# HF Community Evals

`tools/hf-community-evals/community_evals_converter.py` reviews one EEE datastore collection, creates local Hugging Face Community Evals YAML previews, audits existing model repo entries and open PRs, and opens PRs only after explicit approval.

## Quick Start

```sh
uv run tools/hf-community-evals/community_evals_converter.py MMLU-Pro \
  --datastore evaleval/EEE_datastore@main
```

Force a fresh rebuild when you need to ignore the matching local cache:

```sh
uv run tools/hf-community-evals/community_evals_converter.py MMLU-Pro \
  --datastore evaleval/EEE_datastore@main \
  --force
```

The positional argument is a collection stem that resolves to a flat datastore index under `flat/indexes/by_collection/`.

## Supported Benchmarks

The current Community Evals workflow supports:

- `gpqa`
- `gpqa_diamond` as an alias for `gpqa`
- `gsm8k`
- `hle`
- `mmlu_pro`

Unsupported benchmark names are hard errors.

## Output Directory

For `MMLU-Pro`, local outputs are written under:

```text
outputs/community_evals_converter_MMLU-Pro/
```

Important files:

- `manifest.json`: converted candidates plus skipped and error metadata.
- `review.json`: duplicate audit findings, audit errors, and PR readiness.
- `yamls/<owner>/<model>/.eval_results/<benchmark>.yaml`: local YAML previews.

`outputs/` is for review artifacts and is not a merge input.

## Review Behavior

The tool downloads the collection JSONL and referenced aggregate objects, validates object hashes and optional sizes, scans for supported Community Evals benchmarks, writes YAML entries with EEE source backlinks, checks model repo existence, audits existing `.eval_results/*.yaml` files on model `main`, and audits changed files in open PR refs.

Candidate statuses include `ready`, `already_present`, `score_conflict`, `missing_hf_model`, and `audit_error`.

## Opening PRs

PR submission is blocked until both prompts succeed:

1. Type exactly `OPEN PRS`.
2. Enter a non-empty commit message.

Only `ready` entries are submitted. Candidate-scoped audit errors block only that candidate; uncategorized audit errors block all PR submission.
