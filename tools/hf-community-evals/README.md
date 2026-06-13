# EEE -> HF Community Evals

Built by Harsha Nelaturu, June 2026.

Use `tools/community_evals_converter.py` to review one EEE datastore collection, generate
local HF Community Evals YAML previews, audit existing scores/open PRs, and
optionally open PRs after explicit approval.

## Quick Start

Use `uv run` for all commands.

```bash
uv run tools/community_evals_converter.py MMLU-Pro \
  --datastore evaleval/EEE_datastore@main
```

This will cache the results for this particular collection and if you would like to force a fresh rebuild:

```bash
uv run tools/community_evals_converter.py MMLU-Pro \
  --datastore evaleval/EEE_datastore@main \
  --force
```

The positional argument is a collection stem. It must resolve exactly to:

```text
https://huggingface.co/datasets/evaleval/EEE_datastore/flat/indexes/by_collection/<collection>.jsonl
```

## Outputs

For `MMLU-Pro`, outputs are written under:

```text
outputs/community_evals_converter_MMLU-Pro/
```

Important output files:

- `manifest.json`: converted candidate records plus skipped/error metadata.
- `review.json`: full review result, duplicate audit findings, audit errors,
  and PR readiness.
- `yamls/<owner>/<model>/.eval_results/<benchmark>.yaml`: local YAML previews.

`outputs/` is ignored by git. Use these files for inspection, not as merge
inputs.

## Review Behavior

The tool:

- downloads the collection JSONL and referenced aggregate objects from the HF
  datastore;
- validates object hashes and optional sizes;
- scans each aggregate record for supported HF benchmark datasets;
- writes YAML entries using the datastore object HF URL as `source.url`;
- keeps flat datastore provenance, including instance-level references when
  present;
- checks model repo existence on Hugging Face;
- audits every existing `.eval_results/*.yaml` file on model `main`;
- audits changed `.eval_results/*.yaml` files in open PR refs;
- compares by dataset/task content, not YAML filename.

Supported benchmarks in this workflow are:

- `mmlu_pro`
- `gpqa`
- `hle`
- `gsm8k`

## Resume And Force

Default reruns reuse exact-match local outputs:

- matching completed `review.json`: skips collection downloads, model checks,
  and duplicate audit;
- matching pre-audit `manifest.json`: skips collection downloads and model
  checks, then resumes at duplicate audit.

The cache must match collection name, datastore input, and HF-check mode.
Invalid exact-match cache files are hard errors. Use `--force` when you want to
ignore the cache and rebuild from the datastore.

## TUI
The final report has:

- `Community Evals Converter`: summary counts.
- `Needs Attention`: capped triage table for blockers and exclusions.

`Needs Attention` uses:

- `Issue`: `audit_error`, `score_conflict`, `already_present`,
  `missing_hf_model`, or `skipped`.
- `Model`: model repo or aggregate model id.
- `Details`: reason or score comparison.
- `Action`: `exclude`, `block entry`, `block all`, or source line.
- `Where`: terminal hyperlink to the HF model PR/file or HF datastore blob URL.

Repeated same-score `already_present` findings are summarized as one count row.
Full details remain in `review.json`.

## PR Submission

The tool only opens PRs after both prompts succeed:

1. Type exactly:

   ```text
   OPEN PRS
   ```

2. Enter a non-empty commit message.

Only `status = ready` entries are submitted.

Excluded statuses:

- `already_present`: same score already exists.
- `score_conflict`: different score already exists.
- `missing_hf_model`: model repo does not resolve on HF.
- `audit_error`: candidate-scoped audit failure.

Candidate-scoped audit errors block only that candidate. Audit errors without a
manifest entry block all PR submission.