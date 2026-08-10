---
name: eee-datastore-pr-review
description: >-
  Review and repair pull requests on the evaleval/EEE_datastore Hugging Face
  dataset. Use when given an EEE_datastore discussion or PR URL, asked to run
  or reproduce `/eee validate changed`, resolve EEE validator errors or
  warnings, research model deployment_type or model_availability, edit the
  changed datastore records, rerun the bot, or prepare canonical-registry
  follow-ups.
---

# Review and repair an EEE datastore PR

Produce the smallest source-backed change that makes the existing PR both
validator-clean and semantically correct. Treat a green validator as necessary,
not sufficient.

## Operating contract

- Treat the current checkout's schemas and `REGISTERED_CHECKS` as the local source
  of truth. Treat the newest bot result for the current PR head as the remote gate.
- Report what the source establishes. Never invent metadata, clamp a score into its
  declared bounds, or change a value merely to silence the validator.
- Make `unknown` a researched conclusion, not a default. Record which relevant
  surfaces were checked before retaining it.
- Distinguish absent record metadata from unavailable source evidence. A missing or
  null `model_info.additional_details` object means the record needs investigation;
  it does not establish either axis as `unknown`.
- Keep work on the supplied `refs/pr/<number>` ref. Do not open a replacement PR for
  another repair round.
- If asked only to review, prepare a patch and findings without uploading or
  commenting. If asked to fix, update the supplied PR, trigger its validator, and
  iterate on that same ref.
- Ask the operator before a policy decision: minting a new canonical id, changing a
  schema/validator rule, dropping non-trivial data, choosing an ambiguous metric or
  bound, or making another structural change. Do not hide such a choice in a data
  repair.

## Load the live EEE rules

Before editing, read these sibling references:

- `../eee-dataset-conversion/reference/datastore-gate.md`
- `../eee-dataset-conversion/reference/fields.md`
- `../eee-dataset-conversion/reference/datastore-submission.md`
- `../eee-dataset-conversion/reference/verification.md`

Read `reference/model-deployment.md` whenever either model deployment axis is
missing, stale, invalid, or suspicious. Read
`../eee-dataset-conversion/reference/registry.md` when an id is unresolved or a
registry update is requested. Load the full `eee-dataset-conversion` skill when the
repair also changes an adapter or regenerated output.

Re-read the allowed deployment values from
`every_eval_ever/validator/validation_core.py` and the live schema. Existing records
and old bot comments may use obsolete vocabularies.

## Workflow

### 1. Establish the exact PR state

1. Parse the dataset repo and discussion number from the supplied URL.
2. Fetch the discussion details, commit history, current head, base ref, file diff,
   conflicts, and every validator comment. Prefer Hugging Face's API or
   `huggingface_hub` over scraping rendered HTML.
3. Select only the newest completed bot run whose fingerprint or head matches the
   current PR. Older green runs describe older data or validator versions.
4. Check out `refs/pr/<number>` in a dedicated datastore worktree or temporary clone.
   Preserve the contributor's branch and unrelated changes.
5. Diff the PR head from its merge base with `main`. Inventory added, modified,
   renamed, and deleted paths; include aggregate/instance companions even if only one
   side appears in the diff.

Record the PR head commit and bot schema/compatibility version in the review notes.
If the bot and local schema differ, label their disagreement as version skew and
investigate it explicitly.

### 2. Reproduce the gate locally

Run the current EEE CLI against changed `.json` and `.jsonl` files at their final
`data/<collection>/<developer>/<model>/...` paths. Pass files or a quoted glob, never
a directory. Include companion files required by semantic validation.

Use:

```text
uv run python -m every_eval_ever validate <changed files>
uv run python -m every_eval_ever.check_duplicate_entries <relevant files>
```

Capture the full output and exit status. Do not rely on Pydantic model construction
or `validate_file()` alone; those can omit semantic checks. If current `main` and the
deployed bot disagree, reproduce both versions when practical and fix toward the
current schema without silently degrading data for an old bot.

### 3. Triage before editing

Group findings by root cause rather than by file. For each group, record:

- affected paths and exact model/result identities;
- local and bot messages;
- whether the issue is mechanical, schema-semantic, or content-semantic;
- the source evidence needed for a correct fix;
- proposed change and confidence.

Inspect content even when the validator omits it. At minimum check suspicious zeroes,
score scale and bounds, metric identity, `source_data`, duplicate overall/subtask
aggregates, stable `evaluation_id`, model identity, answer leakage, and companion
pairing. An out-of-range score requires finding the source scale or source value; do
not cap, clamp, or round it into validity.

Inspect the raw JSON before constructing an `EvaluationLog`. The model layer may
auto-fill absent deployment keys with `unknown`, hiding whether the contributor
actually supplied `additional_details`, supplied only one axis, or supplied neither.

### 4. Research ambiguous metadata

For deployment warnings, apply `reference/model-deployment.md` to each exact model
variant and evaluation run. Determine the two axes independently. Do not infer one
from the other, from the developer folder, or from a provider-wide rule.

Search all relevant primary surfaces before choosing `unknown`: record payload and
run config, generating adapter, pinned model card, evaluator methodology, paper and
appendix, source repository, and official API/release documentation. Use current web
research where facts may have changed, but pin the evidence revision or date relevant
to the submitted evaluation.

Batch models only after proving that they share the same evidence. Keep an evidence
table with raw model label, canonical model id, both decisions, source URL/revision,
and confidence.

### 5. Make the repair

- Edit only files implicated by a finding. Avoid mass reformatting unrelated data.
- Preserve UUID filenames and stable evaluation identities unless identity itself is
  the defect.
- When `model_info.additional_details` is absent or null, create the object only after
  researching both axes. When it already exists, merge the researched keys without
  discarding unrelated source metadata.
- Keep `additional_details` values as strings. Add concise evidence/provenance there
  when the source has no typed home and the decision would otherwise be opaque.
- If generated records are wrong, fix or prepare the generating adapter in the code
  repo as well; otherwise the next refresh will restore the defect. Keep adapter code
  out of the datastore PR and cross-link its separate PR.
- Review the resulting diff for accidental deletion, unrelated churn, and a mechanical
  replacement applied to semantically different models.

### 6. Verify the repaired head

Rerun the local validator and duplicate checker, then repeat the content spot-check.
Require every changed file and companion to pass. Review warnings even if the command
or bot says “Ready to Merge.”

Compare the final changed-path inventory with the initial inventory. Explain every
new path, deletion, identity change, or source-value change in the decision log.

### 7. Update and monitor the existing PR

When the task authorizes a fix, upload exact add/delete operations to the existing
`refs/pr/<number>` with `huggingface_hub.HfApi.create_commit`; set the current PR head
as `parent_commit` so concurrent updates fail instead of being overwritten. Never set
`create_pr=True` for a repair round.

After the commit lands:

1. Comment `/eee validate changed` on the same discussion with
   `HfApi.comment_discussion`.
2. Monitor until a completed run matches the new head/fingerprint.
3. Re-read every error and warning, repair locally, and repeat on the same ref.
4. Stop only when both the current local CLI and matching bot run are clean, or when a
   genuine policy/ambiguity/auth/conflict blocker needs the operator.

Do not post claims or comments on the contributor's behalf during a review-only task.

### 8. Handle registry work without inventing a registry

Resolve model, benchmark, metric, harness, and organization ids against the registry
when a resolver or registry checkout exists. Search existing canonicals and aliases
before proposing anything new.

If the registry repository and its contribution workflow are available:

1. Read its `AGENTS.md`, `CONTRIBUTING.md`, and registry skill.
2. Add an alias to an existing canonical when evidence supports it.
3. Ask the operator before deliberately creating a new canonical.
4. Validate in that repo and open a separate registry PR; cross-link it with the data
   and adapter PRs.

If the registry is unavailable or not yet implemented, do not invent its file format.
Emit a registry-candidate table in the review report with entity type, raw value,
candidate canonical, evidence, confidence, and whether the candidate is an alias or a
new entity. Leave the datastore value source-faithful and mark resolution status
explicitly.

## Completion report

Return:

- PR URL, starting head, final head, and matching bot run/version;
- files changed, grouped by root cause;
- local validation and duplicate-check results;
- content spot-checks performed;
- deployment/availability evidence table, including researched `unknown` values;
- registry and adapter follow-ups with cross-links or candidate tables;
- decision log and any unresolved blocker.

Do not call a PR complete merely because all files parse or the bot prints “Ready to
Merge.”
