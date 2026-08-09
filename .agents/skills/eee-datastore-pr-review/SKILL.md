---
name: eee-datastore-pr-review
description: >-
  Review and repair pull requests on the evaleval/EEE_datastore Hugging Face
  dataset. Use when given an EEE_datastore discussion or PR URL, asked to run
  or reproduce `/eee validate changed`, resolve EEE validator errors or
  warnings, research model deployment_type or model_availability, edit the
  changed datastore records, prepare human-approved deployment metadata
  proposals, distinguish record omissions from genuinely unavailable metadata,
  rerun the bot, or prepare canonical-registry follow-ups.
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
- Distinguish absent record metadata from unavailable source evidence for every field.
  Describe an unchecked absence as "not surfaced in the submitted record," not as
  missing from the underlying evaluation. A missing or null
  `model_info.additional_details` object means the record needs investigation; it does
  not establish either axis as `unknown`.
- Keep work on the supplied `refs/pr/<number>` ref. Do not open a replacement PR for
  another repair round.
- If asked only to review, prepare a patch and findings without uploading or
  commenting. If asked to fix, update the supplied PR, trigger its validator, and
  iterate on that same ref.
- Before changing `deployment_type` or `model_availability` on the live PR, obtain
  explicit human approval of a proposal bound to the current PR head. Authorization
  to fix the PR is not approval of research-derived field values.
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

Read `reference/metadata-missingness.md` whenever a field is absent, null, defaulted,
or claimed to be unavailable. Apply it to deployment metadata and reproducibility
fields such as temperature and maximum output tokens.

Read `reference/model-deployment.md` whenever either model deployment axis is
missing, stale, invalid, or suspicious. Read
`../eee-dataset-conversion/reference/registry.md` when an id is unresolved or a
registry update is requested. Load the full `eee-dataset-conversion` skill when the
repair also changes an adapter or regenerated output.

Re-read the allowed deployment values from
`every_eval_ever/validator/validation_core.py` and the live schema. Existing records
and old bot comments may use obsolete vocabularies.

## Progress checkpoints

Emit an incremental checkpoint to the caller at every boundary below. Checkpoints are
run receipts, not Hugging Face discussion comments: do not post them to the PR unless
the operator explicitly asks. Each checkpoint must include the phase, current PR head
SHA, facts established since the prior checkpoint, affected file/model counts, command
exit statuses or evidence URLs when applicable, blockers, and the next action.
When a progress or parent-message channel is available, send the checkpoint through it
and continue in the same run. Do not end a turn merely to deliver a checkpoint.

Required checkpoints:

1. **Snapshot:** after selecting the PR head and matching bot run.
2. **Diagnosis:** after reproducing the gate and grouping its findings.
3. **Research proposal:** after resolving model-specific evidence. Render
   `assets/deployment-metadata-proposal.md`, report its SHA-256, and pause for explicit
   human approval before editing deployment fields or mutating the live PR.
4. **Local repair:** after the repaired diff passes local validation and content review.
5. **Remote receipt:** immediately after each uploaded commit or validator-trigger
   comment, including the returned commit SHA or discussion event id.
6. **Bot result:** after each completed bot run, tied to its head/fingerprint; repeat
   diagnosis and repair checkpoints for another iteration.

For a phase lasting more than 60 seconds, emit a heartbeat at least once per minute
with the current evidence surface or bounded poll, completed/remaining counts, and
whether local or remote state changed. Use bounded polling calls of at most 45 seconds
so progress messages can be delivered. Continue after ordinary checkpoints. The
research proposal is an approval gate; do not continue past it without an explicit
approval matching both its digest and PR head.

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

Classify every apparent omission with `reference/metadata-missingness.md`. Do not call
an absent field genuinely missing while its status is `record_absent` or
`research_incomplete`. If a README, eval card, methodology page, leaderboard, paper,
repository, or API exposes it, classify it as `available_not_surfaced` and identify
the adapter/submission gap.

Inspect content even when the validator omits it. At minimum check suspicious zeroes,
score scale and bounds, metric identity, `source_data`, duplicate overall/subtask
aggregates, stable `evaluation_id`, model identity, answer leakage, and companion
pairing. An out-of-range score requires finding the source scale or source value; do
not cap, clamp, or round it into validity.

Inspect the raw JSON before constructing an `EvaluationLog`. The model layer may
auto-fill absent deployment keys with `unknown`, hiding whether the contributor
actually supplied `additional_details`, supplied only one axis, or supplied neither.

### 4. Research ambiguous metadata

Apply `reference/metadata-missingness.md` before deciding any absent field is truly
unavailable. For deployment warnings, then apply `reference/model-deployment.md` to
each exact model variant and evaluation run. Determine the two axes independently. Do
not infer one from the other, from the developer folder, or from a provider-wide rule.

Search all relevant primary surfaces before choosing `unknown`: record payload and
run config, generating adapter, pinned model card, evaluator methodology, paper and
appendix, source repository, and official API/release documentation. Use current web
research where facts may have changed, but pin the evidence revision or date relevant
to the submitted evaluation.

Batch models only after proving that they share the same evidence.

Before editing either deployment axis:

1. Copy `assets/deployment-metadata-proposal.md` to run notes outside the datastore
   repository and fill one row per exact submitted `model_info.id`. Do not substitute
   a folder slug or an unapproved canonical id.
2. Use only the five table columns in the template. Reference sources as `S1`, `S2`,
   and so on; list each full URL and pinned revision/date once below the table. Record
   deployment (`D`) and availability (`A`) confidence and sources separately inside
   their shared cells.
3. Include every model whose deployment fields would change, including mechanical
   vocabulary migrations. Propose `unknown` only for `conflicting_sources` or
   `unavailable_after_search`, and complete the template's unknown-rationale table.
   If any axis is `research_incomplete`, do not finalize the proposal.
4. Compute the completed file's SHA-256. Return the rendered Markdown (or a clickable
   path in a shared workspace), digest, PR head, model count, and affected-file count.
5. Stop and request explicit human approval of that digest at that PR head. Do not edit
   the records, upload a commit, or comment on the Hugging Face discussion while
   approval is pending.

Approval covers only the exact table and head named by the human. Before applying it,
re-fetch `refs/pr/<number>`. If the head changed, evidence changed, a proposed value
changed, or a new model entered scope, regenerate the artifact and obtain approval
again. Record the approver and approval time in the decision log.

### 5. Make the repair

- For deployment metadata, begin only after the research proposal is explicitly
  approved and the remote head still matches it. Apply only its approved rows.
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
- Treat `available_not_surfaced` as an extraction defect: backfill the approved value
  in the data repair and prepare an adapter/submission follow-up so regeneration does
  not erase it.
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

If the commit changes either deployment axis, require the approved proposal digest and
head in the decision log before uploading. General authorization to fix is insufficient.

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
- missingness classification for each absent field, including source-available
  extraction gaps and the surfaces checked before any unavailable claim;
- deployment/availability evidence table, including researched `unknown` values;
- research proposal path or rendered table, SHA-256, approved head, approver, and
  approval time;
- registry and adapter follow-ups with cross-links or candidate tables;
- decision log and any unresolved blocker.

Do not call a PR complete merely because all files parse or the bot prints “Ready to
Merge.”
