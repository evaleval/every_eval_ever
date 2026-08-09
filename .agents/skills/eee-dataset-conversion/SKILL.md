---
name: eee-dataset-conversion
description: >-
  Convert an evaluation dataset or leaderboard into the Every Eval Ever (EEE)
  schema — aggregate `.json` logs (eval.schema.json) and optional instance
  `_samples.jsonl` sidecars (instance_level_eval.schema.json). Use when asked to
  write an EEE adapter, add a dataset/leaderboard to the EEE datastore, map
  benchmark results into EEE, or debug why an EEE record won't validate.
license: MIT
metadata:
  version: 0.1.0
---

# Converting evaluation results into Every Eval Ever (EEE)

> **Rule you will keep relearning: all records must validate, but validating ≠
> correct.** Most real defects (answer leakage, double-counted aggregates,
> hardcoded scorers, non-idempotent ids) pass the schema and are still wrong.
> Always spot-check *content*, not just validity.

*Written against EEE `SCHEMA_VERSION` `0.3.0` (import it from
`every_eval_ever.helpers`; never hardcode). If that value has moved, re-verify the
field claims in `reference/` against the live schema — the schema always wins.
`tests/test_skill_conversion.py` pins this marker and re-validates this skill's
templates + frozen reference records, so a schema or validator change fails CI here
rather than in your PR. If that test is red, fix the skill, then regenerate the
frozen records.*

> **How this runs.** A person (the operator) runs you and can answer questions
> mid-run — you are **not fully autonomous**. When a choice *sets policy* (step 7's
> ask-list), ask the operator instead of deciding silently. Decide and log
> everything else. Finish with a PR that is ready to merge yet makes every
> non-obvious decision visible, so the maintainer who reviews it can comment and
> the skill/schema can improve. Two humans: the operator gates live; the PR informs
> the maintainer.

## When this skill applies
A source has model×benchmark scores (a leaderboard, a paper table, an HF results
dataset, a harness dump) and you must emit EEE records. Two artifacts:
- **Aggregate `.json`** — one `EvaluationLog` per model (or per model×benchmark),
  holding the headline scores. Always produced.
- **Instance `_samples.jsonl` — one record per example. Only if you have
  per-item data and want it.

## Workflow (do these in order)
1. Inspect the source first — you can't map fields you haven't seen. Establish:
   distinct models · benchmarks/subtasks · the metric and its range · is there
   per-item data · the harness · timestamps · provenance (paper + each
   benchmark's own dataset repo). These facts are usually spread across many
   surfaces and which lives where varies per dataset, so gather every *relevant*
   surface before recording a field as unknown** — see `reference/fields.md` §sources
   for the surface checklist, the coverage-vs-fill split, and which wins when they
   disagree. Filter hygiene junk (`.ipynb_checkpoints`, `*-checkpoint.json`) and
   segregate hand-curated baselines from harness runs.
2. **Decide the shape** — `source_type` is set by the artifact you hold, not who
   ran the compute: raw per-item outputs → `evaluation_run` (even if a third party ran
   them); only-aggregate reported numbers → `documentation` (a leaderboard scrape stays
   `documentation`). Then: aggregate-only vs +instances; grain (one log per model =
   default, or per model×benchmark when a benchmark has its own instance sidecar). See
   `reference/fields.md` §shape.
3. **Copy a template / reference adapter** — `templates/aggregate_adapter.py`
   (always) and, for per-item data, `templates/instance_sidecar.py` (runnable
   skeletons verified against the live validator). For a fuller real example,
   mirror `every_eval_ever/adapters/llm_stats` (aggregate/documentation),
   `.../hfopenllm_v2` (documentation, many models), or `.../openeval` (aggregate +
   instance sidecars). Adapters live at
   `every_eval_ever/adapters/<name>/adapter.py` and run as
   `uv run python -m every_eval_ever.adapters.<name>.adapter`; `__init__.py` just
   marks the package. Don't hand-roll the write path or the drop path — the repo
   owns both: publish through `save_evaluation_logs` (aggregate-only) or
   `converters.common.publication.publish_evaluation_logs` (with instance sidecars), and
   account for every rejected row via `SourceConversionResult` + `save_failure_report` +
   a non-zero exit. See `reference/datastore-gate.md` §publish.
4. **Fill fields carefully** — the field traps are the whole game. Load
   `reference/fields.md` (aggregate) and `reference/instance-level.md` (jsonl).
5. **Canonicalize ids** — model + benchmark ids must resolve in the
   eval-card-registry (else they fragment the data). Default: resolve live against
   the hosted resolver and use `canonical_id` for the join-key fields, with an opt-out
   flag + never-fatal fallback to the raw id (marked unverified). **But never key
   `evaluation_id` on the resolved id** — that's a moving join key; the record identity
   rides the raw source id. See `reference/registry.md`.
6. **Verify** — `python -m every_eval_ever validate <files>` (files/glob, **not** a
   dir), an offline unit test, ruff, a live smoke run, and a content spot-check.
   The validator's *semantic* checks run only on the CLI, and only when the file sits at
   its final `data/<collection>/<dev>/<model>/` path. They are the merge gate, listed in
   `reference/datastore-gate.md`. See `reference/verification.md`.
7. **Ask, then log your decisions.** Two channels, don't confuse them:
   - **Ask the operator (live)** when a choice *sets policy*: creating a new
     canonical id · dropping a non-trivial share of the data · an ambiguous metric
     choice · bounding an unbounded metric · re-hosting large data · **the source
     won't fit without a structural change** (a schema field, an edit to a base
     adapter or a shared converter, relaxing a validator rule). Don't decide these
     silently — the person running you is there to answer. The structural one is not
     yours to fold into this PR: its design gets agreed before a PR exists, and
     carrying it here would hold the adapter behind that discussion.
   - **Log (in the PR)** every *non-obvious* choice — not just where it was hard. A
     confident wrong choice produces no "friction," so log decisions, not pain.
   Finish with a ready-to-merge PR carrying the decision log below. General gaps
   (would recur on other datasets) also become a separate `skill`-labeled PR or a
   `skill-gap` issue — you needn't know the fix; flagging where you guessed is enough.

### Decision log (paste into the PR description)
- **Decision / where** — the field or step (e.g. `source_data` for a DB dump).
- **Chose / instead of** — what you did and the alternative you rejected.
- **Confidence** — high / medium / low (low = please, maintainer, look here).
- **General?** — `yes` (→ `skill`/`skill-gap` PR/issue) or `no` (dataset-specific).
- **Coverage** (once per adapter) — "N source rows → M records, K dropped (reason)".
  **No silent caps** — if you filtered/sampled/capped anything, say so here.

## Load a reference only when you need it (progressive disclosure)
| Read this | When |
|---|---|
| `reference/fields.md` | Filling any aggregate field; "which of the 3 `source_*` / 3 `*_name` fields?" |
| `reference/instance-level.md` | Emitting `_samples.jsonl`: required fields, the `interaction_type` XOR, `sample_hash`, `answer_attribution`, the sidecar write-order |
| `reference/gotchas.md` | Something validates but looks wrong; `inf`, double-counting, CI optional-deps, big-parquet reads |
| `reference/registry.md` | Model/benchmark ids won't resolve; adding aliases |
| `reference/datastore-gate.md` | What the CLI/bot enforce beyond the schema: paths, UUID4 names, companion pairing, score bounds, deployment axes, publishing |
| `reference/datastore-submission.md` | Opening/updating the HF datastore PR: batching, the `/eee validate` bot, iterating without opening a new PR |
| `reference/verification.md` | Before opening a PR; the checklist |

## The three PRs a contribution usually is
1. **Adapter code** → this repo (`every_eval_ever/adapters/<name>/adapter.py` +
   `__init__.py`, a `README.md` (recommended), `tests/test_<name>_adapter.py`, + a row
   in `every_eval_ever/adapters/README.md`). **Code only — no generated records here.**
2. **Canonical ids** → the `eval-card-registry` repo (aliases / new canonicals) —
   see its own `CONTRIBUTING.md` and the `registry-entity-aliases` skill there.
3. **Generated data** → the `EEE_datastore` HF dataset (`data/<collection>/`, via a PR
   with `HfApi().upload_folder(..., create_pr=True)`) — see
   `reference/datastore-submission.md` for batching and the review bot.
Cross-link them. Reviewers ask for the adapter whenever data arrives without it, so
open the code PR even when the conversion was a one-off script.

Schemas are the source of truth — when a reference and the schema disagree, the
schema wins; read `eval.schema.json` / `instance_level_eval.schema.json`.
