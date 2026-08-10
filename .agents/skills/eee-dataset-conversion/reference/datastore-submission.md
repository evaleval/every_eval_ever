# Submitting to the `EEE_datastore` HF dataset

**The repo `README.md` §Contributor Guide is the source of truth for the mechanics** —
the `data/<collection>/<developer>/<model>/{uuid}.json` layout, the uuid convention, the
`HfApi().upload_folder(..., create_pr=True)` call, adding files to an existing
`refs/pr/<n>`, and the `/eee validate changed` bot command. Read it; don't re-derive it.

*This file adds only what the README doesn't say — the things that cost real submissions
real time. What makes a record valid is `datastore-gate.md`; what a field means is
`fields.md`.*

- **Collection naming is a decision, not a formality** — see `fields.md` §collection. A
  source with many sub-leaderboards needs collision-proof names (`<owner>__<slug>`), and
  the collection is derived from `evaluation_results[0].source_data.dataset_name` unless
  you pass `collection_override`.
- **Batch large uploads.** A single commit with thousands of files can 504 — and the
  commit may land server-side while the client errors, leaving a half-submitted PR to
  abandon. Upload in chunks of a few hundred and put `(n/N)` in the title so reviewers
  know the set is incomplete until the last batch lands.
- **Iterate on the same PR ref.** The README shows how; the reason matters. Opening a
  fresh PR per round of bot warnings is the single largest source of churn in this
  datastore's history — one submission took five PRs to clear one `deployment_type`
  warning — and reviewers lose the thread.
- **A "Ready to Merge" verdict can still carry warnings.** Clear them before asking for
  review: commonly missing `deployment_type`/`model_availability`, `hf_dataset` without
  `hf_repo`, or `other` with no URL provenance. The bot also reports its own
  **compatibility version**; if that differs from your local `SCHEMA_VERSION`, expect
  vocabulary skew (`datastore-gate.md` §deployment) and ask, rather than downgrading
  records to satisfy an older gate.
- **Adapter code never goes in the data PR.** It belongs in
  `every_eval_ever/adapters/<name>/` in the GitHub repo; reviewers ask for it every time
  data arrives without it. Cross-link the two PRs.
- **Generated records never go in the code repo.** Point smoke runs at a temp dir;
  writing into a checkout's `data/` is only for a deliberate refresh.

## What the PR description needs
- **Source** — the leaderboard/paper, the *dataset* the eval ran on, and a pinned
  revision (commit SHA / dataset revision / snapshot date), not a mutable `main`.
- **Coverage** — "N source rows → M records, K dropped (reason)". Expect to be asked
  "what about the other models?"; answer it up front, and name any cap or sample you
  applied. The `adapter_reports/<collection>_failures.json` your adapter wrote is the
  evidence for this line.
- **Cross-links** — the adapter PR in `every_eval_ever`, and any alias PR in
  `eval-card-registry`.
- **Decisions** — the non-obvious calls, the alternative you rejected, and your
  confidence (SKILL.md step 7). Low confidence is a request for maintainer attention, not
  an admission of failure.
- **Instances** — if you shipped `_samples.jsonl`, say why (and if you deliberately did
  not re-host public raw data, say that too).
