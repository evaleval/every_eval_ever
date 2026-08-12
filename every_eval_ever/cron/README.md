# Scheduled adapter ingestion

A daily GitHub Actions run that refreshes one adapter at a time and sends each
adapter's records to its own Hugging Face pull request on the datastore.

```bash
uv run python -m every_eval_ever.cron list          # every adapter and its schedule
uv run python -m every_eval_ever.cron plan          # the matrix due today
uv run python -m every_eval_ever.cron run --adapter hle --dry-run
```

## What one run does

```
catalog  which adapters exist, their collections, argv, cadence and timeout
   ↓
runner    run the adapter into a private staging tree, with a timeout
          → only files at data/<collection>/<dev>/<model>/<uuid>.json count
          → the packaged validator must pass, with no warnings
          → fingerprint each record BEFORE stamping it
   ↓
provenance  mark survivors: type_of_addition=cron, run date, adapter, run URL
   ↓
store     snapshot the raw source and update this adapter's ledger
   ↓
submit    upload into this adapter's own datastore pull request
```

Nothing is shared between adapters. Each run stages into its own temporary
directory and uploads only what it validated, so one adapter's failure cannot
reach another adapter's pull request.

## Outcomes

| Status | Meaning | Job |
|---|---|---|
| `completed` | Clean refresh | green |
| `partial` | Adapter exited non-zero but accounted for every dropped row in a provenance report; its valid records are published | green, annotated |
| `skipped_missing_credential` | The adapter's API key is not configured | red |
| `skipped_missing_dependency` | An optional package (e.g. `datasets`) is not installed | green |
| `failed` | Crash, timeout, misplaced output, failed validation, duplicate records, an unsnapshotted source, or an empty refresh | red |

A missing credential is red because the adapter is in today's matrix only
because the catalog says it should run. Green, it is indistinguishable from an
unchanged leaderboard, which is how an adapter goes missing for a month. An
adapter that should not run at all is `runnable=False` in the catalog instead,
and never reaches a job. A missing *package* stays green: `with_packages` is
installed by the workflow from the matrix, so an absent one is a packaging
problem rather than a secret nobody added.

An empty refresh is deliberately a failure. "0 valid, 0 invalid" is what a
broken output directory looks like, not an up-to-date leaderboard.

## De-duplication

Each record is fingerprinted with `normalized_hash`, which ignores
`evaluation_id` and `retrieved_timestamp` — so a re-scrape of an unchanged
leaderboard fingerprints identically. The cron drops one more field first:
`detailed_evaluation_results.file_path` names the record's sample sidecar and
is written with a fresh UUID4 every conversion, so a record with instance data
would otherwise look new every day. Its `checksum` stays in the hash, so a
sidecar whose contents really changed still counts as a new record.
`normalized_hash` itself is untouched — inside a single batch, two records
naming different sample files are two records, and `check-duplicates` has to
keep seeing that.

Fingerprints already published are kept in `state/<adapter>.fingerprints` and
skipped on later runs.

Skipping is reversible and audited: every skipped record's model id and
fingerprint is listed in that run's `run.json`. To republish everything once,
use `--force-full`; to reset an adapter permanently, delete its
`state/<adapter>.fingerprints`.

The order matters. `cron_run_date` changes daily and is part of the record, so
fingerprints are taken *before* the provenance stamp; otherwise nothing would
ever look unchanged. `tests/test_cron_provenance.py` pins that.

## Raw source data

`every_eval_ever/helpers/raw_capture.py` snapshots what an adapter fetched.
It is inert unless `EEE_RAW_CAPTURE_DIR` is set, which the runner does, so
adapters behave identically when run by hand.

- Adapters using `helpers.fetch.fetch_json` / `fetch_csv` are captured with no
  adapter-side code at all.
- Adapters with their own single HTTP call site call `raw_capture.record(...)`
  there.
- Sources already addressable at a revision — Hugging Face datasets, git
  clones — record a pointer via `record_hf_dataset` / `record_git_checkout`
  instead of a second copy of bytes that are already durably hosted.

Payloads are content-addressed (`<sha256><ext>`) with a `manifest.jsonl`
beside them. A payload unchanged since the previous run is referenced, not
re-uploaded.

Capture never fails a conversion — an unwritable or oversized payload is
recorded as a `dropped` manifest line and the adapter carries on — but it does
fail the *run*. Records whose source was not kept cannot be checked against it
later, which is the whole reason for keeping it, and the case that hides is
the mixed one: two sources, the first snapshotted, the second over a cap, and
records that look complete from the outside. The manifest is still uploaded,
so the reason survives.

The caps are 64 MB per payload and 512 MB per run, overridable with
`EEE_RAW_CAPTURE_MAX_PAYLOAD_MB` and `EEE_RAW_CAPTURE_MAX_TOTAL_MB`. A source
that outgrows one turns that adapter's job red until the cap is raised, which
is the intended trade: a loud stop rather than a quiet gap in the archive.

## The raw store

```
evaleval/EEE_raw   (dataset, main)
  raw/<adapter>/<YYYY-MM-DD>/<sha256><ext>    payload bytes
  raw/<adapter>/<YYYY-MM-DD>/manifest.jsonl   one line per capture
  raw/<adapter>/<YYYY-MM-DD>/run.json         outcome, coverage, PR link
  state/<adapter>.json                        PR number, last run, last status
  state/<adapter>.fingerprints                one sha256 per published record
```

Everything one run writes lands in a single commit, guarded by the commit the
state was read at, so two overlapping runs collide loudly instead of one
silently overwriting the other.

## The pull request

One per adapter, titled `[Submission] cron: <adapter> — automated ingestion`,
with `eee-cron-adapter: <adapter>` in the body.

Two things identify it, and neither is the title. Only pull requests opened by
the account the token resolves to are candidates at all — the datastore is
public, so anyone can open one carrying our marker, and adopting it would
commit records onto a branch and a description a stranger controls. Among
those, the body marker says which adapter it belongs to. The number is
remembered in the ledger, and both it and any cold-start lookup are confirmed
against both checks before anything is uploaded.

So renaming a pull request does not strand it, and titling one to look like
ours does not hand it our records. Merged or closed means a fresh one is
opened. If two open pull requests claim the same adapter the run stops rather
than guessing.

The description is rewritten each run with the coverage line — source rows,
records produced, dropped, skipped as unchanged, uploaded.

## Setup

1. Create the raw store dataset (default `evaleval/EEE_raw`), private. It holds
   whole source payloads, so it is not meant to be browsable; the token in the
   next step needs access to it.
2. Add a `cron` environment to the GitHub repository with an `HF_TOKEN`
   secret that can write to the raw store and open pull requests on the
   datastore.
3. Per-adapter credentials as secrets — `uv run python -m
   every_eval_ever.cron list` shows which adapters want one. Every adapter
   that names one needs it: without it that adapter's job fails, naming the
   variable, while the rest of the matrix carries on.
4. Optional `EEE_DATASTORE_REPO_ID` / `EEE_RAW_REPO_ID` repository variables
   to point a rehearsal at throwaway repositories.

## Adding an adapter to the schedule

Add an `AdapterSpec` to `every_eval_ever/adapters/catalog.py`. The adapter
must accept `--output-dir` and expose `parse_args(argv)`;
`tests/test_adapter_catalog.py` checks the entry against the adapter's own
parser and fails if any adapter package is neither registered nor listed as
legacy.

Give heavy adapters `cadence='weekly'` with a `weekday`, and a realistic
`timeout_minutes` — it bounds both the subprocess and the GitHub job.

## Operating it

```bash
# What would this adapter publish today? Nothing is uploaded.
uv run python -m every_eval_ever.cron run --adapter hle --dry-run \
    --workdir /tmp/eee-cron

# Inspect what a dry run produced.
uv run python -m every_eval_ever validate '/tmp/eee-cron/staging/data/*/*/*/*.json*'
cat /tmp/eee-cron/raw/manifest.jsonl

# Republish everything for one adapter, ignoring the ledger.
uv run python -m every_eval_ever.cron run --adapter hle --force-full
```

A run without a Hugging Face token stops and says so rather than quietly
becoming a dry run. Not publishing has to be asked for with `--dry-run`,
because a missing or expired secret would otherwise leave the scheduled job
green while it published nothing, which is indistinguishable from an
unchanged leaderboard until somebody goes looking.

## Known limitations

This is a deliberate MVP.

- `bfcl`, `cocoabench` and `sciarena` are registered but not schedulable:
  they need a local input file and have no live fetch path. `exgentic` is
  also parked — its upstream Hugging Face dataset no longer resolves.
  `uv run python -m every_eval_ever.cron list` shows each reason; re-enabling
  one is a single field in the catalog.
- Most adapters key `evaluation_id` on the scrape time, so a changed record
  arrives as a new file rather than an update to the previous one. The
  fingerprint ledger stops unchanged records piling up; reconciling genuine
  updates against records already in the datastore is not attempted here.
- Records already in the datastore are not backfilled with the cron marker.
- The cron never merges its own pull requests.
