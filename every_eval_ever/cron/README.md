# Scheduled adapter ingestion

A daily GitHub Actions run that refreshes one adapter at a time and commits
each adapter's validated records straight to the datastore.

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
provenance  mark survivors: type_of_addition=cron, run date, adapter, run URL,
            and which inferred model fields came out unknown
   ↓
store     snapshot the raw source and update this adapter's ledger
   ↓
submit    commit to the datastore, one commit series per run
```

Nothing is shared between adapters. Each run stages into its own temporary
directory and uploads only what it validated, so one adapter's failure cannot
reach another adapter's records.

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

## What a published record carries

In `source_metadata.additional_details`:

| Key | Value |
|---|---|
| `type_of_addition` | `cron` |
| `cron_run_date` | UTC date the source was pulled, `2026-08-10` |
| `cron_adapter` | the catalog key that produced it |
| `cron_run_url` | link to the workflow run, when there is one |
| `cron_unknown_inferred_fields` | which of `deployment_type` and `model_availability` came out `unknown`, comma-separated; absent when both are known |

The ticket asks for those last two to stay `unknown` for now and be fixed
later. Naming them per record is what makes later a filter rather than a
re-read of the whole datastore. Both default to `unknown` in the schema, so a
record that omits one is saying the same thing as one that spells it out.

## De-duplication

Each record is fingerprinted with `normalized_hash`, which ignores
`evaluation_id` and `retrieved_timestamp`, so a re-scrape of an unchanged
leaderboard fingerprints identically. The cron drops one more field first:
`detailed_evaluation_results.file_path` names the record's sample sidecar and
is written with a fresh UUID4 every conversion, so a record with instance data
would otherwise look new every day. Its `checksum` stays in the hash, so a
sidecar whose contents really changed still counts as a new record.
`normalized_hash` itself is untouched: inside a single batch, two records
naming different sample files are two records, and `check-duplicates` has to
keep seeing that.

Fingerprints are kept in two files with different meanings, and both are
skipped on later runs. `state/<adapter>.fingerprints` holds records that are
in the datastore; a record's fingerprint joins it the moment its commit
lands. `state/<adapter>.pending` holds records the retired pull-request flow
committed to an adapter's open pull request before runs published directly;
new runs write nothing to it. While any remain, each run settles them against
that pull request's fate: merged promotes them into the durable ledger,
closed-without-merging drops them so the records are resubmitted instead of
being filtered out of every later run by fingerprints for data the datastore
never accepted.

Skipping is reversible and audited: every skipped record's model id and
fingerprint is listed in that run's `run.json`. To republish everything once,
use `--force-full`; to reset an adapter permanently, delete its
`state/<adapter>.fingerprints` and `state/<adapter>.pending`.

A third file, `state/<adapter>.inflight`, holds records between the moment a
run commits its snapshot and the moment it records what it published. See
"The raw store" below for what a non-empty one means at the start of a run.

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
- Sources already addressable at a revision, such as Hugging Face datasets and
  git clones, record a pointer via `record_hf_dataset` / `record_git_checkout`
  instead of a second copy of bytes that are already durably hosted.

Payloads are content-addressed (`<sha256><ext>`) with a `manifest.jsonl`
beside them. A payload unchanged since the previous run is referenced, not
re-uploaded.

Capture never fails a conversion: an unwritable or oversized payload is
recorded as a `dropped` manifest line and the adapter carries on. It does fail
the *run*. Records whose source was not kept cannot be checked against it
later, which is the whole reason for keeping it, and the case that hides is
the mixed one: two sources, the first snapshotted, the second over a cap, and
records that look complete from the outside. The manifest is still uploaded,
so the reason survives.

No manifest at all fails the run too, unless the catalog entry says
`captures_raw=False`. Every adapter that fetches live sources writes one
through the paths above, so its absence means the capture hooks never ran:
a sink that was unwritable from the first byte, or an adapter fetching around
the shared helpers. Records without kept source bytes must not publish just
because the failure was total rather than partial. The exemption exists for
adapters that convert local files and have nothing to snapshot.

The caps are 64 MB per payload and 512 MB per run, overridable with
`EEE_RAW_CAPTURE_MAX_PAYLOAD_MB` and `EEE_RAW_CAPTURE_MAX_TOTAL_MB`. A source
that outgrows one turns that adapter's job red until the cap is raised, which
is the intended trade: a loud stop rather than a quiet gap in the archive.

## The raw store

This layout is the contract. Post-fix tooling and any later de-duplication
work read these paths, so treat a change to them as a change other things
depend on.

```
evaleval/EEE_raw   (private dataset, main)
  raw/<adapter>/<date>/<run>/<sha256><ext>    payload bytes
  raw/<adapter>/<date>/<run>/manifest.jsonl   one line per capture
  raw/<adapter>/<date>/<run>/run.json         outcome, coverage, records committed
  state/<adapter>.json                        last run, last status
  state/<adapter>.fingerprints                one sha256 per published record
  state/<adapter>.pending                     one sha256 per record the retired
                                              flow left waiting on a pull request
  state/<adapter>.inflight                    records this run is about to
                                              publish, written before it does
```

`<adapter>` is the catalog key, which is also the job name and the name in the
datastore commit messages.

`<date>` is the UTC run date and `<run>` names the run within it:
`run-<workflow run id>-<attempt>` under Actions, `local-<HHMMSS>` otherwise,
overridable with `--run-id`. Two runs of one adapter on one day are ordinary,
a cancelled job or a source that was down at 03:17 followed by a manual re-run,
and both write a manifest and a report at fixed names. Under a date alone the
second overwrote the first's account of what it fetched. Payload bytes are
content-addressed, so nothing is stored twice; only the two files that describe
a run are per run. `state/<adapter>.json` carries `last_raw_prefix`, the whole
path, since a date no longer names one directory.

A run makes two commits here, one either side of publishing to the datastore,
each guarded by the commit the previous one left, so two overlapping runs
collide loudly instead of one silently overwriting the other.

The first carries the raw snapshot and `state/<adapter>.inflight`: the
fingerprint and datastore paths of every record this run is about to publish.
The second carries the run report, the ledger and an emptied in-flight file,
with one exception: records whose batch errored while the datastore was
unreadable stay in flight, because whether they landed is exactly the question
the file exists to answer, and the next run settles it. Publishing is the one
step a re-run cannot undo, so it happens between the two commits rather than
before both. A run that uploads records and then fails to write
its ledger, because the job was cancelled or the raw store was briefly
unreachable, would otherwise leave them in the datastore with nothing
naming them, and the next run would publish the same evaluations again under
fresh UUID paths.

A non-empty in-flight file at the start of a run is exactly that case. Each
record it names is checked against where it was headed — the datastore, or
for a file the retired pull-request flow wrote, the pull request it was
publishing into: the ones that arrived are recorded, the ones that did not
are published again. A check the Hub cannot answer stops the run, since one
wrong guess buries records and the other duplicates them. Settling the same
file twice settles it the same way, so a run that dies before its own commit
costs nothing.

## Publishing

Records go straight to the datastore's default branch: one commit series per
adapter run, batched at 300 files per commit so a single giant commit cannot
time out ambiguously, and batches never split a record across two commits, so
each commit publishes whole records or none of them. The review that used to
happen on a pull request happens before publication instead — the packaged
validator must pass with no warnings, the ledger drops records already
published, and every record carries `type_of_addition: cron` with its run's
date, adapter and workflow URL — so a human reviewing after the fact can find
and correct any batch. Each commit's message is
`cron: <adapter> <date> (<n> record(s))` and its description carries what the
pull request body used to: run date, status, coverage line, raw snapshot path
and workflow run.

A commit that errors after the Hub accepted it is adopted rather than
repeated: the datastore's file listing arbitrates, a batch proven present
counts as committed, and the upload carries on. A batch that provably never
landed fails the run with everything earlier recorded in the ledger, so the
retry publishes only the remainder. A batch that cannot be checked either way
stays in flight (see above) for the next run to settle.

### What remains of the pull-request flow

Earlier versions published into one open pull request per adapter, identified
by the opening account plus an `eee-cron-adapter: <adapter>` line in the
body, with fingerprints held pending until a reviewer merged. Runs now settle
whatever that flow left behind: pending fingerprints are promoted when their
pull request turns out merged and dropped when it was closed without merging,
and an old in-flight file is checked against the pull request it named. The
settling machinery in `submit.py` can be deleted once no adapter's state
names a pull request.

## Setup

1. Nothing, if the token in the next step can create datasets: the first run
   creates the raw store (default `evaleval/EEE_raw`) as a **private** dataset
   and reads it back to confirm. Create it by hand if you would rather, but
   create it private.

   Privacy is enforced, not requested. The store holds whole source payloads,
   kept so a published record can be checked against what it was converted
   from. Republishing them is a different thing that nobody agreed to. Every
   run checks before it starts and again immediately before each commit, and
   refuses to write to a public dataset. It never changes a repository's
   visibility for you: that decision has consequences for anyone already
   reading it, and making it silently at 03:17 is not the cron's call. A
   `repo_info` call that fails for any reason other than "not found" also
   stops the run, because a 500 is not evidence that a public dataset is
   absent.
2. Add a `cron` environment to the GitHub repository with an `HF_TOKEN`
   secret that can write to the raw store and to the datastore. A read-only
   token is rejected before the adapter starts, as is a datastore this token
   cannot read.
3. Per-adapter credentials as secrets. `uv run python -m
   every_eval_ever.cron list` shows which adapters want one. Every adapter
   that names one needs it: without it that adapter's job fails, naming the
   variable, while the rest of the matrix carries on.
4. Optional `EEE_DATASTORE_REPO_ID` / `EEE_RAW_REPO_ID` repository variables
   to point a rehearsal at throwaway repositories.

Everything answerable without running the adapter is answered first: the
token resolves to an account and is not read-only, the datastore is readable,
and the raw store exists and is private. All of it would surface later anyway,
at the publish step, but by then a leaderboard has been scraped for
forty-five minutes for nothing. A role the Hub does not report is not treated
as read-only, since only a commit proves a fine-grained token's scopes.

## Adding an adapter to the schedule

Add an `AdapterSpec` to `every_eval_ever/adapters/catalog.py`. The adapter
must accept `--output-dir` and expose `parse_args(argv)`;
`tests/test_adapter_catalog.py` checks the entry against the adapter's own
parser and fails if any adapter package is neither registered nor listed as
legacy.

Give heavy adapters `cadence='weekly'` with a `weekday`, and a realistic
`timeout_minutes`, which bounds the adapter subprocess. The GitHub job gets
that plus `JOB_TIMEOUT_BUFFER_MINUTES`, because the job also checks out the
repository, installs the environment, uploads the snapshot and commits the
records. A job cancelled during that last part is the one case where records
reach the datastore with nothing in the ledger recording them.

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
  they need a local input file and have no live fetch path. `hfopenllm_v2`
  is parked because the Open LLM Leaderboard is no longer maintained
  upstream, so a refresh of its frozen archive would fetch nothing new.
  `uv run python -m every_eval_ever.cron list` shows each reason;
  re-enabling one is a single field in the catalog.
- `mercor_eval` is paused (`runnable=False`): its Exports API is broken
  upstream, so a scheduled run has nothing to fetch. Flip `runnable` back
  on in the catalog once Mercor is stable. The `allow_source_outage`
  machinery remains available for a source that is flaky rather than down.
- The five `helm_*` units and `rewardbench` are paused for staleness, not
  breakage: their upstreams still serve but have stopped updating, so a
  weekly refresh refetches unchanged data. Each is one `runnable` flip away
  from rejoining the schedule if its upstream resumes publishing.
- Most adapters key `evaluation_id` on the scrape time, so a changed record
  arrives as a new file rather than an update to the previous one. The
  fingerprint ledger stops unchanged records piling up; reconciling genuine
  updates against records already in the datastore is not attempted here.
- Records already in the datastore are not backfilled with the cron marker.
- Published records pass the packaged validator and carry provenance, but
  nothing reviews them for sense before they land; a bad batch is found and
  corrected after the fact via its `cron_*` provenance fields.
- A source that outgrows a capture cap fails its adapter until the cap is
  raised. Nothing splits or streams a large payload.
- The write token is checked for a reported read-only role, which a
  fine-grained token need not report. Only a commit proves those scopes, so
  such a token still fails at the publish step rather than up front.
