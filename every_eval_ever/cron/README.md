# Daily adapter refresh

A scheduled refresh of the source adapters. Each adapter runs on its own, stores
what it fetched permanently, marks its records as cron-produced, validates them,
and commits them to that adapter's own pull request on the
[datastore](https://huggingface.co/datasets/evaleval/EEE_datastore).

Workflow: [`.github/workflows/adapter_cron.yml`](../../.github/workflows/adapter_cron.yml).

## Running it by hand

```bash
uv run python -m every_eval_ever.cron list

uv run python -m every_eval_ever.cron run vals_ai \
    --work-dir /tmp/cron-vals-ai \
    --dry-run
```

`--dry-run` does everything except the commits, which makes it the way to check
an adapter before letting the schedule near it. Exit codes: `0` published, `3`
nothing new to publish, `1` failed. (`3` rather than `2`: argparse exits `2` on
a usage error, and the workflow treats the nothing-new code as success — a flag
typo must fail loudly, not read as a quiet day.) Generated records go under
`<work-dir>/data/`, raw payloads under `<work-dir>/raw/`, and the run summary to
`<work-dir>/summary.json`.

Publishing for real needs `HF_TOKEN` with write access to both the datastore and
the private raw dataset. `--no-archive-raw` keeps a local run from writing to the
raw dataset at all.

Check the credentials without running anything:

```bash
uv run python -m every_eval_ever.cron preflight
```

It reports the token identity, whether both destinations are reachable, creates
the private raw dataset if it is missing, and names any adapter held back by a
missing API key. The scheduled workflow runs this first, because the refresh
fails closed — a token that cannot store raw data would otherwise fail every
adapter at its last step. Note that a `--dry-run` refresh does not archive, so
preflight is the only thing that exercises raw-dataset access without publishing.

## Adding an adapter to the schedule

Add a `CronAdapter` to `CRON_ADAPTERS` in [`schedule.py`](schedule.py). An
adapter directory that is in neither `CRON_ADAPTERS` nor `EXCLUDED_ADAPTERS`
fails `tests/test_cron_schedule.py`, so a new adapter has to be a decision
rather than an oversight.

The runner invokes an adapter with the scratch tree as its working directory and
lets it write to its own default `data/` path. It does not pass `--output-dir`,
because adapters disagree about whether that flag names the base directory or
the collection directory.

Declare `raw_policy` honestly — it is what the run reports as archived:

| Policy | Meaning |
|---|---|
| `VIA_FETCH_HELPERS` | Fetches through `helpers.fetch`, so the shared hook archives the response body as served |
| `VIA_ADAPTER_FLAG` | Archived by the adapter's own `--save-raw-*` flag, named in `raw_args` |
| `UPSTREAM_VERSIONED` | A HuggingFace dataset revision or a git commit; deliberately not re-archived |
| `NOT_CAPTURED` | The adapter calls an HTTP client directly and exposes no flag — a gap |

A `VIA_ADAPTER_FLAG` dump is archived but does **not** decide whether the source
moved. Those files are derived: `hle`'s wraps the payload in a `fetched_at`
timestamp and `vals_ai`'s is a normalized form, so comparing them across runs
reports a change every single day. Only a body stored exactly as the server sent
it counts, which in practice means the shared hook.

## What the cron guarantees

**One adapter cannot break another.** Every adapter is a separate workflow job
with its own timeout, so a hang or a failure is confined to it. Within one
adapter, an invocation that fails does not stop the remaining ones, and the
records that did convert are still published — an adapter reports a *partial*
conversion by writing every valid record and then exiting non-zero, so a
non-zero exit is reported, never treated as a reason to discard data.

**Every record says where it came from.** `source_metadata.additional_details`
carries `type_of_addition: cron`, `cron_run_date`, `cron_adapter`, and
`cron_run_url`. Records whose inferred deployment axes came out `unknown` also
name them in `cron_unknown_inferred_fields`, which is how a later pass finds the
records still needing a real value.

**Raw data is kept permanently, and privately.** Payloads go to a private
dataset of their own, `evaleval/EEE_raw` (override with the `EEE_RAW_REPO_ID`
repository variable), laid out as:

```
blobs/<ab>/<sha256>.<ext>              # one copy per distinct payload, ever
ledger/<adapter>/<date>-<run>.jsonl    # one row per payload per run
reports/<adapter>/<date>-<run>/…       # adapter failure reports (raw rows)
state/<adapter>.json                   # what the last successful publish was
state/<adapter>.attempt.json           # a publish in flight (cleared on success)
```

Privacy is enforced, not assumed: preflight fails if the raw dataset is public,
the archive re-checks visibility immediately before every commit and refuses to
write to a public dataset, and the workflow's artifact carries only the run
summary. Captured payloads *and adapter failure reports* — which embed raw
source rows — go exclusively to the private dataset (`reports/<adapter>/…`),
because artifacts on a public repository are downloadable by anyone signed in.

The adapter subprocess never holds the cron's write-capable `HF_TOKEN`: it is
scrubbed from the child environment, which does all its fetching with source
credentials only. A source that genuinely needs authenticated Hugging Face
*read* access declares `source_hf_token` in the schedule and receives the
separate `EEE_SOURCE_HF_TOKEN` — a least-privilege read token — as its
`HF_TOKEN`.

Payloads are **content-addressed**, so a source that has not changed since
yesterday costs nothing but a ledger row — which is what makes keeping every day
affordable. The ledger is the index, and it is a dataset in its own right:

```python
from datasets import load_dataset

ledger = load_dataset(
    'json', data_files='ledger/**/*.jsonl', ...  # from evaleval/EEE_raw
)
```

Each row carries the adapter, run date, run URL, source URL, SHA-256, byte
count, and the blob it landed in — so any record traces back to the exact bytes
it came from. A row is written even for a payload too large to store, and even
for a run that published nothing, so the ledger says what was fetched on every
date rather than only on the days something changed.

Archiving happens **before** anything is published, and a failure to archive
stops the run. Records should not reach the datastore without their raw
provenance stored. The workflow also uploads `adapter_reports/` and the run summary as a 90-day
artifact for debugging; raw payloads deliberately never leave the private
dataset.

**An unchanged source publishes nothing.** A run fingerprints the source and
stops if it matches the previous run — using the verbatim response bodies where
it has them, and otherwise the generated records with per-run values
(`retrieved_timestamp`, `evaluation_id`, the record UUID, the cron stamp)
stripped out. This is run-level, not record-level: a run publishes everything it
produced or nothing at all, so no individual record is ever dropped.
Record-level de-duplication is still open, and it is why the high-volume
`hfopenllm_v2` adapter is not scheduled yet.

The previous fingerprint is read from `state/<adapter>.json`, which is written
**only after a successful publish** — three properties hang off that one
sentence. A run can never compare against a fingerprint it wrote itself (the
ledger is written before publishing, which is why it cannot be the gate). A run
that failed to publish leaves the old state, so its records are retried the
next day instead of skipped as "unchanged" and silently lost. And a partial
publish records `partial: true`, which the next run treats as "publish
regardless", so the records that failed conversion get another attempt. A
*persistently identical* partial run — same output fingerprint **and** same
failure identity — is skipped, so a source that is half-broken for a month does
not re-add its successful records daily; any change on either side publishes.

State that is confirmed absent means a first run and publishes. State that
exists but cannot be read or parsed **fails the run** instead: guessing "first
run" there would republish an entire unchanged record set, and a failed run
simply retries tomorrow.

Every record published without its raw source archived is a provenance gap, so
a declared-capture adapter whose run produced records but recorded a capture
failure — or captured nothing at all — fails before publishing. Capture
failures survive the adapter subprocess as error rows in the manifest, and the
archive happens *before* the failure returns, so the successful sibling
captures and the error evidence are already permanent when the job goes red.

A publish is batched, and a batch can die midway with earlier batches already
on the pull request. Before the first batch, the run records the exact paths it
is about to add (`state/<adapter>.attempt.json`); a later run finds that
dangling attempt, deletes whichever of its files reached the pull request, and
republishes — one copy of each record, not a stack of retries. The attempt
record is cleared in the same commit that records the publish state, so the two
can never disagree. If recording the state fails even after a retry, the run
exits non-zero with the pull request URL in the summary: the records are
published, only the gate is stale, and the next run reconciles.

An enabled adapter missing its credential **fails its own job** rather than
being quietly dropped from the matrix — a red job names the missing variable,
while the other adapters' jobs proceed. Quiet skips are reserved for adapters
deliberately declared disabled.

## Known gaps

- `hal`, `lexam`, and `mt_bench` archive no raw data: they call an HTTP client
  directly and expose no raw-dump flag. They fall back to the output
  fingerprint, so they still will not republish unchanged records.
- `hle`, `terminal_bench_2`, and `vals_ai` archive raw data through their own
  flags, which is enough to keep it but not to compare runs, so they are gated
  on their output too. Routing them through `helpers.fetch` would fix that.
- Records accumulate in each adapter's pull request until someone merges it.
  Nothing here de-duplicates against what the datastore already holds.
- The workflow installs `--all-extras` because three adapters read HuggingFace
  datasets through `datasets`, which reaches the lock only via the `helm` extra.
  A dedicated extra would be lighter.
