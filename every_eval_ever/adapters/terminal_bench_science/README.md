# Terminal-Bench-Science

Converts the [Terminal-Bench-Science](https://terminal-bench-science.ai)
leaderboard into `data/terminal-bench-science/`.

Terminal-Bench-Science evaluates agents on expert-curated scientific research
workflows — tasks authored and reviewed by domain scientists across the life,
physical, earth, mathematical and engineering sciences, run in a terminal
sandbox by [Harbor](https://harborframework.com). Release 0.1 is 70 tasks
attempted 3 times each, so one leaderboard row is **210 trials** of one
agent+model pair, and the published figure ("Resolution Rate") is the share of
those trials a task's verifier accepted.

```bash
uv run python -m every_eval_ever.adapters.terminal_bench_science.adapter \
    --output-dir /tmp/terminal-bench-science-smoke/data/terminal-bench-science
```

Needs no extras: the adapter uses only the core package and `requests`.

Only release 0.1 (`v0-1-eval`) of the
`terminal-bench-science/terminal-bench-science` package is supported. The CLI
and saved-payload replay reject other leaderboards rather than labeling them
as release 0.1.

## What each record holds

One `EvaluationLog` per leaderboard row, with six results:

| `evaluation_name` | Scope | `n` |
|---|---|---|
| `terminal-bench-science-0.1` | all 70 tasks | 210 trials |
| `terminal-bench-science-0.1.life-sciences` | life-science tasks | 57 trials |
| `terminal-bench-science-0.1.physical-sciences` | physical-science tasks | 51 trials |
| `terminal-bench-science-0.1.earth-sciences` | earth-science tasks | 24 trials |
| `terminal-bench-science-0.1.mathematical-sciences` | mathematical-science tasks | 51 trials |
| `terminal-bench-science-0.1.engineering-sciences` | engineering-science tasks | 27 trials |

The domains partition the overall result, so each one names its level in
`score_details.details.aggregation_level` (`overall` or `domain:<key>`) and
carries its own `n`. Take the overall or the parts, not both.

`score_details.details` also keeps what EEE has no typed field for: the trial
and pass counts behind the rate, the tasks/trials split, and the run's total
tokens and USD cost.

## Source

The leaderboard page reads
`https://terminal-bench-science.ai/api/leaderboard?package=<package>&name=<name>`,
a plain JSON GET, and so does this adapter. Harbor Hub serves the same
leaderboard at
[`hub.harborframework.com`](https://hub.harborframework.com/datasets/terminal-bench-science/terminal-bench-science/latest?tab=leaderboard),
but only through a Next.js server action whose action id changes on every
deploy, and its payload omits the per-domain metrics — so the Hub is recorded
as `source_data` provenance for the task set rather than used as the transport.

`source_type` is `documentation`: these are published aggregate numbers, not
raw per-item outputs. Per-trial pass/fail is available in the payload's
`task_matrix`, but the run transcripts, prompts and model outputs that the
instance-level schema requires are not, so no `_samples.jsonl` sidecar is
emitted.

The leaderboard carries no revision of its own, so `--emit-source-version`
prints a digest of the published row ids, statuses and metrics. It changes when
a row is added, restated or hidden, and not otherwise.

## Decisions worth a reviewer's eye

**The metric is namespaced, not the registry's `accuracy`.** The registry's
`accuracy` is a 0–1 proportion over items; this is the share of 3-trial
attempts a verifier accepted, published on a percent scale. `metric_id` is
`terminal-bench-science.accuracy` with bounds `[0, 100]`, matching the sibling
[`terminal_bench`](../terminal_bench/) adapter, so the two Terminal-Bench
leaderboards join with each other. The alternative — rescaling to 0–1 and
joining to global `accuracy` — would merge a trial-level resolution rate with
MCQ accuracy.

**Model ids are unverified.** The eval-card-registry has no model entry for the
releases on this leaderboard, and the source gives display labels
(`Fable 5.1`) rather than API ids. The organization half is canonicalized
through the registry (`Anthropic` → `anthropic`, `Z.AI` → `zai`); the model
half is the source's own label, slugified. Records say so:
`model_id_verified: "false"`, `model_id_source: "leaderboard_labels"`, plus the
`developer_registry_*` provenance. Resolution never fails the run.
`--no-registry-resolve` skips it entirely and records `registry_disabled`.

**`terminal-bench-science` and `harbor` are not canonical yet.** Neither the
benchmark nor the harness resolves, so `eval_library.additional_details`
carries `harness_registry_strategy: no_canonical` rather than implying one was
found. Both want a follow-up PR to
[`eval-card-registry`](https://github.com/evaleval/eval-card-registry).

**`model_availability` comes from a curated map.** The source states neither
deployment axis. Every row is an API-served model driven by an agent harness,
so `deployment_type` is uniformly `externally_managed`; weights availability is
read from `MODEL_AVAILABILITY_BY_ORG`, keyed by canonical org id, and an
organization missing from it yields `unknown` rather than a guess.

**`evaluation_id` is keyed on the leaderboard's row id**, not on
`retrieved_timestamp`, so a re-scrape of an unchanged leaderboard is
idempotent. The agent and reasoning effort ride in the id too, because one
model appears under several scaffolds and would otherwise collapse to a single
record.

**The standard error is the binomial one.** The published figure matches
`sqrt(p(1-p)/n)` on the percent scale over the trials, so it is recorded with
`method: "analytic"` and `num_samples` = the trial count. The test asserts this
against a hand-computed value, so a source that switched estimators would fail
rather than quietly relabel.

## Coverage

A row the leaderboard is not publishing (`status != "display"`) is recorded as
a `SourceRecordExclusion` — reported, but it does not fail the refresh. A row
that cannot be represented is a `SourceRecordFailure` and the command exits
non-zero.

Published rows must include all five domain results. Trial and pass counts
must be non-negative whole numbers, with positive trial counts divisible by
three. Missing domains or invalid counts fail the row and are included in the
failure report; counts are never rounded to make them fit.
