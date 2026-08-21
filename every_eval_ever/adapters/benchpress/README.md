# BenchPress adapter

Converts the **BenchPress score matrix** (`microsoft/benchpress-score-matrix`) into
Every Eval Ever records.

BenchPress is an **aggregator**: it re-reports model scores scraped from provider
blogs, tech reports, model cards, leaderboards, and third-party aggregators, each
cell carrying its own citation (`reference_url`) and provenance (`source_type`).
It is handled like the `llm_stats` adapter — `source_type=documentation`,
`source_role=aggregator`, and output logs are **split by `evaluator_relationship`**
(derived per score from BenchPress's `source_type`).

- Data source: the public CSV mirror on HuggingFace
  (`data/{models,benchmarks,scores_all}.csv` + `metadata.json`).
- Paper: https://arxiv.org/abs/2606.24020 · Dataset:
  https://huggingface.co/datasets/microsoft/benchpress-score-matrix

## Usage

Live export (fetches the current data from HuggingFace), writing outside the repo:

```bash
uv run python -m every_eval_ever.adapters.benchpress.adapter \
  --output-dir /tmp/eee-benchpress \
  --save-raw-json /tmp/eee-benchpress-raw.json
```

Validate the produced logs with the real CLI:

```bash
uv run python -m every_eval_ever validate '/tmp/eee-benchpress/*/*/*.json'
```

Replay a saved payload without hitting the network:

```bash
uv run python -m every_eval_ever.adapters.benchpress.adapter \
  --input-json /tmp/eee-benchpress-raw.json \
  --output-dir /tmp/eee-benchpress-replay
```

To intentionally prepare a data refresh, use `--output-dir data/benchpress` and
validate before deciding whether to include generated files.

## Updates / versioning

BenchPress is a living dataset; its `metadata.json` is the documented freshness
manifest ("Export counts, source commit, and matrix construction metadata"). This
adapter uses it as the version anchor: `generated_at_utc` becomes
`retrieved_timestamp`, and `source_git_commit` + `generated_at_utc` are recorded
in every record's `source_metadata.additional_details`, so consumers can detect a
new snapshot (the commit/timestamp changes) and re-run.

The export is four separate files, so a run resolves what it was asked for — the
default tip, a branch, a tag — to one commit sha, reads all four at that sha, and
records it as `benchpress_dataset_revision`. `--revision` replays a snapshot; one
that is already a full sha is used as given.

## Notes / mapping

- `retrieved_timestamp` comes from `metadata.generated_at_utc` (override with
  `--retrieved-timestamp`).
- Only the rows BenchPress itself accepts are exported: `audit_status` in
  `{verified, verified_third_party}`. `dropped`, `needs_review` and `flagged`
  rows are outside its own canonical matrix and are reported as exclusions;
  `--include-unaccepted` exports them anyway.
- `evaluator_relationship` is `third_party` for an independent `source_type`
  (leaderboard, aggregator, academic paper). `first_party` takes **both** a
  provider-authored `source_type` **and** a citation whose *domain* carries the
  scored model's provider name (`openai.com`, `cdn.amazon.science`,
  `moonshotai.github.io`). Anything else is `other` — a competitor's score
  tabulated in a provider's own document, and a provider's page on a host that
  does not spell its name (`storage.googleapis.com`, `huggingface.co/Qwen/…`,
  `lf3-static.bytednsdoc.com`), both land there.
- A score outside the range its own benchmark declares (the export mixes scales
  within a benchmark — `mt_bench_101` declares 1–10 and carries values up to
  90.2) is reported as an unconvertible source row, not rescaled by guess.
- Every source row is accounted for individually: a value that will not parse, a
  field the schema rejects, or a second differing result for a benchmark the log
  already carries (`evaluation_result_id` is the join key for instance records) is
  reported against the row it came from rather than ending the run or losing to
  the first. Failures make the run exit non-zero; exclusions are itemized in the
  same report and do not. The report is rewritten on every run, a clean one
  included, and swapped in atomically.
- `model_info.id` = `<provider-slug>/<benchpress-slug>` (e.g. `openai/gpt-oss-120b`);
  the raw slug is kept in `model_info.additional_details.benchpress_model_id`. The
  eval-card-registry resolves these to canonical ids downstream.
- The per-score citation (`reference_url`) is first in `source_data.url`, with the
  benchmark/dataset URL second; `source_data.additional_details.reported_by` records
  the citation host.
- Metric bounds are the metric's TRUE bounds: a declared `range` wins; otherwise
  per-family bounds with `±inf` where genuinely unbounded (elo/rating/index/raw →
  `[-inf, inf]`; dollars/wer → `[0, inf]`; pct/bleu → `[0, 100]`). `MetricConfig`
  bounds are the one place EEE accepts an unbounded value; it is written as the
  JSON *string* `"Infinity"`, so these records publish through the shared
  `save_evaluation_logs` like every other adapter's.
- `eval_library` is the aggregator (`BenchPress`); the per-score harness is kept in
  `evaluation_results[].metric_config.additional_details`.
- The public CSV mirror does not expose per-cell `candidates` (only the count) or
  benchmark `cost` evidence; those are omitted (and noted) until the canonical JSON
  is published.
