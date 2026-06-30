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
uv run python -m utils.benchpress.adapter \
  --output-dir /tmp/eee-benchpress \
  --save-raw-json /tmp/eee-benchpress-raw.json
```

Validate the produced logs with the real CLI:

```bash
uv run python -m every_eval_ever validate /tmp/eee-benchpress
```

Replay a saved payload without hitting the network:

```bash
uv run python -m utils.benchpress.adapter \
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

## Notes / mapping

- `retrieved_timestamp` comes from `metadata.generated_at_utc` (override with
  `--retrieved-timestamp`).
- `model_info.id` = `<provider-slug>/<benchpress-slug>` (e.g. `openai/gpt-oss-120b`);
  the raw slug is kept in `model_info.additional_details.benchpress_model_id`. The
  eval-card-registry resolves these to canonical ids downstream.
- The per-score citation (`reference_url`) is first in `source_data.url`, with the
  benchmark/dataset URL second; `source_data.additional_details.reported_by` records
  the citation host.
- Metric bounds are the metric's TRUE bounds: a declared `range` wins; otherwise
  per-family bounds with `±inf` where genuinely unbounded (elo/rating/index/raw →
  `[-inf, inf]`; dollars/wer → `[0, inf]`; pct/bleu → `[0, 100]`). EEE has no
  unbounded `score_type`, so the adapter writes `inf` as the JSON `Infinity` token
  (`json.dumps(allow_nan=True)`); EEE's loader (`json.loads` + pydantic) reads it
  back as `float('inf')`. (Note: pydantic's `model_dump_json` would null `inf`, so
  the adapter serializes the records itself — see `write_log`.)
- `eval_library` is the aggregator (`BenchPress`); the per-score harness is kept in
  `evaluation_results[].metric_config.additional_details`.
- The public CSV mirror does not expose per-cell `candidates` (only the count) or
  benchmark `cost` evidence; those are omitted (and noted) until the canonical JSON
  is published.
