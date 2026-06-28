# Adapters

One-off adapter scripts that fetch leaderboard data from external sources and convert it to the Every Eval Ever schema. These are run manually, not via the main CLI.

## Usage

Each adapter is run with `uv run python -m utils.<name>.adapter`.

## Adapters

| Adapter | Data Source | Description |
|---------|-------------|-------------|
| `arc_agi` | ARC Prize leaderboard JSON | Converts ARC-AGI leaderboard data and merges canonical model aliases. |
| `artificial_analysis` | Artificial Analysis LLM API | Converts Artificial Analysis LLM benchmark, pricing, and performance results into `data/artificial-analysis-llms/`. |
| `vals_ai` | Vals.ai benchmark leaderboards | Scrapes Vals.ai benchmark pages and converts their embedded leaderboard results into `data/vals-ai/`. |
| `bfcl` | BFCL leaderboard CSV | Converts BFCL leaderboard data with per-metric evaluation names and bounded continuous scores. |
| `sciarena` | SciArena leaderboard API | Converts SciArena leaderboard results. |
| `global-mmlu-lite` | Kaggle API | Fetches Global MMLU Lite leaderboard results from Kaggle. |
| `kaggle` | Kaggle Benchmarks API | Generalized Kaggle Community Benchmarks adapter. Converts any benchmark via `--benchmark owner/slug`, or discovers and converts all published benchmarks via `--all` (uses the `ListBenchmarks` RPC). Handles numeric and boolean task results. |
| `hfopenllm_v2` | HuggingFace Spaces API | Fetches the Open LLM Leaderboard v2 (4576+ models). |
| `helm` | HELM leaderboard | Converts HELM leaderboard data. Supports `--leaderboard_name` for Capabilities/Lite/Classic/Instruct/MMLU. |
| `llm_stats` | LLM Stats API | Converts LLM Stats model, benchmark, and score API data into `data/llm-stats/`. |
| `mt_bench` | LMSYS / FastChat | Converts MT-Bench GPT-4 single-answer judgments into `data/mt-bench/`. Emits overall, turn-1, and turn-2 means per model. |
| `openeval` | HuggingFace | Converts OpenEval response scores from `human-centered-eval/OpenEval` into `data/openeval/`; pass `--include-instances` to also write `*_samples.jsonl` sidecars. |
| `rewardbench` | HuggingFace | Fetches RewardBench v1 (CSV) and RewardBench v2 (JSON) leaderboard data. |
| `terminal_bench_2` | tbench.ai | Fetches Terminal-Bench 2.0 agentic coding benchmark results. |
| `hle` | Scale SEAL leaderboard | Converts the Scale SEAL Humanity's Last Exam leaderboard into `data/hle/`. Emits per-model accuracy (with 95% CI) and calibration error. |
| `mmlu_pro` | TIGER-Lab leaderboard CSV | Converts the MMLU-Pro leaderboard (`TIGER-Lab/mmlu_pro_leaderboard_submission`) into `data/mmlu-pro/`. Emits per-model overall + 14 per-subject accuracies. |

## Notes

- These are one-off scripts, not integrated into the main CLI.
- They require network access to fetch live leaderboard data.
- Some adapters (e.g. `rewardbench`, `helm`) may take several minutes to complete due to the number of models.
- Run `uv run python -m utils.<name>.adapter --help` for adapter-specific options.
- The script for livecodebenchpro is out-dated and will be updated at a later date.
- Generated adapter outputs under `data/<source>/` and saved raw payloads are
  generated artifacts. Prefer temporary output paths for smoke runs unless a
  data refresh is intentionally part of the change.

### Kaggle Benchmarks

Convert one or more named benchmarks (smoke run, output outside the repo):

```bash
uv run python -m utils.kaggle.adapter \
  --benchmark cohere-labs/global-mmlu-lite \
  --output-dir /tmp/eee-kaggle
```

Discover and convert all published benchmarks via the `ListBenchmarks` RPC
(slow — 1000+ benchmarks; use `--limit` to cap during testing):

```bash
uv run python -m utils.kaggle.adapter --all --limit 10 --output-dir /tmp/eee-kaggle
```

Notes:
- The `--all` discovery path calls an unauthenticated Kaggle internal RPC
  (`benchmarks.BenchmarkService/ListBenchmarks`) that requires the anonymous
  XSRF cookie/header handshake; the adapter handles this automatically.
- Numeric results in `[0, 1]` are emitted as bounded `continuous` metrics and
  boolean results as `binary`; numeric results outside `[0, 1]` have an unknown
  scale (Kaggle does not expose metric bounds), so they are left untyped rather
  than given fabricated `[0, 1]` bounds.
- Each result carries its Kaggle `evaluationDate` as `evaluation_timestamp` when
  present (~70% of results).
- The `--all` path additionally enriches each record from the benchmark's
  scoring config (only available via `ListBenchmarks`): `lower_is_better` from
  `sortOrder`, `metric_kind` from `aggregationType`, `metric_unit` from
  `displayType`, and `benchmark_id`/`aggregation_type`/`display_type` in
  `source_metadata.additional_details`. The targeted `--benchmark` path does not
  have this metadata, so those fields fall back to defaults there.
- This supersedes the single-purpose `global-mmlu-lite` adapter, which is kept
  for backwards compatibility.

### Vals.ai

Run a live smoke export from the repository root, writing generated output
outside the repo:

```bash
uv run python -m utils.vals_ai.adapter --output-dir /tmp/eee-vals-ai
```

To intentionally prepare a data refresh, use `--output-dir data/vals-ai` and
validate the result before deciding whether to include generated files.

For smaller smoke runs, fetch one benchmark:

```bash
uv run python -m utils.vals_ai.adapter \
  --benchmark finance_agent \
  --output-dir /tmp/eee-vals-ai-smoke \
  --save-raw-json /tmp/eee-vals-ai-raw.json
```

Replay a saved normalized payload without hitting the network:

```bash
uv run python -m utils.vals_ai.adapter \
  --input-json /tmp/eee-vals-ai-raw.json \
  --output-dir /tmp/eee-vals-ai-replay
```

Validate generated records with:

```bash
uv run python -m every_eval_ever validate /tmp/eee-vals-ai-smoke
```
