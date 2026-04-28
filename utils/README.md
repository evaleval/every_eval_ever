# Adapters

One-off adapter scripts that fetch leaderboard data from external sources and convert it to the Every Eval Ever schema. These are run manually, not via the main CLI.

## Usage

Each adapter is run with `uv run python -m utils.<name>.adapter`.

## Adapters

| Adapter | Data Source | Description |
|---------|-------------|-------------|
| `arc_agi` | ARC Prize leaderboard JSON | Converts ARC-AGI leaderboard data and merges canonical model aliases. |
| `artificial_analysis` | Artificial Analysis LLM API | Converts Artificial Analysis LLM benchmark, pricing, and performance results into `data/artificial-analysis-llms/`. |
| `bfcl` | BFCL leaderboard CSV | Converts BFCL leaderboard data with per-metric evaluation names and bounded continuous scores. |
| `sciarena` | SciArena leaderboard API | Converts SciArena leaderboard results. |
| `global-mmlu-lite` | Kaggle API | Fetches Global MMLU Lite leaderboard results from Kaggle. |
| `hfopenllm_v2` | HuggingFace Spaces API | Fetches the Open LLM Leaderboard v2 (4576+ models). |
| `helm` | HELM leaderboard | Converts HELM leaderboard data. Supports `--leaderboard_name` for Capabilities/Lite/Classic/Instruct/MMLU. |
| `llm_stats` | LLM Stats API | Converts LLM Stats model, benchmark, and score API data into `data/llm-stats/`. |
| `openeval` | HuggingFace | Converts OpenEval response scores from `human-centered-eval/OpenEval` into `data/openeval/`; pass `--include-instances` to also write `*_samples.jsonl` sidecars. |
| `rewardbench` | HuggingFace | Fetches RewardBench v1 (CSV) and RewardBench v2 (JSON) leaderboard data. |
| `terminal_bench_2` | tbench.ai | Fetches Terminal-Bench 2.0 agentic coding benchmark results. |

## Notes

- These are one-off scripts, not integrated into the main CLI.
- They require network access to fetch live leaderboard data.
- Some adapters (e.g. `rewardbench`, `helm`) may take several minutes to complete due to the number of models.
- Run `uv run python -m utils.<name>.adapter --help` for adapter-specific options.
- The script for livecodebenchpro is out-dated and will be updated at a later date.
