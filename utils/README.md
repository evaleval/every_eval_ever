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
| `hfopenllm_v2` | HuggingFace Spaces API | Fetches the Open LLM Leaderboard v2 (4576+ models). |
| `helm` | HELM leaderboard | Converts HELM leaderboard data. Supports `--leaderboard_name` for Capabilities/Lite/Classic/Instruct/MMLU. |
| `llm_stats` | LLM Stats API | Converts LLM Stats model, benchmark, and score API data into `data/llm-stats/`. |
| `mercor_eval` | Mercor Evaluation Exports API | Fetches authenticated Mercor benchmark leaderboards and writes aggregate EEE records. |
| `mt_bench` | LMSYS / FastChat | Converts MT-Bench GPT-4 single-answer judgments into `data/mt-bench/`. Emits overall, turn-1, and turn-2 means per model. |
| `openeval` | HuggingFace | Converts OpenEval response scores from `human-centered-eval/OpenEval` into `data/openeval/`; pass `--include-instances` to also write `*_samples.jsonl` sidecars. |
| `rewardbench` | HuggingFace | Fetches RewardBench v1 (CSV) and RewardBench v2 (JSON) leaderboard data. |
| `terminal_bench_2` | tbench.ai | Fetches Terminal-Bench 2.0 agentic coding benchmark results. |
| `hle` | Scale SEAL leaderboard | Converts the Scale SEAL Humanity's Last Exam leaderboard into `data/hle/`. Emits per-model accuracy (with 95% CI) and calibration error. |
| `mmlu_pro` | TIGER-Lab leaderboard CSV | Converts the MMLU-Pro leaderboard (`TIGER-Lab/mmlu_pro_leaderboard_submission`) into `data/mmlu-pro/`. Emits per-model overall + 14 per-subject accuracies. |

### Mercor Evaluation Exports

Set the API key in the environment and run the adapter:

```bash
export MERCOR_EVAL_API_EVALEVAL_KEY="<your-key>"
uv run python -m utils.mercor_eval.adapter
```

For a credential-free offline smoke run:

```bash
uv run python -m utils.mercor_eval.adapter \
  --input-json tests/data/mercor_eval/api_payload.json \
  --output-dir /tmp/mercor-eval-offline
```

The adapter exports aggregate leaderboard metrics only. Mercor's criterion
results do not include the task input, model output, messages, or answer
attribution required by the EEE instance-level schema.
Records are generated under benchmark-specific datastore directories, for
example `data/apex-agents/<developer>/<model>/<uuid>.json`. Generated records
are intended for the Hugging Face datastore submission, not the GitHub adapter
PR.

## Notes

- These are one-off scripts, not integrated into the main CLI.
- They require network access to fetch live leaderboard data.
- Some adapters (e.g. `rewardbench`, `helm`) may take several minutes to complete due to the number of models.
- Run `uv run python -m utils.<name>.adapter --help` for adapter-specific options.
- Generated adapter outputs under `data/<source>/` and saved raw payloads are
  generated artifacts. Prefer temporary output paths for smoke runs unless a
  data refresh is intentionally part of the change.

### Legacy integrations

`arc_agi`, `livecodebenchpro`, and `mercor_eval` are retained for historical
and offline use, but their upstream sources are no longer usable for an active
refresh (`mercor_eval` currently returns an empty response). They are excluded
from active-adapter migration and compliance requirements. Deterministic
offline tests for their existing behavior may remain in the test suite.

### Partial conversions and provenance

An adapter may encounter a source row or metric that cannot be represented as
a valid EEE record—for example, a missing model identity or a non-numeric
score. It still writes every valid record. It also writes a strict JSON
provenance report under `adapter_reports/`, outside `data/`, with the source
reference, raw source fragment when available, and reason for each omission.
The command then exits non-zero so automation can distinguish a complete
refresh from a partial one.

Intentional non-evaluation rows, such as a published random baseline, are
recorded as exclusions in the same report but do not make the command fail.
The report is not an `EvaluationLog` and must not be passed to the validator.

### Vals.ai

Run a live smoke export from the repository root, writing generated output
outside the repo:

```bash
uv run python -m utils.vals_ai.adapter \
  --output-dir /tmp/eee-vals-ai/data/vals-ai
```

To intentionally prepare a data refresh, use `--output-dir data/vals-ai` and
validate the result before deciding whether to include generated files.

For smaller smoke runs, fetch one benchmark:

```bash
uv run python -m utils.vals_ai.adapter \
  --benchmark finance_agent \
  --output-dir /tmp/eee-vals-ai-smoke/data/vals-ai \
  --save-raw-json /tmp/eee-vals-ai-raw.json
```

Replay a saved normalized payload without hitting the network:

```bash
uv run python -m utils.vals_ai.adapter \
  --input-json /tmp/eee-vals-ai-raw.json \
  --output-dir /tmp/eee-vals-ai-replay/data/vals-ai
```

Validate generated records with:

```bash
uv run python -m every_eval_ever validate \
  '/tmp/eee-vals-ai-smoke/data/vals-ai/*/*/*.json*'
```
