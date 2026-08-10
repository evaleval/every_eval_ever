# Adapters

One-off adapter scripts that fetch leaderboard data from external sources and convert it to the Every Eval Ever schema. These are run manually, not via the main CLI.

## Writing a new adapter

Start from the `eee-dataset-conversion` agent skill —
[`.agents/skills/eee-dataset-conversion/SKILL.md`](../../.agents/skills/eee-dataset-conversion/SKILL.md).
It carries the field semantics, the merge-gate checks (`reference/datastore-gate.md`),
runnable templates, and the datastore submission mechanics. `tests/test_skill_conversion.py`
re-validates those templates against the live validator, so they stay current.

## Usage

Each adapter is run with `uv run python -m every_eval_ever.adapters.<name>.adapter`.

## Adapters

| Adapter | Data Source | Description |
|---------|-------------|-------------|
| `arc_agi` | ARC Prize leaderboard JSON | Converts ARC-AGI leaderboard data and merges canonical model aliases. |
| `artificial_analysis` | Artificial Analysis LLM API | Converts Artificial Analysis LLM benchmark, pricing, and performance results into `data/artificial-analysis-llms/`. |
| `vals_ai` | Vals.ai benchmark leaderboards | Scrapes Vals.ai benchmark pages and converts their embedded leaderboard results into `data/vals-ai/`. |
| `bfcl` | BFCL leaderboard CSV | Converts BFCL leaderboard data with per-metric evaluation names and bounded continuous scores. |
| `sciarena` | SciArena leaderboard API | Converts SciArena leaderboard results. |
| `global_mmlu_lite` | Kaggle API | Fetches Global MMLU Lite leaderboard results from Kaggle. |
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
| `lexam` | LEXam project website | Converts the LEXam legal-reasoning leaderboard (open-question judge scores + 4-choice MCQ accuracy) into `data/lexam/`. |
| `vectara_hallucination_leaderboard` | HuggingFace (`vectara/results`) | Converts the Vectara Hallucination Leaderboard result files, pinned to a source commit, into `data/vectara-hallucination-leaderboard/`. Emits 4 aggregate metrics plus per-category and per-text-complexity breakdowns (40 scores per model). |

### Mercor Evaluation Exports

Set the API key in the environment and run the adapter:

```bash
export MERCOR_EVAL_API_EVALEVAL_KEY="<your-key>"
uv run python -m every_eval_ever.adapters.mercor_eval.adapter
```

For a credential-free offline smoke run:

```bash
uv run python -m every_eval_ever.adapters.mercor_eval.adapter \
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

### LEXam

```bash
uv run python -m every_eval_ever.adapters.lexam.adapter --output-dir data
```

One record per model, with one result per published leaderboard column:

| Metric | Evaluation | Scale | Scope |
|---|---|---|---|
| Open Question Judge Score | `lexam.open_question` | `[0,100]` | test split, n=2,541, scored by a pointwise-minimum ensemble of three judges |
| Multiple-Choice Accuracy | `lexam.mcq_4_choices` | `[0,1]` | n=1,655; the 4-choice config only, not the 8/16/32-choice ones |

The site prints both columns as percentages. Each is emitted on the scale of
its registry metric, with the published percentage kept in
`score_details.details`.

Model ids, metric ids, bounds and direction come from the eval-card-registry
through `registry_snapshot.json`, which vendors the entities this adapter emits
and is pinned to the registry revision they came from. The tests fail if an
emitted value drifts from the pin, so regenerate it after a registry change:

```bash
uv run python -m every_eval_ever.adapters.lexam.refresh_registry_snapshot \
    --registry /path/to/eval-card-registry
```

Add `--check` to test the pin without writing: it exits non-zero and names both
revisions. Metric `review_status` is read from the snapshot, so a metric
promoted upstream needs a refresh rather than a code change.

Inference settings, serving and standard errors are not on the leaderboard;
they come from the paper (arXiv:2505.12864v7 §3.3, appendix F, Tables 1 and 10)
and from LEXam's own runner, `litellm_eval.py`, which names the served model for
15 of the 36 rows. Each record cites the source it used, and a standard error is
attached only while the scraped score still equals the published one.

Submission follows the datastore mechanics in the conversion skill. One
adapter-specific caveat: record filenames are fresh uuids per run, so a second
`upload_folder` onto an open submission PR adds another copy of every model.
Update a submission by deleting `data/lexam/` and adding the new records in a
single `create_commit`.

## Notes

- These are one-off scripts, not integrated into the main CLI.
- They require network access to fetch live leaderboard data.
- Some adapters (e.g. `rewardbench`, `helm`) may take several minutes to complete due to the number of models.
- Run `uv run python -m every_eval_ever.adapters.<name>.adapter --help` for adapter-specific options.
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
uv run python -m every_eval_ever.adapters.vals_ai.adapter \
  --output-dir /tmp/eee-vals-ai/data/vals-ai
```

To intentionally prepare a data refresh, use `--output-dir data/vals-ai` and
validate the result before deciding whether to include generated files.

For smaller smoke runs, fetch one benchmark:

```bash
uv run python -m every_eval_ever.adapters.vals_ai.adapter \
  --benchmark finance_agent \
  --output-dir /tmp/eee-vals-ai-smoke/data/vals-ai \
  --save-raw-json /tmp/eee-vals-ai-raw.json
```

Replay a saved normalized payload without hitting the network:

```bash
uv run python -m every_eval_ever.adapters.vals_ai.adapter \
  --input-json /tmp/eee-vals-ai-raw.json \
  --output-dir /tmp/eee-vals-ai-replay/data/vals-ai
```

Validate generated records with:

```bash
uv run python -m every_eval_ever validate \
  '/tmp/eee-vals-ai-smoke/data/vals-ai/*/*/*.json*'
```

### Vectara Hallucination Leaderboard

The adapter enumerates every per-model result file in `vectara/results` at the
pinned `SOURCE_COMMIT` and emits one record per model. Run a live export
outside the repository:

```bash
uv run python -m every_eval_ever.adapters.vectara_hallucination_leaderboard.adapter \
  --output-dir /tmp/eee-vectara/data/vectara-hallucination-leaderboard \
  --save-raw-json /tmp/eee-vectara-raw.json
```

Replay the saved snapshot without hitting the network:

```bash
uv run python -m every_eval_ever.adapters.vectara_hallucination_leaderboard.adapter \
  --input-json /tmp/eee-vectara-raw.json \
  --output-dir /tmp/eee-vectara-replay/data/vectara-hallucination-leaderboard
```

Bump `SOURCE_COMMIT` to pick up a newer leaderboard run. The evaluated corpus
is private, so the log records the public result file as provenance rather than
a redistributable dataset. The pinned files record no serving platform, so
`deployment_type` stays `unknown`; `model_availability` is derived from the
source's `accessibility` annotation.

Provenance that is constant for a run — the source file, commit, resolve URL,
scoring model and temperature policy — lives once in `source_metadata`. Each of
the 40 results carries only what varies, because repeating the constants on
every result doubled the size of each record. That invariant is pinned by
`test_constant_provenance_is_not_repeated_per_result`.
