# Every Eval Ever

> [EvalEval Coalition](https://evalevalai.com) — "We are a researcher community developing scientifically grounded research outputs and robust deployment infrastructure for broader impact evaluations."

**Every Eval Ever** is a shared schema and crowdsourced eval database. It defines a standardized metadata format for storing AI evaluation results — from leaderboard scrapes and research papers to local evaluation runs — so that results from different frameworks can be compared, reproduced, and reused. The three components that make it work:

- 📋 **A metadata schema** ([`eval.schema.json`](eval.schema.json)) that defines the information needed for meaningful comparison of evaluation results, including [instance-level data](instance_level_eval.schema.json)
- 🔧 **Validation** that checks data against the schema before it enters the repository
- 🔌 **Converters** for [Inspect AI](eval_converters/inspect/), [HELM](eval_converters/helm/), and [lm-eval-harness](eval_converters/lm_eval/), so you can transform your existing evaluation logs into the standard format

### Terminology

| Term | Our Definition | Example |
|---|---|---|
| **Single Benchmark** | Standardized eval using one dataset to test a single capability, producing one score | MMLU — ~15k multiple-choice QA across 57 subjects |
| **Composite Benchmark** | A collection of simple benchmarks aggregated into one overall score, testing multiple capabilities at once | BIG-Bench bundles >200 tasks with a single aggregate score |
| **Metric** | Any numerical value used to score performance on a benchmark (accuracy, F1, precision, recall, …) | A model scores 92% accuracy on MMLU |

## 🚀 Contributor Guide
Leaderboard/evaluation data is split-up into files by individual model, and data for each model is stored using [`eval.schema.json`](eval.schema.json). The repository is structured into folders as `data/{benchmark_name}/{developer_name}/{model_name}/`.

### TL;DR How to successfully submit

1. Data must conform to [`eval.schema.json`](eval.schema.json) (current version: `0.2.0`)
2. Validation runs automatically on every PR via [`validate_data.py`](utils/validate_data.py)
3. An EvalEval member will review and merge your submission

### UUID Naming Convention

Each JSON file is named with a **UUID (Universally Unique Identifier)** in the format `{uuid}.json`. The UUID is automatically generated (using standard UUID v4) when creating a new evaluation result file. This ensures that:
- **Multiple evaluations** of the same model can exist without conflicts (each gets a unique UUID)
- **Different timestamps** are stored as separate files with different UUIDs (not as separate folders)
- A model may have multiple result files, with each file representing different iterations or runs of the leaderboard/evaluation
- UUID's can be generated using Python's `uuid.uuid4()` function.

**Example**: The model `openai/gpt-4o-2024-11-20` might have multiple files like:
- `e70acf51-30ef-4c20-b7cc-51704d114d70.json` (evaluation run #1)
- `a1b2c3d4-5678-90ab-cdef-1234567890ab.json` (evaluation run #2)

Note: Each file can contain multiple individual results related to one model. See [examples in /data](data/).

### How to add new eval:

1. Add a new folder under [`data/`](data/) with a codename for your eval.
2. For each model, use the HuggingFace (`developer_name/model_name`) naming convention to create a 2-tier folder structure.
3. Add a JSON file with results for each model and name it `{uuid}.json`.
4. [Optional] Include a [`utils/`](utils/) folder in your benchmark name folder with any scripts used to generate the data (see e.g. [`utils/global-mmlu-lite/adapter.py`](utils/global-mmlu-lite/adapter.py)).
5. [Validate] Validation runs automatically via [`validate-data.yml`](.github/workflows/validate-data.yml) using [`validate_data.py`](utils/validate_data.py) to check JSON files against the schema before merging.
6. [Submit] Two ways to submit your evaluation data:
   - **Option A: Drag & drop via Hugging Face** — Go to [evaleval/EEE_datastore](https://huggingface.co/datasets/evaleval/EEE_datastore) → click "Files and versions" → "Contribute" → "Upload files" → drag and drop your data → select "Open as a pull request to the main branch". See [step-by-step screenshots](https://docs.google.com/document/d/1dxTQF8ncGCzaAOIj0RX7E9Hg4THmUBzezDOYUp_XdCY/edit?usp=sharing).
   - **Option B: Clone & PR** — Clone the [HuggingFace repository](https://huggingface.co/datasets/evaleval/EEE_datastore), add your data under `data/`, and open a pull request

### Schema Instructions

1. **`model_info`**: Use HuggingFace formatting (`developer_name/model_name`). If a model does not come from HuggingFace, use the exact API reference. Check [examples in /data/livecodebenchpro](data/livecodebenchpro/). Notably, some do have a **date included in the model name**, but others **do not**. For example:
- OpenAI: `gpt-4o-2024-11-20`, `gpt-5-2025-08-07`, `o3-2025-04-16`
- Anthropic: `claude-3-7-sonnet-20250219`, `claude-3-sonnet-20240229`
- Google: `gemini-2.5-pro`, `gemini-2.5-flash`
- xAI (Grok): `grok-2-2024-08-13`, `grok-3-2025-01-15`

2. **`evaluation_id`**: Use `{benchmark_name/model_id/retrieved_timestamp}` format (e.g. `livecodebenchpro/qwen3-235b-a22b-thinking-2507/1760492095.8105888`).

3. **`inference_platform`** vs **`inference_engine`**: Where possible specify where the evaluation was run using one of these two fields.
- `inference_platform`: Use this field when the evaluation was run through a remote API (e.g., `openai`, `huggingface`, `openrouter`, `anthropic`, `xai`).
- `inference_engine`: Use this field when the evaluation was run locally. This is now an object with `name` and `version` (e.g. `{"name": "vllm", "version": "0.6.0"}`).

4. The `source_type` on `source_metadata` has two options: `documentation` and `evaluation_run`. Use `documentation` when results are scraped from a leaderboard or paper. Use `evaluation_run` when the evaluation was run locally (e.g. via an eval converter).

5. **`source_data`** is specified per evaluation result (inside `evaluation_results`), with three variants:
- `source_type: "url"` — link to a web source (e.g. leaderboard API)
- `source_type: "hf_dataset"` — reference to a HuggingFace dataset (e.g. `{"hf_repo": "google/IFEval"}`)
- `source_type: "other"` — for private or proprietary datasets

6. The schema is designed to accommodate both numeric and level-based (e.g. Low, Medium, High) metrics. For level-based metrics, the actual 'value' should be converted to an integer (e.g. Low = 1, Medium = 2, High = 3), and the `level_names` property should be used to specify the mapping of levels to integers.

7. **Timestamps**: The schema has three timestamp fields — use them as follows:
- `retrieved_timestamp` (required) — when this record was created, in Unix epoch format (e.g. `1760492095.8105888`)
- `evaluation_timestamp` (top-level, optional) — when the evaluation was run
- `evaluation_results[].evaluation_timestamp` (per-result, optional) — when a specific evaluation result was produced, if different results were run at different times

8. Additional details can be provided in several places in the schema. They are not required, but can be useful for detailed analysis.
- `model_info.additional_details`: Use this field to provide any additional information about the model itself (e.g. number of parameters)
- `evaluation_results.generation_config.generation_args`: Specify additional arguments used to generate outputs from the model
- `evaluation_results.generation_config.additional_details`: Use this field to provide any additional information about the evaluation process that is not captured elsewhere


### Instance-Level Data

For evaluations that include per-sample results, the individual results should be stored in a companion `{uuid}.jsonl` file in the same folder (one JSONL per JSON, sharing the same UUID). The aggregate JSON file refers to its JSONL via the `detailed_evaluation_results` field. The instance-level schema ([`instance_level_eval.schema.json`](instance_level_eval.schema.json)) supports three interaction types:

- **`single_turn`**: Standard QA, MCQ, classification — uses `output` object
- **`multi_turn`**: Conversational evaluations with multiple exchanges — uses `interactions` array
- **`agentic`**: Tool-using evaluations with function calls and sandbox execution — uses `interactions` array with `tool_calls`

Each instance captures: `input` (raw question + reference answer), `answer_attribution` (how the answer was extracted), `evaluation` (score, is_correct), and optional `token_usage` and `performance` metrics. Instance-level JSONL files are produced automatically by the [eval converters](eval_converters/README.md).

Example `single_turn` instance:

```json
{
  "schema_version": "instance_level_eval_0.2.0",
  "evaluation_id": "math_eval/meta-llama/Llama-2-7b-chat/1706000000",
  "model_id": "meta-llama/Llama-2-7b-chat",
  "evaluation_name": "math_eval",
  "sample_id": 4,
  "interaction_type": "single_turn",
  "input": { "raw": "If 2^10 = 4^x, what is the value of x?", "reference": "5" },
  "output": { "raw": "Rewrite 4 as 2^2, so 4^x = 2^(2x). Since 2^10 = 2^(2x), x = 5." },
  "answer_attribution": [{ "source": "output.raw", "extracted_value": "5" }],
  "evaluation": { "score": 1.0, "is_correct": true }
}
```

### Agentic Evaluations

For agentic evaluations (e.g., SWE-Bench, GAIA), the aggregate schema captures configuration under `generation_config.generation_args`:

```json
{
  "agentic_eval_config": {
    "available_tools": [
      {"name": "bash", "description": "Execute shell commands"},
      {"name": "edit_file", "description": "Edit files in the repository"}
    ]
  },
  "eval_limits": {"message_limit": 30, "token_limit": 100000},
  "sandbox": {"type": "docker", "config": "compose.yaml"}
}
```

At the instance level, agentic evaluations use `interaction_type: "agentic"` with full tool call traces recorded in the `interactions` array. See the [Inspect AI test fixture](tests/data/inspect/) for a GAIA example with docker sandbox and tool usage.

## ✅ Data Validation

This repository has a pre-commit that will validate that JSON files conform to the JSON schema. The pre-commit requires using [uv](https://docs.astral.sh/uv/) for dependency management.

To run the pre-commit on git staged files only:

```sh
uv run pre-commit run
```

To run the pre-commit on all files:

```sh
uv run pre-commit run --all-files
```

To run the pre-commit on specific files:

```sh
uv run pre-commit run --files a.json b.json c.json
```

To install the pre-commit so that it will run before `git commit` (optional):

```sh
uv run pre-commit install
```

## 🗂️ Repository Structure

```
data/
└── {benchmark_name}/
    └── {developer_name}/
        └── {model_name}/
            ├── {uuid}.json          # aggregate results
            └── {uuid}.jsonl         # instance-level results (optional)
```

Example evaluations included in the schema v0.2 release:

| Evaluation | Data |
|---|---|
| Global MMLU Lite | [`data/global-mmlu-lite/`](data/global-mmlu-lite/) |
| HELM Capabilities v1.15 | [`data/helm_capabilities/`](data/helm_capabilities/) |
| HELM Classic | [`data/helm_classic/`](data/helm_classic/) |
| HELM Instruct | [`data/helm_instruct/`](data/helm_instruct/) |
| HELM Lite | [`data/helm_lite/`](data/helm_lite/) |
| HELM MMLU | [`data/helm_mmlu/`](data/helm_mmlu/) |
| HF Open LLM Leaderboard v2 | [`data/hfopenllm_v2/`](data/hfopenllm_v2/) |
| LiveCodeBench Pro | [`data/livecodebenchpro/`](data/livecodebenchpro/) |
| RewardBench | [`data/reward-bench/`](data/reward-bench/) |

Schemas: [`eval.schema.json`](eval.schema.json) (aggregate) · [`instance_level_eval.schema.json`](instance_level_eval.schema.json) (per-sample JSONL)

Each evaluation has its own directory under [`data/`](data/). Within each evaluation, models are organized by developer and model name. Instance-level data is stored in optional `{uuid}.jsonl` files alongside aggregate `{uuid}.json` results.

## 📋 The Schema in Practice

For a detailed walk-through, see the [blogpost](https://evalevalai.com/infrastructure/2026/02/15/everyevalever-launch/).

Each result file captures not just scores but the context needed to interpret and reuse them. Here's how it works, piece by piece:

**Where did the evaluation come from?** Source metadata tracks who ran it, where the data was published, and the relationship to the model developer:

```json
"source_metadata": {
  "source_name": "Live Code Bench Pro",
  "source_type": "documentation",
  "source_organization_name": "LiveCodeBench",
  "evaluator_relationship": "third_party"
}
```

**Generation settings matter.** Changing temperature or the number of samples alone can shift scores by several points — yet they're routinely absent from leaderboards. We capture them explicitly:

```json
"generation_config": {
  "generation_args": {
    "temperature": 0.2,
    "top_p": 0.95,
    "max_tokens": 2048
  }
}
```

**The score itself.** A score of 0.31 on a coding benchmark (pass@1) means higher is better. The same 0.31 on RealToxicityPrompts means lower is better. The schema standardizes this interpretation:

```json
"evaluation_results": [{
  "evaluation_name": "code_generation",
  "metric_config": {
    "evaluation_description": "pass@1 on code generation tasks",
    "lower_is_better": false,
    "score_type": "continuous",
    "min_score": 0,
    "max_score": 1
  },
  "score_details": {
    "score": 0.31
  }
}]
```

The schema also supports **level-based metrics** (e.g. Low/Medium/High) and **uncertainty** reporting (confidence intervals, standard errors). See [`eval.schema.json`](eval.schema.json) for the full specification.

## 🔧 Auto-generation of Pydantic Classes for Schema

Run following bash commands to generate pydantic classes for `eval.schema.json` and `instance_level_eval.schema.json` (to easier use in data converter scripts):

```bash
uv run datamodel-codegen --input eval.schema.json --output eval_types.py --class-name EvaluationLog --output-model-type pydantic_v2.BaseModel --input-file-type jsonschema --formatters ruff-format ruff-check
uv run datamodel-codegen --input instance_level_eval.schema.json --output instance_level_types.py --class-name InstanceLevelEvaluationLog --output-model-type pydantic_v2.BaseModel --input-file-type jsonschema --formatters ruff-format ruff-check
```

## 🔌 Eval Converters

We have prepared converters to make adapting to our schema as easy as possible. At the moment, we support converting local evaluation harness logs from `Inspect AI`, `HELM` and `lm-evaluation-harness` into our unified schema. Each converter produces aggregate JSON and optionally instance-level JSONL output.

| Framework | Command | Instance-Level JSONL |
|---|---|---|
| [Inspect AI](eval_converters/inspect/) | `uv run python3 -m eval_converters.inspect --log_path <path>` | Yes, if samples in log |
| [HELM](eval_converters/helm/) | `uv run python3 -m eval_converters.helm --log_path <path>` | Always |
| [lm-evaluation-harness](eval_converters/lm_eval/) | `uv run python -m eval_converters.lm_eval --log_path <path>` | With `--include_samples` |

For full CLI usage and required input files, see the [Eval Converters README](eval_converters/README.md).

## 🏆 ACL 2026 Shared Task

We are running a [Shared Task](https://evalevalai.com/events/shared-task-every-eval-ever/) at **ACL 2026 in San Diego** (July 7, 2026). The task invites participants to contribute to a unifying database of eval results:

- **Track 1: Public Eval Data Parsing** — Parse leaderboards (Chatbot Arena, Open LLM Leaderboard, AlpacaEval, etc.) and academic papers into [our schema](eval.schema.json) and contribute to a unifying database of eval results!
- **Track 2: Proprietary Evaluation Data** — Convert proprietary evaluation datasets into [our schema](eval.schema.json) and contribute to a unifying database of eval results!

| Milestone | Date |
|---|---|
| Submission deadline | May 1, 2026 |
| Results announced | June 1, 2026 |
| Workshop at ACL 2026 | July 7, 2026 |

Qualifying contributors will be invited as co-authors on the shared task paper.

## 📎 Citation

```bibtex
@misc{everyevalever2026schema,
  title   = {Every Eval Ever Metadata Schema v0.2},
  author  = {EvalEval Coalition},
  year    = {2026},
  month   = {February},
  url     = {https://github.com/evaleval/every_eval_ever},
  note    = {Schema Release}
}
```