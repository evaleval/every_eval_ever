# Every Eval Ever

[![PyPI version](https://img.shields.io/pypi/v/every-eval-ever.svg)](https://pypi.org/project/every-eval-ever/)
[![PyPI downloads](https://img.shields.io/pypi/dm/every-eval-ever.svg)](https://pypistats.org/packages/every-eval-ever)

> [EvalEval Coalition](https://evalevalai.com) — "We are a researcher community developing scientifically grounded research outputs and robust deployment infrastructure for broader impact evaluations."

**Every Eval Ever** is a shared schema and crowdsourced eval database. It defines a standardized metadata format for storing AI evaluation results — from leaderboard scrapes and research papers to local evaluation runs — so that results from different frameworks can be compared, reproduced, and reused. The three components that make it work:

- 📋 **A metadata schema** ([`eval.schema.json`](eval.schema.json)) that defines the information needed for meaningful comparison of evaluation results, including [instance-level data](instance_level_eval.schema.json)
- 🔧 **Validation** that checks data against the schema before it enters the repository
- 🔌 **Converters** for [Inspect AI](every_eval_ever/converters/inspect/), [HELM](every_eval_ever/converters/helm/), and [lm-eval-harness](every_eval_ever/converters/lm_eval/), so you can transform your existing evaluation logs into the standard format

Install the package:

```bash
pip install every-eval-ever
```

Optional converter dependencies:

```bash
pip install 'every-eval-ever[inspect]'
pip install 'every-eval-ever[helm]'
pip install 'every-eval-ever[all]'
```

### Terminology

| Term | Our Definition | Example |
|---|---|---|
| **Single Benchmark** | Standardized eval using one dataset to test a single capability, producing one score | MMLU — ~15k multiple-choice QA across 57 subjects |
| **Composite Benchmark** | A collection of simple benchmarks aggregated into one overall score, testing multiple capabilities at once | BIG-Bench bundles >200 tasks with a single aggregate score |
| **Metric** | Any numerical or categorical value used to score performance on a benchmark (accuracy, F1, precision, recall, …) | A model scores 92% accuracy on MMLU |

## 🚀 Contributor Guide
New data can be contributed to the [Hugging Face Dataset](https://huggingface.co/datasets/evaleval/EEE_datastore) using the following process:

Leaderboard/evaluation data is split-up into files by individual model, and data for each model is stored using [`eval.schema.json`](eval.schema.json). The repository is structured into folders as `data/{benchmark_name}/{developer_name}/{model_name}/`.

### TL;DR How to successfully submit

1. Data must conform to [`eval.schema.json`](eval.schema.json) (current version: `0.2.2`)
2. The validation pipeline will automatically verify the data submitted in the pull request, but can also be manually triggered by typing ```/eee validate changed``` in a comment on the HF PR.
3. An EvalEval member will review and merge your submission

### PR Naming Convention

Use these prefixes in your pull request titles:

- `[Submission]` - New evaluation data
- `[Issue #N]` - Fix for a specific GitHub issue
- `[Feature]` - New functionality not tied to an issue
- `[Docs]` - Documentation changes
- `[ACL Shared Task]` - Shared task submissions (priority review)

### UUID Naming Convention

Each JSON file is named with a **UUID (Universally Unique Identifier)** in the format `{uuid}.json`. The UUID is automatically generated (using standard UUID v4) when creating a new evaluation result file. This ensures that:
- **Multiple evaluations** of the same model can exist without conflicts (each gets a unique UUID)
- **Different timestamps** are stored as separate files with different UUIDs (not as separate folders)
- A model may have multiple result files, with each file representing different iterations or runs of the leaderboard/evaluation
- UUID's can be generated using Python's `uuid.uuid4()` function.

**Example**: The model `openai/gpt-4o-2024-11-20` might have multiple files like:
- `e70acf51-30ef-4c20-b7cc-51704d114d70.json` (evaluation run #1)
- `a1b2c3d4-5678-90ab-cdef-1234567890ab.json` (evaluation run #2)

Note: Each file can contain multiple individual results related to one model. See [examples in the datastore](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data).

### How to add new eval:

1. Add a new folder under [`data/`](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data) on the Hugging Face datastore with a codename for your eval.
2. For each model, use the Hugging Face (`developer_name/model_name`) naming convention to create a 2-tier folder structure.
3. Add a JSON file with results for each model and name it `{uuid}.json`.
4. [Optional] Include a [`utils/`](utils/) folder in your benchmark name folder with any scripts used to generate the data (see e.g. [`utils/global-mmlu-lite/adapter.py`](utils/global-mmlu-lite/adapter.py)).
5. [Submit] Two ways to submit your evaluation data:
   - **Option A: Drag & drop via Hugging Face** — Go to [evaleval/EEE_datastore](https://huggingface.co/datasets/evaleval/EEE_datastore) → click "Files and versions" → "Contribute" → "Upload files" → drag and drop your data → select "Open as a pull request to the main branch". See [step-by-step screenshots](https://docs.google.com/document/d/1dxTQF8ncGCzaAOIj0RX7E9Hg4THmUBzezDOYUp_XdCY/edit?usp=sharing).
   - **Option B: Clone & PR** — Clone the [Hugging Face repository](https://huggingface.co/datasets/evaleval/EEE_datastore), add your data under `data/`, and open a pull request

### Schema Instructions

1. **`model_info`**: Use Hugging Face formatting (`developer_name/model_name`). If a model does not come from Hugging Face, use the exact API reference. Check [examples in data/livecodebenchpro](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data/livecodebenchpro). Notably, some do have a **date included in the model name**, but others **do not**. For example:
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
- `source_type: "hf_dataset"` — reference to a Hugging Face dataset (e.g. `{"hf_repo": "google/IFEval"}`)
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

For evaluations that include per-sample results, the individual results should be stored in a companion `{uuid}_samples.jsonl` file in the same folder (one JSONL per JSON, sharing the same UUID). The aggregate JSON file refers to its JSONL via the `detailed_evaluation_results` field. The instance-level schema ([`instance_level_eval.schema.json`](instance_level_eval.schema.json)) supports three interaction types:

- **`single_turn`**: Standard QA, MCQ, classification — uses `output` object
- **`multi_turn`**: Conversational evaluations with multiple exchanges — uses `messages` array
- **`agentic`**: Tool-using evaluations with function calls and sandbox execution — uses `messages` array with `tool_calls`

Each instance captures: `input` (raw question + reference answer), `answer_attribution` (how the answer was extracted), `evaluation` (score, is_correct), and optional `token_usage` and `performance` metrics. Instance-level JSONL files are produced automatically by the [eval converters](every_eval_ever/converters/README.md).

Example `single_turn` instance:

```json
{
  "schema_version": "instance_level_eval_0.2.2",
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

At the instance level, agentic evaluations use `interaction_type: "agentic"` with full tool call traces recorded in the `messages` array. See the [Inspect AI test fixture](tests/data/inspect/) for a GAIA example with docker sandbox and tool usage.

## ✅ Data Validation

Validation uses Pydantic models generated from the JSON schemas. This validates aggregate `.json` files against `EvaluationLog` and instance-level `_samples.jsonl` files line-by-line against `InstanceLevelEvaluationLog`. Requires [uv](https://docs.astral.sh/uv/).

### Validate files with the package CLI

```sh
# Single aggregate file
uv run python -m every_eval_ever validate data/benchmark/dev/model/uuid.json

# Instance-level JSONL
uv run python -m every_eval_ever validate data/benchmark/dev/model/uuid_samples.jsonl

# Entire directory (recurses into subdirectories)
uv run python -m every_eval_ever validate data/benchmark/dev/model/

# Multiple paths
uv run python -m every_eval_ever validate file1.json file2_samples.jsonl data/
```

File type is determined by extension: `.json` validates against `EvaluationLog`, `.jsonl` validates each line against `InstanceLevelEvaluationLog`.

#### Output formats

```sh
# Rich terminal output (default)
uv run python -m every_eval_ever validate data/

# Machine-readable JSON
uv run python -m every_eval_ever validate --format json data/

# GitHub Actions annotations
uv run python -m every_eval_ever validate --format github data/
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--format {rich,json,github}` | `rich` | Output format |
| `--max-errors N` | `50` | Maximum errors reported per JSONL file |

Exit code is `0` if all files pass and `1` if any fail.

## 🗂️ Data Structure

Evaluation data is hosted on the [Hugging Face datastore](https://huggingface.co/datasets/evaleval/EEE_datastore). The folder structure is:

```
data/
└── {benchmark_name}/
    └── {developer_name}/
        └── {model_name}/
            ├── {uuid}.json          # aggregate results
            └── {uuid}_samples.jsonl # instance-level results (optional)
```

Example evaluations included in the schema v0.2 release:

| Evaluation | Data |
|---|---|
| Global MMLU Lite | [`data/global-mmlu-lite/`](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data/global-mmlu-lite) |
| HELM Capabilities v1.15 | [`data/helm_capabilities/`](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data/helm_capabilities) |
| HELM Classic | [`data/helm_classic/`](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data/helm_classic) |
| HELM Instruct | [`data/helm_instruct/`](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data/helm_instruct) |
| HELM Lite | [`data/helm_lite/`](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data/helm_lite) |
| HELM MMLU | [`data/helm_mmlu/`](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data/helm_mmlu) |
| HF Open LLM Leaderboard v2 | [`data/hfopenllm_v2/`](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data/hfopenllm_v2) |
| LiveCodeBench Pro | [`data/livecodebenchpro/`](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data/livecodebenchpro) |
| RewardBench | [`data/reward-bench/`](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data/reward-bench) |

Schemas: [`eval.schema.json`](eval.schema.json) (aggregate) · [`instance_level_eval.schema.json`](instance_level_eval.schema.json) (per-sample JSONL)

Each evaluation has its own directory under [`data/`](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data) on the Hugging Face datastore. Within each evaluation, models are organized by developer and model name. Instance-level data is stored in optional `{uuid}_samples.jsonl` files alongside aggregate `{uuid}.json` results.

## 📋 The Schema in Practice

For a detailed walk-through, see the [blogpost](https://evalevalai.com/infrastructure/2026/02/17/everyevalever-launch/).

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

Run the following commands to generate the package-local Pydantic classes from the canonical package-local schemas:

```bash
uv run datamodel-codegen --input every_eval_ever/schemas/eval.schema.json --output every_eval_ever/eval_types.py --class-name EvaluationLog --output-model-type pydantic_v2.BaseModel --input-file-type jsonschema --formatters ruff-format ruff-check
uv run datamodel-codegen --input every_eval_ever/schemas/instance_level_eval.schema.json --output every_eval_ever/instance_level_types.py --class-name InstanceLevelEvaluationLog --output-model-type pydantic_v2.BaseModel --input-file-type jsonschema --formatters ruff-format ruff-check
uv run python post_codegen.py
```

## 🔌 Eval Converters

We have prepared converters to make adapting to our schema as easy as possible. At the moment, we support converting local evaluation harness logs from `Inspect AI`, `HELM` and `lm-evaluation-harness` into our unified schema. Each converter produces aggregate JSON and optionally instance-level JSONL output.

| Framework | Command | Instance-Level JSONL |
|---|---|---|
| [Inspect AI](every_eval_ever/converters/inspect/) | `every_eval_ever convert inspect --log_path <path>` | Yes, if samples in log |
| [HELM](every_eval_ever/converters/helm/) | `every_eval_ever convert helm --log_path <path>` | Always |
| [lm-evaluation-harness](every_eval_ever/converters/lm_eval/) | `every_eval_ever convert lm_eval --log_path <path> --include_samples` | With `--include_samples` |

For full CLI usage and required input files, see the [Eval Converters README](every_eval_ever/converters/README.md).

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
