# Every Eval Ever

> [EvalEval Coalition](https://evalevalai.com) — "We are a researcher community developing scientifically grounded research outputs and robust deployment infrastructure for broader impact evaluations."

📄 **[Paper (arXiv:2606.14516)](https://arxiv.org/abs/2606.14516)**

**Every Eval Ever** is a shared schema and crowdsourced eval database. It defines a standardized metadata format for storing AI evaluation results — from leaderboard scrapes and research papers to local evaluation runs — so that results from different frameworks can be compared, reproduced, and reused. The three components that make it work:

- 📋 **A metadata schema** ([`eval.schema.json`](every_eval_ever/schemas/eval.schema.json)) that defines the information needed for meaningful comparison of evaluation results, including [instance-level data](every_eval_ever/schemas/instance_level_eval.schema.json)
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

> [!NOTE]
> **`helm` extra + nltk's import guard.** The `helm` extra pulls in `nltk`, and
> nltk ≥ 3.10.1 ships an import guard (`nltk/inisec.py`, a CWE-427 mitigation)
> that blocks nltk-initiated imports of any module whose file resolves *under the
> current working directory*. With the common in-project virtualenv layout
> (`uv`'s `.venv/`, or any venv inside the repo), site-packages sits under the
> CWD, so the guard trips on nltk's own dependencies and importing the HELM
> converter fails. The extra currently caps `nltk<3.10.1`, so this does not bite
> by default. If you move to a newer nltk, keep the environment **outside** the
> checkout — the same thing CI does via `UV_PROJECT_ENVIRONMENT` — e.g.
> `UV_PROJECT_ENVIRONMENT=/tmp/eee-venv uv sync --extra helm`, or create your
> venv outside the repository. Upstream tracker:
> [nltk#3730](https://github.com/nltk/nltk/issues/3730).

### Terminology

| Term | Our Definition | Example |
|---|---|---|
| **Single Benchmark** | Standardized eval using one dataset to test a single capability, producing one score | MMLU — ~15k multiple-choice QA across 57 subjects |
| **Composite Benchmark** | A collection of simple benchmarks aggregated into one overall score, testing multiple capabilities at once | BIG-Bench bundles >200 tasks with a single aggregate score |
| **Metric** | Any numerical or categorical value used to score performance on a benchmark (accuracy, F1, precision, recall, …) | A model scores 92% accuracy on MMLU |

## 🚀 Contributing

Contributing data, writing an adapter, or changing the schema? **[CONTRIBUTING.md](CONTRIBUTING.md)** is the guide: how your PR gets reviewed, how to submit data to the datastore, the naming conventions, and the conventions for filling in each field.

The rest of this README is for *reading* EEE — what the schema says, how to validate a record, and what the converters do.

## 🧩 Instance-Level Data

For evaluations that include per-sample results, the individual results should be stored in a companion `{uuid}_samples.jsonl` file in the same folder (one JSONL per JSON, sharing the same UUID). The aggregate JSON file refers to its JSONL via the `detailed_evaluation_results` field. The instance-level schema ([`instance_level_eval.schema.json`](every_eval_ever/schemas/instance_level_eval.schema.json)) supports three interaction types:

- **`single_turn`**: Standard QA, MCQ, classification — uses `output` object
- **`multi_turn`**: Conversational evaluations with multiple exchanges — uses `messages` array
- **`agentic`**: Tool-using evaluations with function calls and sandbox execution — uses `messages` array with `tool_calls`

Each instance captures: `input` (raw question + reference answer), `answer_attribution` (how the answer was extracted), `evaluation` (score, is_correct), and optional `token_usage` and `performance` metrics. Instance-level JSONL files are produced automatically by the [eval converters](every_eval_ever/converters/README.md).

Example `single_turn` instance:

```json
{
  "schema_version": "0.3.0",
  "evaluation_id": "math_eval/meta-llama/Llama-2-7b-chat/1706000000",
  "model_id": "meta-llama/Llama-2-7b-chat",
  "evaluation_name": "math_eval",
  "sample_id": "4",
  "interaction_type": "single_turn",
  "input": { "raw": "If 2^10 = 4^x, what is the value of x?", "reference": ["5"] },
  "output": { "raw": ["Rewrite 4 as 2^2, so 4^x = 2^(2x). Since 2^10 = 2^(2x), x = 5."] },
  "answer_attribution": [
    { "turn_idx": 0, "source": "output.raw", "extracted_value": "5", "extraction_method": "match", "is_terminal": true }
  ],
  "evaluation": { "score": 1.0, "is_correct": true }
}
```

## 🤖 Agentic Evaluations

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

Validation rejects invalid JSON, applies the generated schema models, and runs the same repository checks used by the PR bot. Aggregate JSON and sample JSONL files are also checked against each other. Requires [uv](https://docs.astral.sh/uv/).

### Validate files with the package CLI

```sh
# Single aggregate file
uv run python -m every_eval_ever validate data/benchmark/dev/model/uuid.json

# Instance-level JSONL
uv run python -m every_eval_ever validate data/benchmark/dev/model/uuid_samples.jsonl

# A fixed-depth glob (quote it so the CLI expands it consistently)
uv run python -m every_eval_ever validate 'data/*/*/*/*.json*'

# Multiple paths
uv run python -m every_eval_ever validate \
  data/benchmark/dev/model/uuid.json \
  data/benchmark/dev/model/uuid_samples.jsonl
```

Run the command from the repository root and use `data/...` paths. File type is determined by extension: `.json` validates against `EvaluationLog`, while `.jsonl` validates each line against `InstanceLevelEvaluationLog`. Paths must be exactly `data/<collection>/<developer>/<model>/<uuid>.json` or the matching `<uuid>_samples.jsonl`. Directory arguments are not accepted; use a fixed-depth or explicit recursive glob to select local files.

When samples exist, both files must be in the same folder and use the same UUID. The aggregate must provide the samples' full repository-relative path (for example, `data/<collection>/<developer>/<model>/<uuid>_samples.jsonl`) in `detailed_evaluation_results.file_path`, and the JSONL must point back to that aggregate. Their evaluation IDs, model IDs, and declared row count must agree.

Local validation checks only the files present in the local checkout and their expected siblings. The PR bot remains responsible for checking every changed datastore path against the complete PR branch.

Collision checks cover the files selected in one validation or publication
operation and files already present at their destination. Validation does not
walk the entire datastore looking for historical route or case-only
collisions; repository-wide checks belong in the PR bot or a separate audit.

For local smoke output outside the checkout, retain the same layout under a
`data/` directory and pass an explicit glob, for example
`'/tmp/run/data/benchmark/*/*/*.json*'`. The CLI maps that absolute path back
to the canonical datastore path before applying the same checks.

#### Output formats

```sh
# Rich terminal output (default)
uv run python -m every_eval_ever validate 'data/*/*/*/*.json*'

# Machine-readable JSON
uv run python -m every_eval_ever validate --format json 'data/*/*/*/*.json*'

# GitHub Actions annotations
uv run python -m every_eval_ever validate --format github 'data/*/*/*/*.json*'
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

Schemas: [`eval.schema.json`](every_eval_ever/schemas/eval.schema.json) (aggregate) · [`instance_level_eval.schema.json`](every_eval_ever/schemas/instance_level_eval.schema.json) (per-sample JSONL)

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

The schema also supports **level-based metrics** (e.g. Low/Medium/High) and **uncertainty** reporting (confidence intervals, standard errors). See [`eval.schema.json`](every_eval_ever/schemas/eval.schema.json) for the full specification.

## 🔧 Auto-generation of Pydantic Classes for Schema

Run the following commands to generate the package-local Pydantic classes from the canonical package-local schemas:

```bash
uv run datamodel-codegen --input every_eval_ever/schemas/eval.schema.json --output every_eval_ever/eval_types.py --class-name EvaluationLog --output-model-type pydantic_v2.BaseModel --input-file-type jsonschema --formatters ruff-format ruff-check
uv run datamodel-codegen --input every_eval_ever/schemas/instance_level_eval.schema.json --output every_eval_ever/instance_level_types.py --class-name InstanceLevelEvaluationLog --output-model-type pydantic_v2.BaseModel --input-file-type jsonschema --formatters ruff-format ruff-check
uv run python -m every_eval_ever.post_codegen
```

Changing the schema or the validator also changes what a contributor has to produce, so
`tests/test_skill_conversion.py` re-validates the contributor-facing
[`eee-dataset-conversion` skill](.claude/skills/eee-dataset-conversion/SKILL.md) — its
templates and one frozen reference conversion — against the live validator. If it goes
red, the guidance is what needs updating; the failure message says which file and gives
the regeneration command. Don't skip it: it is the check that keeps the docs from
quietly telling the next contributor something untrue.

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

- **Track 1: Public Eval Data Parsing** — Parse leaderboards (Chatbot Arena, Open LLM Leaderboard, AlpacaEval, etc.) and academic papers into [our schema](every_eval_ever/schemas/eval.schema.json) and contribute to a unifying database of eval results!
- **Track 2: Proprietary Evaluation Data** — Convert proprietary evaluation datasets into [our schema](every_eval_ever/schemas/eval.schema.json) and contribute to a unifying database of eval results!

| Milestone | Date |
|---|---|
| Submission deadline | May 1, 2026 |
| Results announced | June 1, 2026 |
| Workshop at ACL 2026 | July 7, 2026 |

Qualifying contributors will be invited as co-authors on the shared task paper.

## 📎 Citation

If Every Eval Ever informs your research, please cite the paper:

```bibtex
@misc{batzner2026evaleverunifyingschema,
      title={Every Eval Ever: A Unifying Schema and Community Repository for AI Evaluation Results}, 
      author={Jan Batzner and Sree Harsha Nelaturu and Damian Stachura and Anastassia Kornilova and Jon Crall and Tommaso Cerruti and Yanan Long and Yifan Mai and Sanchit Ahuja and Asaf Yehudai and Marek Šuppa and John P. Lalor and Oluwagbemike Olowe and Jatin Ganhotra and Brian H. Hu and Eliya Habba and Andrew M. Bean and Chang Liu and Sander Land and Steven Dillmann and Aniketh Garikaparthi and Elron Bandel and Saki Imai and James Edgell and Wm. Matthew Kennedy and Jenny Chim and Patrick Meusling and Asteria Kaeberlein and Venkata Ramachandra Karthik Chundi and Manasi Patwardhan and Martin Ku and Austin Meek and Leon Knauer and Brian Wingenroth and Srishti Yadav and Usman Gohar and Felix Friedrich and Michelle Lin and Jennifer Mickel and Arman Cohan and Stella Biderman and Irene Solaiman and Zeerak Talat and Anka Reuel and Mubashara Akhtar and Gjergji Kasneci and Avijit Ghosh and Leshem Choshen},
      year={2026},
      eprint={2606.14516},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.14516}, 
}
```
