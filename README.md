# every_eval_ever

## What is EvalEval?

> "We are a researcher community developing scientifically grounded research outputs and robust deployment infrastructure for broader impact evaluations."[EvalEval Coalition](https://evalevalai.com)

The EvalEval Coalition focuses on conducting rigorous research on AI evaluation methods, building practical infrastructure for evaluation work, and organizing collaborative efforts across their researcher community. This repository, **every_eval_ever**, provides a standarized metadata format for storing evaluation results from various leaderboards, research, and local evaluations.

## Contributor Guide
Leaderboard/evaluation data is split-up into files by individual model, and data for each model is stored using [this JSON Schema](https://github.com/evaleval/every_eval_ever/blob/main/eval.schema.json). The repository is structured into folders as `{leaderboard_name}/{developer_name}/{model_name}/`.

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

1. Add a new folder under `/data` with a codename for your eval.
2. For each model, use the HuggingFace (`developer_name/model_name`) naming convention to create a 2-tier folder structure.
3. Add a JSON file with results for each model and name it `{uuid}.json`.
4. [Optional] Include a `scripts` folder in your eval name folder with any scripts used to generate the data.
5. [Validate] Validation Script: Adds workflow (`workflows/validate-data.yml`) that runs validation script (`scripts/validate_data.py`) to check JSON files against schema and report errors before merging.

### Schema Instructions

1. **`model_info`**: Use HuggingFace formatting (`developer_name/model_name`). If a model does not come from HuggingFace, use the exact API reference. Check [examples in /data/livecodebenchpro](data/livecodebenchpro/). Notably, some do have a **date included in the model name**, but others **do not**. For example:
- OpenAI: `gpt-4o-2024-11-20`, `gpt-5-2025-08-07`, `o3-2025-04-16`
- Anthropic: `claude-3-7-sonnet-20250219`, `claude-3-sonnet-20240229`
- Google: `gemini-2.5-pro`, `gemini-2.5-flash`
- xAI (Grok): `grok-2-2024-08-13`, `grok-3-2025-01-15`

2. **`evaluation_id`**: Use `{eval_name/model_id/retrieved_timestamp}` format (e.g. `livecodebenchpro/qwen3-235b-a22b-thinking-2507/1760492095.8105888`).

3. **`inference_platform`** vs **`inference_engine`**: Where possible specify where the evaluation was run using one of these two fields.
- `inference_platform`: Use this field when the evaluation was run through a remote API (e.g., `openai`, `huggingface`, `openrouter`, `anthropic`, `xai`).
- `inference_engine`: Use this field when the evaluation was run through a local inference engine (e.g. `vLLM`, `Ollama`).

4. The `source_type` has two options: `documentation` and `evaluation_platform`. Use `documentation` when the evaluation results are extracted from a documentation source (e.g., a leaderboard website or API). Use `evaluation_platform` when the evaluation was run locally or through an evaluation platform.

5. The schema is designed to accomodate both numeric and level-based (e.g. Low, Medium, High) metrics. For level-based metrics, the actual 'value' should be converted to an integer (e.g. Low = 1, Medium = 2, High = 3), and the 'level_names' propert should be used to specify the mapping of levels to integers.

6. Additional details can be provided in several places in the schema. They are not required, but can be useful for detailed analysis.
- `model_info.additional_details`: Use this field to provide any additional information about the model itself (e.g. number of parameters)
- `evaluation_results.generation_config.generation_args`: Specify additional arguments used to generate outputs from the model
- `evaluation_results.generation_config.additional_details`: Use this field to provide any additional information about the evaluation process that is not captured elsewhere


## Data Validation

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

## Repository Structure

```
data/
├── {eval_name}/
│   └── {developer_name}/
│       └── {model_name}/
│           └── {uuid}.json
│
scripts/
└── validate_data.py

.github/
└── workflows/
    └── validate-data.yml
```

Each evaluation (e.g., `livecodebenchpro`, `hfopenllm_v2`) has its own directory under `data/`. Within each evaluation, models are organized by model name, with a `{uuid}.json` file containing the evaluation results for that model.

## Detailed Example

```
{
  "schema_version": "0.1.0",
  "evaluation_id": "hfopenllm_v2/Qwen_Qwen2.5-Math-72B-Instruct/1762652579.847774", # {eval_name}/{model_id}/{retrieved_timestamp}
  "retrieved_timestamp": "1762652579.847775",  # UNIX timestamp
  "source_data": [
    "https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted"
  ],
  "source_metadata": { # This information will be repeated in every model file
    "source_name": "HF Open LLM v2",
    "source_type": "documentation" # This can be documentation OR evaluation_run
    "source_organization_name": "Hugging Face",
    "evaluator_relationship": "third_party"
  },
  "model_info": {
    "name": "Qwen/Qwen2.5-Math-72B-Instruct",
    "developer": "Qwen",
    "inference_platform": "unknown",
    "id": "Qwen/Qwen2.5-Math-72B-Instruct",
    "additional_details": { # Optional details about the model
        "precision": "bfloat16",
        "architecture": "Qwen2ForCausalLM",
        "params_billions": 72.706
    }
  },
  "evaluation_results": [
    {
      "evaluation_name": "IFEval",
      "metric_config": { # This information will be repeated in every model file
        "evaluation_description": "Accuracy on IFEval",
        "lower_is_better": false,
        "score_type": "continuous",
        "min_score": 0,
        "max_score": 1
      },
      "score_details": {
        "score": 0.4003466358151926
      }
    }
...
  ]
}
```

**Level-based metrics example**

```
    {
      "evaluation_name": "Data Transparency Rating",
      "metric_config": {
        "evaluation_description": "Evaluation of data documentation transparency",
        "lower_is_better": false,
        "score_type": "level",
        "level_names": ["Low", "Medium", "High"]
      },
      "score_details": {
        "score": 1
      }
    }
```

## Automatic Evaluation Log Converters
A collection of scripts to convert evaluation logs from local runs of evaluation benchmarks (e.g., Inspect AI and lm-eval-harness).

### Installation
- Install the required dependencies:

```bash
uv sync
```

### Inspect
Convert eval log from Inspect AI into json format with following command:

```bash
uv run inspect log convert path_to_eval_file_generated_by_inspect --to json --output-dir inspect_json
```

Then we can convert Inspect evaluation log into unified schema via `eval_converters/inspect/converter.py`. Conversion for example data can be generated via below script: 

for example:

```bash
uv run python3 -m scripts.eval_converters.inspect --log_path tests/data/inspect/data_pubmedqa_gpt4o_mini.json
```


Full manual for conversion of your own Inspect evaluation log into unified is available below:

```bash
usage: __main__.py [-h] [--log_path LOG_PATH] [--output_dir OUTPUT_DIR] [--source_organization_name SOURCE_ORGANIZATION_NAME]
                   [--evaluator_relationship {first_party,third_party,collaborative,other}] [--source_organization_url SOURCE_ORGANIZATION_URL]
                   [--source_organization_logo_url SOURCE_ORGANIZATION_LOGO_URL]

options:
  -h, --help            show this help message and exit
  --log_path LOG_PATH
  --output_dir OUTPUT_DIR
  --source_organization_name SOURCE_ORGANIZATION_NAME
                        Orgnization which pushed evaluation to the every-eval-ever.
  --evaluator_relationship {first_party,third_party,collaborative,other}
                        Relationship of evaluation author to the model
  --source_organization_url SOURCE_ORGANIZATION_URL
  --source_organization_logo_url SOURCE_ORGANIZATION_LOGO_URL
```