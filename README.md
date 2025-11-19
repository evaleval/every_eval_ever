# every_eval_ever

## Contributor Guide
Leaderboard/evaluation data is split-up into files by individual model, and data for each model is stored using [this JSON Schema](https://github.com/evaleval/evalHub/blob/main/schema/eval.schema.json). The repository is structured into folders as `{leaderboard_name}/{developer_name}/{model_name}/`. The individual JSON files are titled `{uuid}.json` - a model may have multiple result files, representing different iterations of the leaderboard/evaluation.

Note: Each file can contain multiple individual results related to one model.

How to add a new leaderboard:

1. Add a new folder under `/data` with a codename for your leaderboard.
2. For each model, use the HuggingFace (`developer_name/model_name`) naming convention to create a 2-tier folder structure.
3. Add a JSON file with results for each model and name it `{uuid}.json`.
4. [Optional] Include a `scripts` folder in your leaderboard folder with any scripts used to generate the data.
5. [Validate] Validation Script: Adds workflow (`workflows/validate-data.yml`) that runs validation script (`scripts/validate_data.py`) to check JSON files against schema and report errors before merging.

Schema Instructions:
1. `model_info`: Use HuggingFace formatting (`developer_name/model_name`). If a model does not come from HuggingFace, use standardized model format or the exact API reference (e.g. `gpt4o-2024-08-06` or `claude-3-sonnet-20240229`). 
2. `evaluation_id`: Use `{org_name}/{eval_name}/{retrieved_timestamp}` format.
3. `inference_platform`: The platform where the model was run (e.g. `openai`, `huggingface`, `openrouter`).

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
