# every_eval_ever

## Contributor Guide

Leaderboard/evaluation data is split-up into files by individual model, and data for each model is stored using [this JSON Schema](https://github.com/evaleval/evalHub/blob/main/schema/eval.schema.json). The repository is structured into folders as `{leaderboard_name}/{developer_name}/{model_name}/`. The individual JSON files are titled `{uuid}.json` - a model may have multiple result files, representing different iterations of the leaderboard/evaluation.

NOTE: Each file can contain multiple individual results related to one model.

To add a new leaderboard:

1. Add a new folder under `/data` with a codename for your leaderboard.
2. For each model, use the HuggingFace (`developer_name/model_name`) naming convention to create a 2-tier folder structure.
3. Add a JSON file with results for each model and name it `{uuid}.json`.
4. [Optional] Include a `scripts` folder in your leaderboard folder with any scripts used to generate the data.

TODO: How to test the data against the JSON Schema?

Notes related to the schema:

1. If a model does not come from HuggingFace, do a reasonable effort of capturing the developer name in a standard format (e.g. `openai` or `anthropic`)
2.
3. TODO: inference_engine vs inference_platform
