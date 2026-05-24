---
layout: default
title: Contributing
nav_order: 5
---

# Contributing

New data can be contributed to the [Hugging Face Dataset](https://huggingface.co/datasets/evaleval/EEE_datastore) using the following process:

Leaderboard/evaluation data is split-up into files by individual model, and data for each model is stored using [eval.schema.json](https://github.com/evaleval/every_eval_ever/blob/main/eval.schema.json). The repository is structured into folders as `data/{benchmark_name}/{developer_name}/{model_name}/`.

## TL;DR How to successfully submit

1. Data must conform to [eval.schema.json](https://github.com/evaleval/every_eval_ever/blob/main/eval.schema.json) (current version is defined in the schema file)
2. The validation pipeline will automatically verify the data submitted in the pull request, but can also be manually triggered by typing `/eee validate changed` in a comment on the HF PR.
3. An EvalEval member will review and merge your submission

## PR Naming Convention

Use these prefixes in your pull request titles:

- `[Submission]` - New evaluation data
- `[Issue #N]` - Fix for a specific GitHub issue
- `[Feature]` - New functionality not tied to an issue
- `[Docs]` - Documentation changes
- `[ACL Shared Task]` - Shared task submissions (priority review)

## UUID Naming Convention

Each JSON file is named with a **UUID (Universally Unique Identifier)** in the format `{uuid}.json`. The UUID is automatically generated (using standard UUID v4) when creating a new evaluation result file. This ensures that:
- **Multiple evaluations** of the same model can exist without conflicts (each gets a unique UUID)
- **Different timestamps** are stored as separate files with different UUIDs (not as separate folders)
- A model may have multiple result files, with each file representing different iterations or runs of the leaderboard/evaluation
- UUIDs can be generated using Python's `uuid.uuid4()` function.

**Example**: The model `openai/gpt-4o-2024-11-20` might have multiple files like:
- `e70acf51-30ef-4c20-b7cc-51704d114d70.json` (evaluation run #1)
- `a1b2c3d4-5678-90ab-cdef-1234567890ab.json` (evaluation run #2)

Note: Each file can contain multiple individual results related to one model. See [examples in the datastore](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data).

## How to add new eval

1. Add a new folder under [data/](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data) on the Hugging Face datastore with a codename for your eval.
2. For each model, use the Hugging Face (`developer_name/model_name`) naming convention to create a 2-tier folder structure.
3. Add a JSON file with results for each model and name it `{uuid}.json`.
4. [Optional] Include a [utils/](https://github.com/evaleval/every_eval_ever/tree/main/utils) folder in your benchmark name folder with any scripts used to generate the data (see e.g. [utils/global-mmlu-lite/adapter.py](https://github.com/evaleval/every_eval_ever/blob/main/utils/global-mmlu-lite/adapter.py)).
5. [Submit] Two ways to submit your evaluation data:
	- **Option A: Drag & drop via Hugging Face** - Go to [evaleval/EEE_datastore](https://huggingface.co/datasets/evaleval/EEE_datastore) -> click "Files and versions" -> "Contribute" -> "Upload files" -> drag and drop your data -> select "Open as a pull request to the main branch". See [step-by-step screenshots](https://docs.google.com/document/d/1dxTQF8ncGCzaAOIj0RX7E9Hg4THmUBzezDOYUp_XdCY/edit?usp=sharing).
	- **Option B: Clone & PR** - Clone the [Hugging Face repository](https://huggingface.co/datasets/evaleval/EEE_datastore), add your data under `data/`, and open a pull request

Before submitting, run:

```bash
uv run python -m every_eval_ever validate data/
```
