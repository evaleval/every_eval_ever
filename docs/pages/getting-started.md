# Getting Started

Use this path when you are preparing Every Eval Ever data from this checkout. The project already has a `pyproject.toml` and `uv.lock`, so do not initialize a second Python environment.

## Install The Repo

```sh
uv sync
uv run every_eval_ever --help
```

Optional converter dependencies are installed by extra:

```sh
uv sync --extra inspect
uv sync --extra helm
uv sync --extra all
```

For package consumers outside this checkout, install the published package and run the installed CLI:

```sh
pip install every-eval-ever
every_eval_ever --help
```

## Choose A Contribution Path

- Manual aggregate entry: write a UUID-named JSON record that follows the [Data Model](/data-model/).
- Aggregate plus samples: add a companion `{uuid}_samples.jsonl` file and point to it from `detailed_evaluation_results`.
- Existing framework output: use one of the [Converters](/converters/) and inspect the generated JSON before submission.
- Hugging Face Community Evals work: use the [HF Community Evals](/hf-community-evals/) converter, which has a separate review and approval flow.

## Validate Early

Validate a file, folder, or mixed list of paths before opening a datastore PR:

```sh
uv run every_eval_ever validate data/my-benchmark/org/model/00000000-0000-4000-8000-000000000000.json
uv run every_eval_ever validate data/my-benchmark/
uv run every_eval_ever validate --format json data/my-benchmark/
```

Validation dispatches by extension. `.json` files are aggregate `EvaluationLog` records. `.jsonl` files are instance-level `InstanceLevelEvaluationLog` records.

## Repository Map

| Path | Purpose |
| --- | --- |
| `every_eval_ever/schemas/` | Bundled JSON Schemas for aggregate and instance-level data. |
| `every_eval_ever/eval_types.py` | Generated Pydantic types for aggregate records. |
| `every_eval_ever/instance_level_types.py` | Generated Pydantic types for sample-level records. |
| `every_eval_ever/validate.py` | Validation implementation used by the CLI. |
| `every_eval_ever/converters/` | Package converters for supported evaluation frameworks. |
| `tools/hf-community-evals/` | Review and PR tooling for HF Community Evals YAML. |
| `tests/` | Unit tests and fixtures for validators, adapters, converters, and docs build behavior. |

## Required Review Habit

Generated converter output is not a silent fallback for missing metadata. Check model identifiers, source metadata, metric definitions, timestamps, and sample references before submitting.
