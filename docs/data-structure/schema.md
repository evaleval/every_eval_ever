---
layout: default
title: Schema
parent: Data Structure
nav_order: 1
---

# Schema

The canonical schemas are:

- [Aggregate schema](https://github.com/evaleval/every_eval_ever/blob/main/eval.schema.json)
- [Instance-level schema](https://github.com/evaleval/every_eval_ever/blob/main/instance_level_eval.schema.json)

Schema versions are defined in the canonical JSON Schema files linked above.

The repository enforces schema compatibility by generating Pydantic models from JSON Schema and applying post-generation patches (`post_codegen.py`). This generation flow is automated in CI and can also be run manually.

## Schema Instructions

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
- `source_type: "url"` - link to a web source (e.g. leaderboard API)
- `source_type: "hf_dataset"` - reference to a Hugging Face dataset (e.g. `{"hf_repo": "google/IFEval"}`)
- `source_type: "other"` - for private or proprietary datasets

6. The schema is designed to accommodate both numeric and level-based (e.g. Low, Medium, High) metrics. For level-based metrics, the actual `value` should be converted to an integer (e.g. Low = 1, Medium = 2, High = 3), and the `level_names` property should be used to specify the mapping of levels to integers.

7. **Timestamps**: The schema has three timestamp fields - use them as follows:
- `retrieved_timestamp` (required) - when this record was created, in Unix epoch format (e.g. `1760492095.8105888`)
- `evaluation_timestamp` (top-level, optional) - when the evaluation was run
- `evaluation_results[].evaluation_timestamp` (per-result, optional) - when a specific evaluation result was produced, if different results were run at different times

8. Additional details can be provided in several places in the schema. They are not required, but can be useful for detailed analysis.
- `model_info.additional_details`: Use this field to provide any additional information about the model itself (e.g. number of parameters)
- `evaluation_results.generation_config.generation_args`: Specify additional arguments used to generate outputs from the model
- `evaluation_results.generation_config.additional_details`: Use this field to provide any additional information about the evaluation process that is not captured elsewhere