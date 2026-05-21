---
layout: default
title: Schema
parent: Data Structure
nav_order: 1
---

# Schema

The canonical schemas are:

- [Aggregate schema](../../eval.schema.json)
- [Instance-level schema](../../instance_level_eval.schema.json)

Both schema definitions are currently version `0.2.2`.

The repository enforces schema compatibility by generating Pydantic models from JSON Schema and applying post-generation patches (`post_codegen.py`). This generation flow is automated in CI and can also be run manually.

For aggregate records, keep these conventions:

1. `evaluation_id` uses `{benchmark_name}/{model_id}/{retrieved_timestamp}`
2. `source_metadata.source_type` is `documentation` or `evaluation_run`
3. `source_data` is set per result (`url`, `hf_dataset`, or `other`)
4. Level-based metrics use integer values plus `level_names`
