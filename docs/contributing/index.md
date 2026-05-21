---
layout: default
title: Contributing
nav_order: 5
---

# Contributing

Data contributions land in the datastore, while validation gates run through the validator/EvalEvalBot workflow.

To contribute evaluation data:

1. Add files under `data/{benchmark}/{developer}/{model}/`
2. Name aggregate files as `{uuid}.json`
3. Optionally add instance-level `{uuid}_samples.jsonl`
4. Validate before submission

Datastore: https://huggingface.co/datasets/evaleval/EEE_datastore

The validator checks datastore pull requests using core checks from this repository and additional checks that are being upstreamed.

Before submitting, run:

```bash
uv run python -m every_eval_ever validate data/
```
