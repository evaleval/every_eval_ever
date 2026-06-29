---
layout: default
title: Home
nav_order: 1
---

# Every Eval Ever

> [EvalEval Coalition](https://evalevalai.com) — "We are a researcher community developing scientifically grounded research outputs and robust deployment infrastructure for broader impact evaluations."

**Every Eval Ever** is a shared schema and crowdsourced eval database. It defines a standardized metadata format for storing AI evaluation results — from leaderboard scrapes and research papers to local evaluation runs — so that results from different frameworks can be compared, reproduced, and reused. The three components that make it work:

- 📋 **A metadata schema** ([eval.schema.json](https://github.com/evaleval/every_eval_ever/blob/main/eval.schema.json)) that defines the information needed for meaningful comparison of evaluation results, including [instance-level data](https://github.com/evaleval/every_eval_ever/blob/main/instance_level_eval.schema.json)
- 🔧 **Validation** that checks data against the schema before it enters the repository
- 🔌 **Converters** for [Inspect AI](https://github.com/evaleval/every_eval_ever/tree/main/every_eval_ever/converters/inspect), [HELM](https://github.com/evaleval/every_eval_ever/tree/main/every_eval_ever/converters/helm), and [lm-eval-harness](https://github.com/evaleval/every_eval_ever/tree/main/every_eval_ever/converters/lm_eval), so you can transform your existing evaluation logs into the standard format

## Project Components

Every Eval Ever is maintained across three connected components:

- [GitHub repository](https://github.com/evaleval/every_eval_ever): the `every_eval_ever` Python package with schema definitions, converters/adapters, tests, and core tooling.
- [EEE Datastore](https://huggingface.co/datasets/evaleval/EEE_datastore): the Hugging Face datastore that stores normalized Every Eval Ever evaluation data.
- [EEE Validator](https://huggingface.co/spaces/evaleval/eee_validator): validator and EvalEvalBot checks used on datastore pull requests, built from repository logic plus additional checks that are being upstreamed.

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