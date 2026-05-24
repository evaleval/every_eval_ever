---
layout: default
title: Validation
parent: Data Structure
nav_order: 2
---

# Validation

Validation uses Pydantic models generated from the JSON schemas. This validates aggregate `.json` files against `EvaluationLog` and instance-level `_samples.jsonl` files line-by-line against `InstanceLevelEvaluationLog`. Requires [uv](https://docs.astral.sh/uv/).

## Validate files with the package CLI

```sh
# Single aggregate file
uv run python -m every_eval_ever validate data/benchmark/dev/model/uuid.json

# Instance-level JSONL
uv run python -m every_eval_ever validate data/benchmark/dev/model/uuid_samples.jsonl

# Entire directory (recurses into subdirectories)
uv run python -m every_eval_ever validate data/benchmark/dev/model/

# Multiple paths
uv run python -m every_eval_ever validate file1.json file2_samples.jsonl data/
```

File type is determined by extension: `.json` validates against `EvaluationLog`, `.jsonl` validates each line against `InstanceLevelEvaluationLog`.

### Output formats

```sh
# Rich terminal output (default)
uv run python -m every_eval_ever validate data/

# Machine-readable JSON
uv run python -m every_eval_ever validate --format json data/

# GitHub Actions annotations
uv run python -m every_eval_ever validate --format github data/
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--format {rich,json,github}` | `rich` | Output format |
| `--max-errors N` | `50` | Maximum errors reported per JSONL file |

Exit code is `0` if all files pass and `1` if any fail.
