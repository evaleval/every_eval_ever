---
layout: default
title: Validation
parent: Data Structure
nav_order: 2
---

# Validation

Validation rejects invalid JSON, applies the generated schema models, and runs the same repository checks used by the PR bot. Each `.json` or `.jsonl` file is validated independently. Requires [uv](https://docs.astral.sh/uv/).

## Validate files with the package CLI

```sh
# Single aggregate file
uv run python -m every_eval_ever validate data/benchmark/dev/model/uuid.json

# Instance-level JSONL
uv run python -m every_eval_ever validate data/benchmark/dev/model/uuid_samples.jsonl

# One model folder (direct files only; does not visit subfolders)
uv run python -m every_eval_ever validate data/benchmark/dev/model/

# Multiple paths
uv run python -m every_eval_ever validate file1.json file2_samples.jsonl
```

Run the command from the repository root and use `data/...` paths. File type is determined by extension: `.json` validates against `EvaluationLog`, while `.jsonl` validates each line against `InstanceLevelEvaluationLog`. Directory arguments include only direct `.json` and `.jsonl` children; validation never walks subfolders.

### Output formats

```sh
# Rich terminal output (default)
uv run python -m every_eval_ever validate data/benchmark/dev/model/

# Machine-readable JSON
uv run python -m every_eval_ever validate --format json data/benchmark/dev/model/

# GitHub Actions annotations
uv run python -m every_eval_ever validate --format github data/benchmark/dev/model/
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--format {rich,json,github}` | `rich` | Output format |
| `--max-errors N` | `50` | Maximum errors reported per JSONL file |

Exit code is `0` if all files pass and `1` if any fail.
