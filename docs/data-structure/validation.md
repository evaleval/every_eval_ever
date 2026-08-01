---
layout: default
title: Validation
parent: Data Structure
nav_order: 2
---

# Validation

Validation rejects invalid JSON, applies the generated schema models, and runs the same repository checks used by the PR bot. Aggregate JSON and sample JSONL files are also checked against each other. Requires [uv](https://docs.astral.sh/uv/).

## Validate files with the package CLI

```sh
# Single aggregate file
uv run python -m every_eval_ever validate data/benchmark/dev/model/uuid.json

# Instance-level JSONL
uv run python -m every_eval_ever validate data/benchmark/dev/model/uuid_samples.jsonl

# A fixed-depth glob (quote it so the CLI expands it consistently)
uv run python -m every_eval_ever validate 'data/*/*/*/*.json*'

# Multiple paths
uv run python -m every_eval_ever validate \
  data/benchmark/dev/model/uuid.json \
  data/benchmark/dev/model/uuid_samples.jsonl
```

Run the command from the repository root and use `data/...` paths. File type is determined by extension: `.json` validates against `EvaluationLog`, while `.jsonl` validates each line against `InstanceLevelEvaluationLog`.

Paths must be exactly `data/<collection>/<developer>/<model>/<uuid>.json` or the matching `<uuid>_samples.jsonl`; subfolders below the model are rejected. When samples exist, both files must share a folder and UUID, the aggregate must declare the samples' full repository-relative `data/...` path, and the samples must point back to that aggregate. Evaluation IDs, model IDs, and any declared `total_rows` must agree. Directory arguments are not accepted; use a glob to select local files.

Local validation checks only the files present in the local checkout and their expected siblings. It does not claim that a partial checkout represents the complete datastore. The PR bot uses the PR diff and branch contents for that authoritative check.

For local smoke output outside the checkout, retain the same layout under a
`data/` directory and pass an explicit glob, for example
`'/tmp/run/data/benchmark/*/*/*.json*'`. The CLI maps that absolute path back
to the canonical datastore path before applying the same checks.

### Output formats

```sh
# Rich terminal output (default)
uv run python -m every_eval_ever validate 'data/*/*/*/*.json*'

# Machine-readable JSON
uv run python -m every_eval_ever validate --format json 'data/*/*/*/*.json*'

# GitHub Actions annotations
uv run python -m every_eval_ever validate --format github 'data/*/*/*/*.json*'
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--format {rich,json,github}` | `rich` | Output format |
| `--max-errors N` | `50` | Maximum errors reported per JSONL file |

Exit code is `0` if all files pass and `1` if any fail.
