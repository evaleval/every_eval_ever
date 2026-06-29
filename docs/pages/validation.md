# Validation

Use validation in two places:

1. Run the package CLI locally before opening a datastore PR.
2. Let the Hugging Face PR bot repeat the checks and report review blockers.

Both paths reject malformed aggregate JSON and instance-level JSONL. Neither path invents missing metadata, normalizes ambiguous model ids, or guesses metric direction.

## Local CLI

Run local validation with the package command:

```sh
uv run every_eval_ever validate data/benchmark/dev/model/uuid.json
uv run every_eval_ever validate data/benchmark/dev/model/uuid_samples.jsonl
uv run every_eval_ever validate data/benchmark/dev/model/
uv run every_eval_ever validate file1.json file2_samples.jsonl data/
```

Directories are expanded recursively. `.json` files are treated as aggregate `EvaluationLog` records. `.jsonl` files are treated as instance-level `InstanceLevelEvaluationLog` rows.

Useful options:

| Option | Use |
| --- | --- |
| `--format rich` | Human-readable terminal output. This is the default. |
| `--format json` | Machine-readable reports for scripts. |
| `--format github` | GitHub annotation output. |
| `--max-errors N` | Cap reported errors per JSONL file. The default is `50`. |

The command exits with `0` when every file passes and `1` when any file fails. If no files are found, it prints an explicit error and exits with `1`.

## Duplicate Check

Run duplicate detection separately for aggregate JSON records:

```sh
uv run every_eval_ever check-duplicates data/benchmark/
```

This command ignores scrape-specific fields such as `evaluation_id` and `retrieved_timestamp`, then reports records with the same substantive content.

## PR Bot

The Hugging Face datastore PR bot runs after a PR opens. It validates changed `data/**/*.json` and `data/**/*.jsonl` files, checks paths, compares aggregate candidates against accepted records, and posts a visible PR report.

Public PR commands:

```text
/eee validate changed
/eee schema status
```

Maintainer-only commands:

```text
/eee validate full [folder]
/eee parquet [benchmark]
/eee parquet update_readme
```

The bot should fail closed for required deployment state. Missing bucket state, dataset metadata, or write credentials should become explicit errors, not silent defaults.

## What Blocks Review

Treat these as blockers:

- JSON parse errors.
- Schema validation errors.
- Unsupported file extensions in the validation target.
- Invalid datastore paths.
- Duplicate aggregate records that match accepted data.
- Forced bot validation that cannot inspect the requested files.

Treat warnings as review items that need an explicit decision. Do not rely on either validator to fill in source metadata, model identifiers, timestamp meaning, or metric semantics.

## Bot State

The PR bot keeps private mutable state in `validator_state.json` and accepted-data metadata in the public datastore `manifest.json`.

Deduplication is semantic and collection-scoped. UUIDs, timestamps, free-form details, paths, and source metadata should not be identity fields.
