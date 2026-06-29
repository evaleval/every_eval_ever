# Converters

Converters transform supported framework outputs into Every Eval Ever aggregate records and, where supported, companion sample files. Review generated output before submission; converters preserve what they can read, but they do not certify that source metadata is complete.

## Supported Package Converters

| Source | Command family | Notes |
| --- | --- | --- |
| lm-eval-harness | `uv run every_eval_ever convert lm_eval` | Can include sample JSONL when source samples are available. |
| Inspect AI | `uv run --extra inspect every_eval_ever convert inspect` | Supports `.eval`, `.json`, and directories of logs. |
| HELM | `uv run --extra helm every_eval_ever convert helm` | Requires HELM run files such as `run_spec.json`, `scenario.json`, and `stats.json`. |
| AlpacaEval | `uv run every_eval_ever convert alpaca_eval` | Fetches public leaderboard data; no local log path is required. |

## lm-eval-harness

```sh
uv run every_eval_ever convert lm_eval \
  --log_path tests/data/lm_eval/results_2026-01-21T03-44-18.458309.json \
  --output_dir data \
  --source_organization_name "Example Lab" \
  --evaluator_relationship third_party
```

Add `--include_samples` when the original run wrote sample logs and you want instance-level JSONL.

## Inspect AI

```sh
uv run --extra inspect every_eval_ever convert inspect \
  --log_path tests/data/inspect/data_pubmedqa_gpt4o_mini.json \
  --output_dir data \
  --source_organization_name "Example Lab"
```

Inspect conversion can operate on one file or a directory of logs. The adapter can also use supplemental evaluation details when a source log lacks metadata that must be explicit in EEE output.

## HELM

```sh
uv run --extra helm every_eval_ever convert helm \
  --log_path tests/data/helm/commonsense:dataset=hellaswag,method=multiple_choice_joint,model=eleutherai_pythia-1b-v0 \
  --output_dir data
```

A single HELM run directory must contain the required HELM output files. A parent directory may contain multiple run directories.

## AlpacaEval

```sh
uv run every_eval_ever convert alpaca_eval --output_dir data
uv run every_eval_ever convert alpaca_eval --version v2 --output_dir data
```

Omitting `--version` converts both supported leaderboard versions.

## Shared Metadata Flags

All package converters accept shared metadata flags:

- `--source_organization_name`
- `--evaluator_relationship`
- `--source_organization_url`
- `--source_organization_logo_url`
- `--eval_library_name`
- `--eval_library_version`

Use these flags to avoid placeholder metadata in submitted records.

## After Conversion

Run [Validation](/validation/) over the generated files, inspect metric definitions, and verify the datastore folder path follows the [Data Model](/data-model/).
