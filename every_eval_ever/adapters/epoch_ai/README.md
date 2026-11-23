# Epoch AI Benchmarking Hub adapter

Converts `benchmark_data.zip` from Epoch AI's Benchmarking Hub (the data behind the
Epoch Capabilities Index) into EEE aggregate records in the `epoch-ai-benchmarking`
collection.

```
uv run python -m every_eval_ever.adapters.epoch_ai.adapter --output-dir /tmp/epoch-ai/data/epoch-ai-benchmarking
uv run python -m every_eval_ever.adapters.epoch_ai.adapter --input benchmark_data.zip --output-dir ...
uv run python -m every_eval_ever validate '/tmp/epoch-ai/data/epoch-ai-benchmarking/*/*/*.json'
```

`--input` reads a local zip; otherwise `--url` (default `https://epoch.ai/data/benchmark_data.zip`)
is fetched. `--no-registry-resolve` skips organization resolution.

## What is converted
- `benchmark_metadata.csv` names, per benchmark, its CSV and headline score column and the
  scale (`1.0` = proportion, `0.01` = percent). One log per CSV row, on that column, on that
  scale (`metric_unit` and bounds follow it; nothing is rescaled).
- Other columns of the row go to `score_details.details` verbatim; `Source`, `Source link`,
  `Notes`, `Log viewer` and `Logs` go to `source_metadata.additional_details` as `epoch_*`.
  The model columns (`Model version`, `Organization`, `Country`, `Release date`,
  `Training compute (FLOP)`, `Training compute notes`) go to `model_info.additional_details`
  as `epoch_*`, so every source column is kept somewhere and an excluded duplicate differs
  from the kept row in its Epoch row id only.
- Metric ids: the columns `Accuracy` and `Overall accuracy` map to `accuracy`, `EM` to
  `exact_match` and `Pass@1 score` to `pass_at_k` (k = 1); every other column keeps a
  benchmark-namespaced `epoch-ai.<benchmark>.score`.
- `eval_library` is `inspect_ai` (version unknown) for Epoch's own runs, which Epoch states run
  on Inspect, and `unknown` for external scores.
- `*_external.csv` rows are scores Epoch compiled from elsewhere: `evaluator_relationship=other`.
  Other CSVs are Epoch's own runs: `third_party`. Both are `source_type=documentation`
  (aggregates only; Epoch's Inspect logs are linked, not re-hosted).
- Model id is `<org>/<model>` with a trailing reasoning-effort or thinking-budget
  suffix (`_high`, `_none`, `_16K`, ...) moved to `additional_details` and kept in
  `evaluation_id`. A version that carries a path (`openai/gpt-oss-120b_high`,
  `chutes/Qwen3-8B`, `accounts/fireworks/models/glm-4p6`) keeps only its last segment, records a
  serving provider as `inference_provider`, and is resolved through Epoch's own `model_group` to
  the group's plain spelling of the same model (`fireworks/kimi-k2p5` -> `kimi-k2.5`); the raw
  spelling stays in `epoch_model_version`. Collaborative models whose repository is not under
  the first listed organization use a reviewed override checked against the Hugging Face
  repository (`bigscience/bloom`, `bigcode/starcoder2-*`, `lmsys/vicuna-13b-v1.3`,
  `PRIME-RL/Eurus-2-7B-PRIME`); the full organization list is kept in `epoch_organizations`.
  The organization is resolved through the vendored eval-card-registry
  snapshot (open-weights models use the registry's HF namespace); unresolved names fall back to a
  slug and say so in `org_registry_strategy`. Deployment axes come from `model_metadata.csv`
  `accessibility`.
- `evaluation_id` = `epoch-ai/<benchmark>/<model version>/<Epoch row id or row hash>/<snapshot date>`.
- Archive-level facts (the zip's sha256 and snapshot date) are not written into the records:
  they would change every record's fingerprint whenever any byte of the archive changes, and
  the cron's raw capture already keeps the zip with its sha256. `retrieved_from` records the
  URL a `--url` run fetched.

## What is not
- ECI and EDI values are not converted.
- Rows with no `Model version`, or no organization anywhere in the archive, or a blank headline
  score, go to the failure report and make the command exit non-zero.
- CSVs that `benchmark_metadata.csv` gives no score column for are reported as exclusions.

## Licensing
Epoch's terms (https://epoch.ai/benchmarks/use-this-data#licensing): Epoch AI's own data is CC BY
4.0, and data "sourced from external projects ... retains its original licensing". Epoch-run
records carry `license: CC-BY-4.0`. External records carry no blanket licence: they have
`epoch_contribution_license: CC-BY-4.0` for Epoch's compilation and notes, a `license_note`
saying the score keeps the original source's licence, and the source in `epoch_source`. Epoch's
citation is on every record.
