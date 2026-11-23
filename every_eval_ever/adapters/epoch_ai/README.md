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
- `*_external.csv` rows are scores Epoch compiled from elsewhere: `evaluator_relationship=other`.
  Other CSVs are Epoch's own runs: `third_party`. Both are `source_type=documentation`
  (aggregates only; Epoch's Inspect logs are linked, not re-hosted).
- Model id is `<org>/<Model version>` with a trailing reasoning-effort or thinking-budget
  suffix (`_high`, `_none`, `_16K`, ...) moved to `additional_details` and kept in
  `evaluation_id`. The organization is resolved through the vendored eval-card-registry
  snapshot (open-weights models use the registry's HF namespace); unresolved names fall back to a
  slug and say so in `org_registry_strategy`. Deployment axes come from `model_metadata.csv`
  `accessibility`.
- `evaluation_id` = `epoch-ai/<benchmark>/<model version>/<Epoch row id or row hash>/<snapshot date>`.

## What is not
- ECI and EDI values are not converted.
- Rows with no `Model version`, or no organization anywhere in the archive, or a blank headline
  score, go to the failure report and make the command exit non-zero.
- CSVs that `benchmark_metadata.csv` gives no score column for are reported as exclusions.

Data: CC BY 4.0, credit Epoch AI (https://epoch.ai/benchmarks/use-this-data). The citation is on every record.
