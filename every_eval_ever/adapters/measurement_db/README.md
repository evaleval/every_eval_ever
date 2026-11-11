# AI Measurement Data Bank (`measurement_db`) adapter

Converts **AI Measurement Data Bank** (`aims-foundations/measurement-db`,
[aimslab.stanford.edu/measurement-db](https://aimslab.stanford.edu/measurement-db))
item-level evaluations into Every Eval Ever records under `data/measurement-db/`.

## What is converted

- **Source:** `aims-foundations/measurement-db` on Hugging Face (six Parquet
  tables per benchmark: `benchmarks`, `subjects`, `items`, `responses`,
  `traces`, `assets`, plus `characterization.yaml` table SHA-256 digests).
- **Deduplication:** By default, skips the 26 benchmarks already covered by
  dedicated item-level or suite adapters in Every Eval Ever (`OVERLAPPING_BENCHMARKS`:
  `helm_*`, `medhelm_*`, `mmlu`, `truthfulqa_mc`) and refuses the 12 benchmarks
  marked `release: withheld` upstream (`WITHHELD_BENCHMARKS`). Pass
  `--include-overlapping` to include the overlapping benchmarks.
- **Aggregate records (`<uuid4>.json`):** One file per `(benchmark_id, subject_id)`
  that has at least one scored response. Multi-condition benchmarks also emit
  one `EvaluationResult` per `test_condition` alongside the overall result.
- **Instance sidecars (`<uuid4>_samples.jsonl`):** Emitted when `--include-instances`
  is passed. Each graded response row produces one `InstanceLevelEvaluationLog`
  whose `sample_hash` is computed directly over `input.raw` (`items.content`) and
  `input.reference` (`grading_criterion.reference_answer`). `output.raw` is read
  from `traces.trace` (capped at 10,000 characters when needed) or falls back to
  the recorded response value when `traces.parquet` is absent.

## Usage

Convert a local `measurement-db` benchmark directory (with companion
`<uuid4>_samples.jsonl` files):

```bash
uv run python -m every_eval_ever.adapters.measurement_db.adapter \
  --benchmark-dir /path/to/measurement-db/benchmarks/matharena \
  --output-dir data/measurement-db \
  --include-instances
uv run python -m every_eval_ever validate 'data/measurement-db/*/*/*.json*'
uv run python -m every_eval_ever.check_duplicate_entries --path data/measurement-db
```

Fetch specific benchmarks from `aims-foundations/measurement-db` on Hugging Face
(requires `HF_TOKEN` with access to the gated dataset):

```bash
uv run python -m every_eval_ever.adapters.measurement_db.adapter \
  --benchmarks matharena mmdocrag \
  --output-dir data/measurement-db \
  --include-instances
```

Print the resolved upstream source commit for scheduled incremental checks:

```bash
uv run python -m every_eval_ever.adapters.measurement_db.adapter \
  --emit-source-version
```
