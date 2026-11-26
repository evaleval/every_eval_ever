# SMB Bench

Converts the run table of [SMB Bench](https://oroagents.com/benchmarks/smb-bench)
(Small-to-Midsize Business Bench, by ORO AI) into `data/smb-bench/`.

SMB Bench is an agentic benchmark. A model runs the buying desk of a simulated
produce wholesaler for a 90-day quarter, and the score is the business's
profit in US dollars. Six businesses are each played three ways (a normal
quarter, a customer quarter and a supplier quarter), which gives 18 tasks, and
every model runs every task 3 times: 54 runs per model.

```bash
uv run python -m every_eval_ever.adapters.smb_bench.adapter \
    --output-dir /tmp/smb-bench-smoke/data/smb-bench
```

Needs no extras and no credentials: the adapter uses the core package,
`huggingface_hub` and `duckdb`. `--input-dir <dir>` replays a local copy of the
dataset (`data/runs.parquet`, `data/models.parquet` and `data/prices.parquet`)
instead of fetching.

## Source

The Hugging Face dataset
[`oro-agents/smb-bench`](https://huggingface.co/datasets/oro-agents/smb-bench)
(CC BY 4.0, [doi:10.57967/hf/11641](https://doi.org/10.57967/hf/11641)), read
at the revision pinned in `DEFAULT_REVISION`:

- `data/runs.parquet`: one row per run, failed runs included (864 rows at the
  pinned revision: 16 models x 18 tasks x 3 repeats);
- `data/models.parquet`: the model id sent in every request and the request
  settings;
- `data/prices.parquet`: how each model's cost was computed.

`source_type` is `evaluation_run`: the adapter holds the per-run outputs and
computes every figure from them. `evaluator_relationship` is `third_party`:
the benchmark's authors ran other developers' models.

No `_samples.jsonl` sidecar is emitted. One "sample" here is a 90-day run of
thousands of tool calls; the dataset publishes each run's trace under
`traces/`, and re-hosting them in the instance schema would duplicate about
seven gigabytes of public data without adding information.

## What each record holds

One `EvaluationLog` per model, with six results:

| `evaluation_name` | `metric_id` | Scope |
|---|---|---|
| `smb-bench-v1` | `smb-bench.mean-profit-usd` | all 18 tasks (the benchmark's headline) |
| `smb-bench-v1.normal-quarters` | `smb-bench.mean-profit-usd` | the 6 normal quarters |
| `smb-bench-v1.customer-quarters` | `smb-bench.mean-profit-usd` | the 6 customer quarters |
| `smb-bench-v1.supplier-quarters` | `smb-bench.mean-profit-usd` | the 6 supplier quarters |
| `smb-bench-v1` | `smb-bench.mean-profit-vs-fixed-rule-usd` | all 18 tasks, paired with the fixed buying rule |
| `smb-bench-v1` | `cost-per-task` | every scored run |

A profit figure is the mean over tasks of the mean of the model's scored runs
on that task, so every task weighs the same. That is how the benchmark
publishes it, and the headline equals the leaderboard's "mean profit per
quarter" to the cent.

The three kinds of quarter partition the 18 tasks, so each profit result names
its level in `score_details.details.aggregation_level` (`overall` or
`kind:<name>`) and carries its own task and run counts. Take the overall or
the parts, not both.

## Decisions worth a reviewer's eye

**Profit is unbounded.** A business can lose or make any amount, so
`min_score` and `max_score` are `-Infinity` and `Infinity`, with
`metric_unit: usd`. The metric ids are namespaced because no registry metric
describes a dollar profit.

**The standard error is over tasks.** `standard_error.value` is the sample
standard deviation (n-1) of the per-task values divided by the square root of
the number of tasks, with `num_samples` the number of tasks (18 overall, 6 per
kind of quarter) and `standard_deviation` that same per-task standard
deviation. Repeats are averaged within a task first: the benchmark's card
states that tasks, not repeats, are its sampling unit. For the result against
the fixed rule the per-task value is the paired difference. The benchmark does
not publish an interval for the dollar means, so none is recorded. The cost
result carries no uncertainty.

**Failed runs are counted, not scored.** A run whose `status` is not `valid`
has no score upstream. It is left out of every mean, never counted as zero,
and reported in the headline result's `details` (`run_status_counts`, and
`unscored_run_ids` with each run's id and status). At the pinned revision 5 of
864 runs failed, across four models, and every model still has a scored run
on every task. If a task ever has no scored run, it is left out of the task
mean and the result's description says how many tasks it covers.

**The fixed buying rule is a baseline, not a model.** The benchmark's scripted
reference plays every task in the same world. It gets no record. Its mean
profit on the same tasks is in `score_details.details.fixed_rule_mean_profit_usd`
and one result reports the model's profit minus the rule's. The benchmark's
card also quotes a scripted expert buyer and a perfect-information solver;
those figures are not in the run table, so they are not converted.

**Customer and supplier quarters come from the benchmark's run pages.** The
run table separates normal quarters (`split == 'control'`) from changed ones
but does not say which changed quarter is which. `QUARTER_KIND_BY_TASK` is the
listing on <https://oroagents.com/smb-bench/runs>, and every row is checked
against `split`; a task missing from it fails its model.

**Cost uses the registry's `cost-per-task`.** One run is one attempt at one
task, and the figure is the mean cost of a scored run in US dollars (bounds
`[0, Infinity]`, lower is better). The benchmark costs completed model calls
from its own price table and calls the result an estimate, not invoices. Most
models are costed at provider list prices; one has no published list price
and is costed at an upper rate. Each cost result therefore carries the price
table's own `cost_method` for that model.

**Model ids are pinned from the registry seed.** `registry_snapshot.json`
holds the canonical id of each model, resolved with the eval-card-registry's
own resolver on a local build of its seed (revision in `_meta`).
`refresh_registry_snapshot.py` regenerates it and tries, in order: the model
id that was served; that id under the organization the benchmark's own model
label resolves to, when the served id names no organization; the label. The
served id and the label stay in the record, with the field that matched
(`model_registry_matched_field`), and `model_availability` is the registry's
`open_weights`. A model missing from the snapshot, or any model under
`--no-registry-resolve`, keeps the source's spelling and is marked
`model_id_verified: "false"`; that never fails the run. Four cases to know:

- `deepseek-v4.1-flash-1017` was served as
  `deepseek-ai/DeepSeek-V4.1-Flash-1017-Fast`, which the registry does not
  hold; it resolves on the benchmark's label to
  `deepseek-ai/DeepSeek-V4.1-Flash-1017`.
- `gpt-6-luna-high` was served as `gpt-6-luna` at reasoning effort high and
  resolves to `openai/gpt-6-luna`. The effort is in `generation_config`.
- `muse-ember` was served as `muse-ember-1.0`. On its own that resolves to a
  draft id with no organization (`muse-ember-1-0`), so it is resolved under
  the label's organization to `meta/muse-ember-1-0`.
- The three Muse ids (`meta/muse-ember-1-0`, `meta/muse-wonder-1-0`,
  `meta/muse-spark-1.3`) are `draft` in the registry, which also holds
  unversioned drafts for the same models.

**The models table's `provider` column is not published.** The dataset
defines it as the maker of the model, not the serving route. The adapter uses
it only as the fallback developer when a model is not in the snapshot;
`model_info.developer` otherwise comes from the registry.

**`eval_library` is the benchmark's own buyer loop**, named `smb-bench`, with
`version: unknown` because the builds differ per model and are listed only in
the dataset card. The benchmark, its three per-kind cuts, the two profit
metrics and this harness are proposed as `draft` entries in the
eval-card-registry.

**`evaluation_id` is keyed on a model's own run rows**
(`smb-bench-v1/smb-bench-buyer__<model label>/runs-<digest prefix>`), not on
the retrieval time and not on the registry id. A re-run gives the same id; a
card-only revision of the dataset, or another model being added to the run
table, does not change it; a change to any of the model's own rows does.

## Coverage

864 run rows become 16 records, one per model, and no model is dropped. 859
rows are scored and enter the figures; the 5 failed rows are reported in their
model's record. A model that cannot be converted (an unknown task, a repeated
run row, a missing row in the models table) is a `SourceRecordFailure` and
the command exits non-zero.

The adapter is registered as not runnable in `catalog.py`: it converts one
pinned revision of a frozen release, and its records were submitted by hand.
