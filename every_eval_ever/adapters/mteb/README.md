# MTEB adapter

Converts [MTEB](https://github.com/embeddings-benchmark/mteb) embedding-model
results into `data/mteb/`.

```sh
uv run python -m every_eval_ever.adapters.mteb.adapter --output-dir data/mteb
uv run python -m every_eval_ever validate 'data/mteb/*/*/*.json'
```

## Source

[`embeddings-benchmark/results`](https://github.com/embeddings-benchmark/results),
the dump MTEB's own leaderboard reads, pinned to a commit (`--ref` overrides):

```
results/<org>__<model>/<model_revision>/<Task>.json
results/<org>__<model>/<model_revision>/model_meta.json
```

Not the [`mteb/results`](https://huggingface.co/datasets/mteb/results) parquet
mirror. The mirror flattens each score to one `(model, task, split, subset,
language)` row and drops the harness version, the evaluation dataset revision
and every named sub-metric, all of which this schema has a field for. The mirror
is still useful as an independent check: every score this adapter published in a
5-model smoke run was compared against it, 4,904 of 4,904 exact, none missing.

The whole repository is fetched once as a tarball rather than one request per
file. The dump holds hundreds of thousands of small JSONs, so a per-file fetch
would neither finish inside a scheduled run's budget nor leave a manifest
anyone could check. Raw capture records the resolved commit as a git pointer,
not a copy of the bytes, since the source is already durably addressable there.

## Scope

The two headline suites, `MTEB(eng, v2)` and `MTEB(Multilingual, v2)`: 151
tasks, the numbers people cite.

The full dump is ~8.8M scores, and one task is 85% of them
(`FloresBitextMining`, one cell per ordered language pair). Publishing all of it
is a re-hosting decision rather than a coverage one, so it is deliberately not
what this adapter does.

### The per-language fan-out, and `--max-subsets-per-task`

Even inside the two scoped suites the same shape appears. Measured across them:

| tasks | scores |
|---|---|
| 141 tasks with <= 150 subsets | 573 |
| 10 tasks with > 150 subsets | 46,495 |

Median subsets per task is 1. With the fan-out tasks in, one model's record is
about 70MB and 88% of it is `FloresBitextMining` alone.

MTEB's leaderboard shows a mean over those cells. **The dump does not contain
that mean**, and an adapter computing it would be inventing a number the source
never stated, so the adapter does not. Instead a task reporting more than
`--max-subsets-per-task` cells (default 150) is left out and reported as an
exclusion naming its exact count. The default sits in the observed gap between
112 and 197 subsets, so it is a boundary in the data rather than a round number.
Pass `--max-subsets-per-task 0` to include everything.

This is the open question worth a maintainer's view: these tasks need either a
task-level aggregation convention or a collection of their own.

## Record shape

One `EvaluationLog` per `(model, model_revision)`; one `EvaluationResult` per
`(task, split, hf_subset)`, which is the grain the source reports at.

- `evaluation_id` is `mteb/<model_repo>/<model_revision>`, keyed on the dump's
  own identity so re-ingest is idempotent. The dump records how long each task
  took but **never when it ran**, so `evaluation_timestamp` is left unset rather
  than filled with a guess.
- `eval_library` carries the real `mteb_version` that wrote the files, and
  `source_metadata.additional_details.mteb_versions_in_run` lists every version
  present when a model's task files were not all written by one.
- `source_data` pins each task's own HF dataset repo **and** `dataset_revision`.
- `model_info.additional_details.model_availability` comes from each model's
  `model_meta.open_weights`; `deployment_type` is `self_deployed` for all of
  them, which is what running an open benchmark against local weights means.
- `metric_config.additional_details.model_trained_on_this_dataset` reports
  MTEB's own `training_datasets` overlap for the task, so a score that is not a
  held-out measurement says so. This is the source's statement, not an inference.

## Metrics

MTEB declares a `main_score` per task, and it varies **within** a task type:
Retrieval alone uses `ndcg_at_10`, `ndcg_at_100`, `recall_at_1`, `mrr_at_5` and
more across its tasks. A task-type-level table would mislabel hundreds of tasks,
so the per-task `main_score` is vendored in `task_metadata.json` and the adapter
re-checks it numerically against the named sub-metric in every score it converts.
A row where they disagree is failed, not published.

### Registry ids

Four are already eval-card-registry canonicals and are used directly:
`accuracy`, `f1`, `ndcg-at-10`, `recall-at-10`, plus `map` and `mrr` for the
legacy spellings below.

The rest have no canonical id yet. They are published under a proposed id with
`metric_config.additional_details.metric_id_status = "proposed"`, so a consumer
joining on `metric_id` can tell a canonical from one awaiting review. **These are
proposed for registration** rather than invented silently, because a made-up
global id fragments the one join the datastore exists for:

| MTEB `main_score` | proposed id | bounds |
|---|---|---|
| `v_measure` | `v-measure` | 0 to 1 |
| `max_ap` | `max-ap` | 0 to 1 |
| `cosine_spearman` | `cosine-spearman` | -1 to 1 |
| `p-MRR` | `p-mrr` | -1 to 1 |
| `map_at_1000` | `map-at-1000` | 0 to 1 |
| `mrr_at_10` | `mrr-at-10` | 0 to 1 |
| `ndcg_at_1` | `ndcg-at-1` | 0 to 1 |
| `max_over_subqueries_map_at_1000` | `mteb.max-over-subqueries-map-at-1000` | 0 to 1 |

The benchmark ids `mteb`, `mteb-eng-v2` and `mteb-multilingual-v2` are also
absent from the registry; 150 of the 151 scoped task names are too.

Note that the registry's existing `mteb-score` and `mmteb-score` entries declare
bounds of **0 to 100**, while this dump stores 0 to 1. Neither is used here for
that reason.

### Legacy metric spellings

Older `mteb` wrote reranking metrics without the cutoff suffix. The two
spellings **never co-occur in one entry**, so there is no run in which they can
be checked against each other, and the adapter does not assert
`map == map_at_1000` on that absence. A row whose declared name is missing is
published under the name its own file uses, with the declared name recorded in
`metric_config.additional_details.mteb_declared_main_score`. Both legacy
spellings are registry canonicals, so this costs no join.

| declared now | older files write | handling |
|---|---|---|
| `map_at_1000` | `map` | published as `map` |
| `mrr_at_10` | `mrr` | published as `mrr` |
| `max_over_subqueries_map_at_1000` | `map` | **failed, not published** |

The last one is deliberate: older files spell it `map` too, but that is also the
plain metric's name, and publishing a multi-subquery protocol's score as
ordinary MAP would merge two different quantities.

## Refreshing the task snapshot

`task_metadata.json` is vendored so the adapter needs neither the `mteb` package
(a heavy dependency) nor a network call to name a metric. Regenerate it with:

```sh
uv run --with mteb --no-project python \
  every_eval_ever/adapters/mteb/refresh_task_metadata.py
```

`tests/test_mteb_adapter.py` fails if a refresh introduces a `main_score` the
adapter has no bounds or id for, so the snapshot cannot drift ahead of the
metric table silently.
