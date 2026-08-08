# WILD-raw adapter

Converts **WILD-raw** (`kensho/WILD-raw`, [arXiv:2604.01418](https://arxiv.org/abs/2604.01418))
into Every Eval Ever records.

WILD-raw is **item-level** evaluation data: ~7.5M `(model, item)` rows for 65
models across 27 benchmarks (109,566 items), each row a single item response —
conversation, model answer, target, binary score, token usage, and scorer output.
Kensho ran the evaluations, so it's an `evaluation_run` source with
`evaluator_relationship = third_party`.

## Mapping
- **Aggregate** — one `EvaluationLog` per (model, benchmark). `evaluation_results`
  = the benchmark overall accuracy (`wild.<task>`) plus one per subtask
  (`wild.<task>.<subtask>`); each is a `continuous` `[0,1]` accuracy (mean of the
  binary item scores), with item counts + mean token usage in
  `metric_config.additional_details`. `model_info.id` is the dataset's HF-form model
  id as-is (e.g. `01-ai/Yi-1.5-34B-Chat`).
- **Instances** (`--include-instances`) — the raw per-item rows become an
  instance-level `<uuid>_samples.jsonl` sidecar (single-turn: `input` = the prompt +
  `target`, `output` = the model's full generation, `evaluation` = score/is_correct,
  `token_usage`), referenced by the aggregate's `detailed_evaluation_results`. This
  is the faithful use of the *raw* dataset; it is off by default because it is large.

## Usage

Reads parquet straight from HuggingFace in **bounded record batches** (never the
full ~7GB, and never a whole 500,000-row row group). The output directory must be a
`data/wild` path, because records are published to
`data/wild/<developer>/<model>/<uuid>.json`. Smoke run over the first shard, writing
outside the repo:

```bash
uv run python -m every_eval_ever.adapters.wild.adapter \
  --output-dir /tmp/eee-wild/data/wild --limit-shards 1
uv run python -m every_eval_ever validate '/tmp/eee-wild/data/wild/*/*/*.json*'
```

Filter models / include a capped instance sample:

```bash
uv run python -m every_eval_ever.adapters.wild.adapter \
  --output-dir /tmp/eee-wild-inst/data/wild \
  --limit-shards 1 --models 01-ai/Yi-1.5-34B-Chat \
  --include-instances --max-instances 2000
```

Full run (all 15 shards; heavy — hours + lots of I/O for instances):

```bash
uv run python -m every_eval_ever.adapters.wild.adapter --output-dir data/wild            # aggregates
uv run python -m every_eval_ever.adapters.wild.adapter --output-dir data/wild --include-instances
```

Local parquet instead of fetching. A local file carries no source commit date, so
`--evaluation-timestamp` is required:

```bash
uv run python -m every_eval_ever.adapters.wild.adapter --parquet data-000*.parquet \
  --output-dir /tmp/eee-wild/data/wild --evaluation-timestamp 1780000000.0
```

Record filenames are fresh uuid4s, so a rerun into a populated output directory is
an error rather than a second copy of every record; pass `--replace-existing` to
replace what is there. Replacement goes by identity rather than by directory — only
the (model, benchmark) pairs this run rewrites, and only once their replacements are
published — so a run whose input covers just some of a model's benchmarks (a
`--limit-shards` smoke run, a subset of local `--parquet` files) leaves the rest of
that model's directory alone, and a run that fails partway leaves the previous
refresh whole. Everything is staged and preflighted before any file is created, and a
failure removes whatever the run created.

A row whose `score` is not a usable binary correctness value is left out of the
aggregate (rather than counted as wrong) and named in
`adapter_reports/wild_failures.json`; the command then exits non-zero so a partial
refresh is distinguishable from a complete one.

## Notes
- Timestamps: `retrieved_timestamp` = when this record was built (**now**);
  `evaluation_timestamp` = when WILD ran the eval, proxied by the HF dataset's
  `lastModified` (overrides: `--retrieved-timestamp` / `--evaluation-timestamp`).
  `evaluation_id` is keyed on the evaluation time so reruns are idempotent — there is
  no `now()` fallback, which would give identical reruns different identities.
- Remote runs pin one concrete commit and read it in both passes. If that commit
  cannot be resolved the run stops rather than reading the mutable `main` ref; pass
  `--revision <sha>` to pin a snapshot yourself. `--revision` may name a branch or
  tag, but only while the lookup can resolve it to a commit — if the lookup itself
  fails, only a 40-character SHA is accepted, because a ref that moves between the
  two passes would have them read different data. A local `--parquet` run records a
  local marker instead of a revision it cannot know.
- `eval_library` = `inspect_ai` — the WILD paper states the evals were run with the
  Inspect AI framework; the scorer (`match`) is in `additional_details`.
- `source_data` points at each **benchmark's own dataset repo** (all verified on
  HF; see `WILD_DATASET_REPO` + `AIME_REPO_BY_SUBTASK`) — e.g. `mmlu`→`cais/mmlu`,
  `arc_*`→`allenai/ai2_arc`, `squad`→`rajpurkar/squad`, `finance_fundamentals`→
  `kensho/bizbench`, `pre_flight`→`AirsideLabs/pre-flight-06`, `bbeh`→`BBEH/bbeh`,
  `aime`→`Maxwell-Jia/AIME_2024`+`math-ai/aime25` (per subtask). It is **not**
  `kensho/WILD-raw` — WILD-raw is the *results* source, recorded in `source_metadata`.
- `source_type=evaluation_run`, `evaluator_relationship=third_party`,
  `interaction_type=single_turn` — all confirmed from the paper (short-horizon QA;
  multi-turn/agentic explicitly out of scope). Generation settings (temp/sampling)
  are not documented → omitted.
- **Instances:** `input.raw` = the user/system turns only, so the answer never leaks
  into the input. `output.raw` = the assistant turn(s), i.e. the model's *full*
  generation — a row with no assistant turn gets an empty list rather than the
  scorer's parsed answer. The parsed answer goes in
  `answer_attribution.extracted_value`, with `source` naming where it came from
  (`answer`, or `scores.<scorer>.answer` when that column is empty) and
  `extraction_method` = the real Inspect scorer (the `scores` key: `match`/`choice`/…).
  `sample_hash` = sha256 over canonical JSON of `{"raw", "reference"}` — the shared
  cross-adapter recipe, so the same item joins across adapters.
  Instances attach to the finest-grain result only: every item belongs to exactly one
  subtask, so re-emitting them under the overall would duplicate ~7.5M rows for no
  new information.
- **Aggregation:** per-item binary `score` → `accuracy` (`continuous [0,1]`) with an
  analytic proportion `standard_error` + `num_samples`. Token means cover only the
  rows that carried complete token usage. A benchmark with ≤1 distinct subtask emits only the overall
  `wild.<task>` result (no duplicate leaf); multi-subtask benchmarks emit the
  overall + one `wild.<task>.<subtask>` per subtask (`metric_parameters` marks
  overall vs subtask; `micro` pooling).
- Requires `pyarrow` (parquet reads), declared as the `wild` extra — install it
  first with `uv sync --extra wild` (or `uv sync --all-extras`); a fresh env without
  it fails at import.

## Benchmark canonicalization (eval-card-registry follow-up)

`evaluation_name` is `wild.<task>[.<subtask>]` and `source_data` carries the dataset
repo; canonicalizing the benchmark id is the registry's job. Of WILD's 27
benchmarks, 20 resolve today; the rest are follow-ups for the registry:
- **Add aliases** to existing canonicals: `arc_easy`, `arc_challenge` →
  `ai2-reasoning-challenge-arc` (AI2 ARC, *not* ARC-AGI); `race_h` → `race`.
- **New canonicals** (genuinely absent from the registry; all sources now
  resolved & public): `squad` (`rajpurkar/squad`), `paws`
  (`google-research-datasets/paws`), `chembench` (`jablonkagroup/ChemBench`),
  `finance_fundamentals` (`kensho/bizbench` — a curated subset), `pre_flight`
  (`AirsideLabs/pre-flight-06`, an Inspect Evals task). Also `bbeh` (`BBEH/bbeh`)
  and `aime` (`Maxwell-Jia/AIME_2024` / `math-ai/aime25`) resolve today but their
  dataset repos are worth recording.
- The AI2-ARC canonical currently has no `dataset_repo`; `allenai/ai2_arc` should be
  added, and easy/challenge are collapsed into one canonical upstream.
- Aggregation reads only the small columns (fast); `--include-instances` reads the
  full rows and is the expensive path — use `--limit-shards` / `--max-instances`
  for smoke runs.
