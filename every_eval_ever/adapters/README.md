# Adapters

One-off adapter scripts that fetch leaderboard data from external sources and convert it to the Every Eval Ever schema. These are run manually, not via the main CLI.

## Writing a new adapter

Start from the `eee-dataset-conversion` agent skill —
[`.agents/skills/eee-dataset-conversion/SKILL.md`](../../.agents/skills/eee-dataset-conversion/SKILL.md).
It carries the field semantics, the merge-gate checks (`reference/datastore-gate.md`),
runnable templates, and the datastore submission mechanics. `tests/test_skill_conversion.py`
re-validates those templates against the live validator, so they stay current.

## Usage

Each adapter is run with `uv run python -m every_eval_ever.adapters.<name>.adapter`.

## The automation contract

[`catalog.py`](catalog.py) declares which adapters the daily ingestion run may
execute, the datastore collections each may write, the exact argv that keeps its
output out of the checkout, and how long it may take. Every adapter package must
appear there or in `LEGACY_ADAPTERS`, and `tests/test_adapter_catalog.py` checks
each entry against the adapter's own parser, so a renamed flag fails a test rather
than a scheduled run. It is called the catalog, not the registry, because "the
registry" in this project is [`eval-card-registry`](https://github.com/evaleval/eval-card-registry).

An adapter that automation runs must therefore:

- expose `parse_args(argv: list[str] | None = None)` at module level;
- accept `--output-dir`, and write **only** under it;
- write records at `<output>/…/<developer>/<model>/{uuid4}.json`; the runner refuses
  anything else, including a collection the catalog did not declare;
- account for dropped source rows through `SourceConversionResult` +
  `save_failure_report` + a non-zero exit, which is what lets a partial refresh be
  told apart from a crash.

`bfcl`, `cocoabench`, `paperswithcode_drugbank`, and `sciarena` are registered as
`runnable=False`: they need local inputs and have no live fetch path.

## Raw source snapshots

[`helpers/raw_capture.py`](../helpers/raw_capture.py) keeps the bytes an adapter
converted, so a later correction can be checked against the input. It is inert unless
`EEE_RAW_CAPTURE_DIR` is set, which only the cron does, so a manual run is unchanged.

Adapters that fetch through `helpers.fetch.fetch_json` / `fetch_csv` are captured
without any adapter code. An adapter with its own HTTP call site calls
`raw_capture.record(...)` there. A source already addressable at a revision, such as
a Hugging Face dataset or a git clone, records a pointer with
`raw_capture.record_hf_dataset(...)` / `record_git_checkout(...)` rather than
re-hosting bytes that are already durably stored.

## Adapters

| Adapter | Data Source | Description |
|---------|-------------|-------------|
| `arc_agi` | ARC Prize leaderboard JSON | Fetches the JSON files behind arcprize.org/leaderboard, maps models to developers via the provider table, and merges canonical model aliases. |
| `artificial_analysis` | Artificial Analysis LLM API | Converts Artificial Analysis LLM benchmark, pricing, and performance results into `data/artificial-analysis-llms/`. |
| `vals_ai` | Vals.ai benchmark leaderboards | Scrapes Vals.ai benchmark pages and converts their embedded leaderboard results into `data/vals-ai/`. |
| `bfcl` | BFCL leaderboard CSV | Converts BFCL leaderboard data with per-metric evaluation names and bounded continuous scores. |
| `sciarena` | SciArena leaderboard API | Converts SciArena leaderboard results. |
| `global_mmlu_lite` | Kaggle API | Fetches Global MMLU Lite leaderboard results from Kaggle. |
| `hfopenllm_v2` | HuggingFace Spaces API | Fetches the Open LLM Leaderboard v2 (4576+ models). The leaderboard is no longer maintained upstream, so this converts a frozen archive and is not scheduled. |
| `helm` | HELM leaderboard | Converts HELM leaderboard data. Supports `--leaderboard_name` for Capabilities/Lite/Classic/Instruct/MMLU. |
| `llm_stats` | LLM Stats API | Converts LLM Stats model, benchmark, and score API data into `data/llm-stats/`. |
| `mercor_eval` | Mercor Evaluation Exports API | Fetches authenticated Mercor benchmark leaderboards and writes aggregate EEE records. |
| `mt_bench` | LMSYS / FastChat | Converts MT-Bench GPT-4 single-answer judgments into `data/mt-bench/`. Emits overall, turn-1, and turn-2 means per model. |
| `open_medical_llm` | HuggingFace (`openlifescienceai/results`) | Converts the Open Medical-LLM Leaderboard's lm-evaluation-harness results into `data/open-medical-llm/`. One record per model, one result per medical benchmark (9). See [`open_medical_llm/README.md`](open_medical_llm/README.md). |
| `openeval` | HuggingFace | Converts OpenEval response scores from `human-centered-eval/OpenEval` into `data/openeval/`; pass `--include-instances` to also write `*_samples.jsonl` sidecars. |
| `rewardbench` | HuggingFace | Fetches RewardBench v1 (CSV) and RewardBench v2 (JSON) leaderboard data. |
| `terminal_bench_2` | tbench.ai | Fetches Terminal-Bench 2.0 agentic coding benchmark results. |
| `hle` | Scale SEAL leaderboard | Converts the Scale SEAL Humanity's Last Exam leaderboard into `data/hle/`. Emits per-model accuracy (with 95% CI) and calibration error. |
| `mmlu_pro` | TIGER-Lab leaderboard CSV | Converts the MMLU-Pro leaderboard (`TIGER-Lab/mmlu_pro_leaderboard_submission`) into `data/mmlu-pro/`. Emits per-model overall + 14 per-subject accuracies. |
| `paperswithcode_drugbank` | Local Papers with Code PostgreSQL dump + reviewed YAML manifest | Manually converts only DrugBank score cells with reviewed model, metric-scale, split, and protocol semantics. Writes `data/paperswithcode-drugbank/`. The same cells also appear in `data/paperswithcode/` from the general adapter over the same dump, unreviewed; both records carry the PwC `pwc_evaluation_id` so the pair is joinable. |
| `lexam` | LEXam project website | Converts the LEXam legal-reasoning leaderboard (open-question judge scores + 4-choice MCQ accuracy) into `data/lexam/`. |
| `vectara_hallucination_leaderboard` | HuggingFace (`vectara/results`) | Converts the Vectara Hallucination Leaderboard result files, pinned to a source commit, into `data/vectara-hallucination-leaderboard/`. Emits 4 aggregate metrics plus per-category and per-text-complexity breakdowns (40 scores per model). |
| `paperswithcode` | Papers with Code PostgreSQL dumps | Converts PwC leaderboard entries into `data/paperswithcode/`. Metric bounds and direction are resolved against a vendored eval-card-registry snapshot; unknown metrics fail the run rather than getting invented bounds. Needs the `paperswithcode` extra. |

### Papers with Code DrugBank

To run this manual adapter, provide a local custom-format PostgreSQL dump and a
reviewed, non-empty YAML qualification manifest (schema version 2). It is not
scheduled because
neither input has a live fetch path, and the repository does not include a
production manifest. The manifest must supply reviewed canonical model,
metric, and benchmark IDs plus the registry commit against which they were
reviewed. The adapter does not resolve or invent identities, and conversion is
atomic: an anchor, hash, or schema mismatch aborts before any output is written.
For split comparisons, each manifest entry must declare `transductive`,
`inductive-s1`, or `inductive-s2`, with matching manifest-declared overlap
labels. A
canonical benchmark ID cannot be reused for conflicting protocols anywhere in
the manifest, so the adapter cannot emit conflicting split scores under one
benchmark key. The adapter preserves the scores and protocol evidence supplied
by the reviewed manifest; it does not independently verify cited source content
or train/test membership from the aggregate dump, and it does not compute a
performance delta between splits.

```bash
uv run --extra paperswithcode python -m \
  every_eval_ever.adapters.paperswithcode_drugbank.adapter \
  --dump /path/to/paperswithcode.dump \
  --overlay /path/to/reviewed-drugbank.yaml \
  --output-dir /tmp/paperswithcode-drugbank/data/paperswithcode-drugbank
```

As of 2026-08-19, the archived Papers with Code DrugBank URL redirects away
from its dataset record. Generated logs retain that URL as raw provenance,
link `source_data` to DrugBank, and record the official Papers with Code archive
in source metadata.

### Mercor Evaluation Exports

Set the API key in the environment and run the adapter:

```bash
export MERCOR_EVAL_API_EVALEVAL_KEY="<your-key>"
uv run python -m every_eval_ever.adapters.mercor_eval.adapter
```

For a credential-free offline smoke run:

```bash
uv run python -m every_eval_ever.adapters.mercor_eval.adapter \
  --input-json tests/data/mercor_eval/api_payload.json \
  --output-dir /tmp/mercor-eval-offline
```

The adapter exports aggregate leaderboard metrics only. Mercor's criterion
results do not include the task input, model output, messages, or answer
attribution required by the EEE instance-level schema.
Records are generated under benchmark-specific datastore directories, for
example `data/apex-agents/<developer>/<model>/<uuid>.json`. Generated records
are intended for the Hugging Face datastore submission, not the GitHub adapter
PR.

### LEXam

```bash
uv run python -m every_eval_ever.adapters.lexam.adapter --output-dir data
```

One record per model, with one result per published leaderboard column:

| Metric | Evaluation | Scale | Scope |
|---|---|---|---|
| Open Question Judge Score | `lexam.open_question` | `[0,100]` | test split, n=2,541, scored by a pointwise-minimum ensemble of three judges |
| Multiple-Choice Accuracy | `lexam.mcq_4_choices` | `[0,1]` | n=1,655; the 4-choice config only, not the 8/16/32-choice ones |

The site prints both columns as percentages. Each is emitted on the scale of
its registry metric, with the published percentage kept in
`score_details.details`.

Model ids, metric ids, bounds and direction come from the eval-card-registry
through `registry_snapshot.json`, which vendors the entities this adapter emits
and is pinned to the registry revision they came from. The tests fail if an
emitted value drifts from the pin, so regenerate it after a registry change:

```bash
uv run python -m every_eval_ever.adapters.lexam.refresh_registry_snapshot \
    --registry /path/to/eval-card-registry
```

Add `--check` to test the pin without writing: it exits non-zero and names both
revisions. Metric `review_status` is read from the snapshot, so a metric
promoted upstream needs a refresh rather than a code change.

Inference settings, serving and standard errors are not on the leaderboard;
they come from the paper (arXiv:2505.12864v7 §3.3, appendix F, Tables 1 and 10)
and from LEXam's own runner, `litellm_eval.py`, which names the served model for
15 of the 36 rows. Each record cites the source it used, and a standard error is
attached only while the scraped score still equals the published one.

Submission follows the datastore mechanics in the conversion skill. One
adapter-specific caveat: record filenames are fresh uuids per run, so a second
`upload_folder` onto an open submission PR adds another copy of every model.
Update a submission by deleting `data/lexam/` and adding the new records in a
single `create_commit`.

### Papers with Code

The source is a nightly PostgreSQL backup of the PwC database, published to the
HF bucket `huggingface/paperswithcode-backups` under `postgres/*.dump`
(`pg_dump -Fc`, ~180–210 MB each). Dumps are read with
[`pgdumplib`](https://pypi.org/project/pgdumplib/), so no PostgreSQL server or
`pg_restore` is needed — install the extra:

```bash
uv sync --extra paperswithcode
```

Auto-downloading the newest dump additionally needs `huggingface_hub>=1.0` for
the bucket API, above the range this repo pins. The `--dump` path (a dump
already on disk) has no such requirement, and the import is lazy, so only
auto-download fails and only when it is actually used.

```bash
# a dump already on disk, two leaderboards, no network
uv run python -m every_eval_ever.adapters.paperswithcode.adapter \
  --dump /tmp/pwc-raw/paperswithcode_hf_20260716_031511.dump \
  --dataset-slug eth3d-relative --dataset-slug re10k-2-view \
  --output-dir /tmp/eee-pwc

# download the newest dump and convert everything (large)
uv run python -m every_eval_ever.adapters.paperswithcode.adapter \
  --all --output-dir data/paperswithcode
```

PwC re-reports numbers rather than running models, so `source_type` is
`documentation`, there is no per-item data and no `_samples.jsonl`. One record
per canonical model id; each result is one (evaluation row × metric) pair. A
re-run over the same dump is byte-stable — `retrieved_timestamp` and
`evaluation_id` are keyed on the dump date, never on wall-clock time. Because
record filenames are fresh uuids, a re-run replaces the output directory's
contents: the new batch is validated and written first, and only then are the
previous run's records removed, so a failed run leaves that run's output intact.

`continuous` metrics need a defined `min_score`/`max_score`, and PwC does not
publish them. They come from the eval-card-registry's canonical metric entries,
vendored in `registry_snapshot.json` and pinned to the registry revision they
came from, so resolution at convert time is a static lookup. A metric that is
absent from the snapshot, or whose name maps to more than one canonical id,
**fails the run** by default and is named in the report; `--allow-unresolved`
emits it with observed-range bounds flagged as such. Reported values are mapped
onto the canonical scale per `(metric, dataset)` leaderboard rather than per
score, so an all-percent board for a `[0,1]` metric is rescaled as a group and a
lone out-of-range value is flagged instead of silently divided. `metric_unit`
names that canonical scale rather than the one PwC declared, so it stays true
after a rescale; the source declaration is kept as `pwc_scale`. A score the
group decision cannot place inside the declared bounds is **not published** —
that cell is omitted and listed in the failure report, since the bounds a record
declares have to contain its score.

Every run prints a full imperfection report — unresolved metrics, unknown
directions, scale anomalies — to stderr. The mode decides only whether to abort
before publishing: strict (the default) exits non-zero before writing anything,
`--allow-unresolved` tolerates only the unresolved class, and `--best-effort`
writes everything representable with each imperfection flagged. No mode ships an
out-of-range score, so a run that dropped one still exits non-zero.

Registering a bound for a new metric is the one part of this adapter that needs
human judgment; [`METRIC_MAINTENANCE.md`](paperswithcode/METRIC_MAINTENANCE.md)
is the procedure, including the observed-range cross-check that keeps a cited
bound honest. Refresh the snapshot after any registry change:

```bash
uv run python -m every_eval_ever.adapters.paperswithcode.refresh_registry_snapshot \
    --seed /path/to/eval-card-registry/seed/metrics.yaml
```

Model ids use the HF `developer/model` form when `hf_model_url` is present.
Effort/mode tiers in PwC model names (`GPT-5.5 Pro (xhigh)`) are kept verbatim;
collapsing them and aliasing the ids belongs in the registry, not here.

## Notes

- These are one-off scripts, not integrated into the main CLI.
- They require network access to fetch live leaderboard data.
- Some adapters (e.g. `rewardbench`, `helm`) may take several minutes to complete due to the number of models.
- Run `uv run python -m every_eval_ever.adapters.<name>.adapter --help` for adapter-specific options.
- Generated adapter outputs under `data/<source>/` and saved raw payloads are
  generated artifacts. Prefer temporary output paths for smoke runs unless a
  data refresh is intentionally part of the change.

### Legacy integrations

`livecodebenchpro` is retained for historical and offline use, but its
upstream source is no longer usable for an active refresh. It is excluded
from active-adapter migration and compliance requirements. Deterministic
offline tests for its existing behavior may remain in the test suite.

`arc_agi` left this list on 2026-08-12: its old endpoint
(`/media/data/leaderboard/evaluations.json`) is gone, but the leaderboard
itself is live, rendered from JSON files under
`https://arcprize.org/media/data/`. The adapter now fetches those
(evaluations, models, providers, datasets), takes each model's developer
from the provider table instead of name heuristics, and is scheduled daily.

`mercor_eval` is paused: its Exports API is broken upstream as of
2026-08-12, so the catalog marks it `runnable=False` until Mercor serves
data again. The adapter itself is healthy and still runs by hand; it exits
`75` on an unreachable API and `1` on a rejected key.

`helm_*` and `rewardbench` are paused for staleness rather than breakage:
HELM's leaderboards are effectively static and RewardBench has not updated
in a while, so a weekly refresh refetches unchanged data. Both sources
still serve, both adapters still run by hand, and re-enabling either is one
`runnable` flip in the catalog.

### Partial conversions and provenance

An adapter may encounter a source row or metric that cannot be represented as
a valid EEE record—for example, a missing model identity or a non-numeric
score. It still writes every valid record. It also writes a strict JSON
provenance report under `adapter_reports/`, outside `data/`, with the source
reference, raw source fragment when available, and reason for each omission.
The command then exits non-zero so automation can distinguish a complete
refresh from a partial one.

Intentional non-evaluation rows, such as a published random baseline, are
recorded as exclusions in the same report but do not make the command fail.
The report is not an `EvaluationLog` and must not be passed to the validator.

### Vals.ai

Run a live smoke export from the repository root, writing generated output
outside the repo:

```bash
uv run python -m every_eval_ever.adapters.vals_ai.adapter \
  --output-dir /tmp/eee-vals-ai/data/vals-ai
```

To intentionally prepare a data refresh, use `--output-dir data/vals-ai` and
validate the result before deciding whether to include generated files.

For smaller smoke runs, fetch one benchmark:

```bash
uv run python -m every_eval_ever.adapters.vals_ai.adapter \
  --benchmark finance_agent \
  --output-dir /tmp/eee-vals-ai-smoke/data/vals-ai \
  --save-raw-json /tmp/eee-vals-ai-raw.json
```

Replay a saved normalized payload without hitting the network:

```bash
uv run python -m every_eval_ever.adapters.vals_ai.adapter \
  --input-json /tmp/eee-vals-ai-raw.json \
  --output-dir /tmp/eee-vals-ai-replay/data/vals-ai
```

Validate generated records with:

```bash
uv run python -m every_eval_ever validate \
  '/tmp/eee-vals-ai-smoke/data/vals-ai/*/*/*.json*'
```

### Vectara Hallucination Leaderboard

The adapter enumerates every per-model result file in `vectara/results` at the
pinned `SOURCE_COMMIT` and emits one record per model. Run a live export
outside the repository:

```bash
uv run python -m every_eval_ever.adapters.vectara_hallucination_leaderboard.adapter \
  --output-dir /tmp/eee-vectara/data/vectara-hallucination-leaderboard \
  --save-raw-json /tmp/eee-vectara-raw.json
```

Replay the saved snapshot without hitting the network:

```bash
uv run python -m every_eval_ever.adapters.vectara_hallucination_leaderboard.adapter \
  --input-json /tmp/eee-vectara-raw.json \
  --output-dir /tmp/eee-vectara-replay/data/vectara-hallucination-leaderboard
```

Bump `SOURCE_COMMIT` to pick up a newer leaderboard run. The evaluated corpus
is private, so the log records the public result file as provenance rather than
a redistributable dataset. The pinned files record no serving platform, so
`deployment_type` stays `unknown`; `model_availability` is derived from the
source's `accessibility` annotation.

Provenance that is constant for a run — the source file, commit, resolve URL,
scoring model and temperature policy — lives once in `source_metadata`. Each of
the 40 results carries only what varies, because repeating the constants on
every result doubled the size of each record. That invariant is pinned by
`test_constant_provenance_is_not_repeated_per_result`.
