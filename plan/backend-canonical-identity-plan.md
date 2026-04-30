# Backend Canonical Identity Plan (Data Audit + Actions)

## Snapshot audited

- **Code repo (`~/every_eval_ever`)** updated to `aa966f7cf` (origin/main).
- **Datastore (`evaleval/EEE_datastore`)** updated to `5edc7b9`.
- Audit scope: all aggregate JSON files under `data/**` (`6448` files, `49659` evaluation results).

## What is happening (evidence from latest data)

1. **Metric identity is mostly missing in production data**
   - `metric_config.metric_name` missing in **37071 / 49659** results.
   - `metric_config.metric_id` missing in **37071 / 49659** results.
   - This is concentrated in major configs: `hfopenllm_v2`, `helm_*`, `reward-bench`, `global-mmlu-lite`, `fibble_arena`, `wordle_arena`, `terminal-bench-2.0`, etc.
   - Concrete live examples:
     - `global-mmlu-lite/xai_grok-3-mini/1773936496.366405` has **19** results and **0 / 19** populated `metric_name` or `metric_id` fields; the only explicit labels are `evaluation_name` values such as `Global MMLU Lite`, `Culturally Sensitive`, `Arabic`, `English`, etc.
     - `wordle_arena/qwen/qwen3-8b/1776347262.820056` has **3** results and **0 / 3** populated `metric_name` or `metric_id` fields.
   - Backend implication: cannot reliably group/compare metrics without string parsing heuristics.

2. **`evaluation_name` is frequently carrying metric semantics**
   - **615** results have metric-like `evaluation_name`.
   - Confirmed examples:
     - `apex-agents`: `evaluation_name: "Overall Pass@1"` (metric semantics in eval field).
     - `bfcl`: `evaluation_name: "bfcl.memory.accuracy"` while metric fields are also populated (eval and metric axes collapsed).
     - `theory_of_mind`: `evaluation_name: "accuracy on theory_of_mind for scorer ..."` (legacy converter style).
     - `wordle_arena/qwen/qwen3-8b/1776347262.820056`: `evaluation_name` values are `wordle_arena_win_rate`, `wordle_arena_avg_attempts`, and `wordle_arena_avg_latency_ms`, so the eval axis is fully metric-shaped.
     - `global-mmlu-lite/xai_grok-3-mini/1773936496.366405`: `evaluation_name` is used for suite/slice labels (`Global MMLU Lite`, `Arabic`, `French`, etc.) while the implicit metric remains unstated, so eval and metric identity are still entangled even though the names are not metric-like.
   - Backend implication: card grouping by evaluation name produces metric-shaped “benchmarks”.

3. **`score_details.details` is overloaded as a nested telemetry dump**
   - Found **52208** JSON-encoded values stored as strings inside `score_details.details`.
   - HELM MMLU example (`Abstract Algebra` / `College Physics`) contains many cross-subject entries (e.g., College Chemistry/Biology stats inside College Physics row), mixing eval slice + telemetry dimensions.
   - Backend implication: requires expensive post-parsing and risks accidental interpretation as benchmark/metric labels.

4. **Benchmark/evaluation_id naming is not consistently aligned**
   - **257** files where `evaluation_id` prefix does not match top-level folder benchmark codename.
   - Main cases:
     - `reward-bench` folder vs `evaluation_id` prefix `reward-bench-2`.
     - `tau-bench-2_{domain}` and `appworld_test_normal` folders vs `evaluation_id` prefixes with hierarchical paths (`tau-bench-2/...`, `appworld/...`).
   - Backend implication: any logic keyed on only one naming source (folder or `evaluation_id`) drifts.

5. **Eval library naming is not standardized**
   - **16 distinct `eval_library.name` values** including mixed casing and source-specific names (`lm-evaluation-harness`, `BFCL`, `Artificial Analysis`, `ARC Prize leaderboard`, `harbor`, `unknown`, etc.).
   - Backend implication: harness-level analytics and joins need alias normalization today.

6. **Fibble family note**
   - Current snapshot no longer has `fibble1_arena`, `fibble2_arena` top-level folders; it is consolidated as `fibble_arena`.
   - But fibble still encodes both slice and metric in `evaluation_name` (`fibble_arena_1lie_win_rate`, `...avg_attempts`), with missing metric IDs.

7. **`detailed_evaluation_results` coverage can be metric-selective inside one aggregate run**
   - Current live example: `wordle_arena/qwen/qwen3-8b/1776347262.820056` exposes **3** aggregate metrics (`win_rate`, `avg_attempts`, `avg_latency_ms`) and links one sample file with **35** rows.
   - All **35 / 35** current sample rows in `9a357c44-1c36-43dc-a764-de1f3e204fe1_samples.jsonl` carry `evaluation_name = "wordle_arena_win_rate"`.
   - The same aggregate currently declares `detailed_evaluation_results.total_rows = 27`, so file-link metadata and actual sample-row counts can already disagree in production.
   - Backend implication: a linked sample file does not imply run-wide instance coverage. Aggregate-to-instance linkage must remain metric-scoped, and instance-availability badges should be computed per metric or per eval-summary node, not per run.

## Backend-centric recommendations (proposed)

1. **Enforce canonical identity at ingestion (hard)**
   - Persist canonical tuple (backend-owned):  
     `(run_id, model_id, benchmark_family_id, eval_slice_id, metric_id, harness_id, result_index)`.
   - Keep raw fields in parallel (`raw_evaluation_name`, `raw_metric_description`, etc.) for audit/debug.

2. **Add registry-backed resolution with confidence**
   - Resolve benchmark/eval-slice/metric/harness via registry aliases (`exact`, `normalized`, `fuzzy`, `manual`).
   - Store `strategy`, `confidence`, `review_status`; quarantine low-confidence rows from card generation.

3. **Add semantic validation gates in ingestion CI**
   - Reject or flag:
     - metric-like `evaluation_name` without explicit metric identity,
     - `evaluation_name == metric_name` collisions,
     - benchmark-family naming drift (`folder` vs `evaluation_id` inconsistencies).
     - linked sample files whose rows cover only a strict subset of aggregate metrics without explicit metric-scoped coverage metadata.
     - linked sample files whose observed row count disagrees with declared `detailed_evaluation_results.total_rows`.
   - Keep structural schema validation, but add these semantic checks as a second gate.

4. **Phase-in stricter schema usage for metrics**
   - Immediate: warn-only for missing `metric_name`/`metric_id`.
   - Next: soft fail in bot with override.
   - Final: hard fail (for new submissions) unless `metric_name` + `metric_id` present.

5. **Serve frontend from canonical IDs only**
   - Frontend card grouping/filtering must use canonical IDs, never raw labels.
   - Raw labels are display metadata only.
   - Instance availability must be attached to canonical metric/eval-summary IDs, not inferred from the existence of any `detailed_evaluation_results` file on the parent run.
   - This prevents recurring “benchmark cards that are actually metrics”.

## Should we fix adapters and regenerate data?

**Short answer: yes, but only for adapter-owned benchmark families.**

### Good candidates for adapter-fix + regenerate

Adapters exist in `utils/` for:
- `hfopenllm_v2`
- `helm` (`helm_lite`, `helm_mmlu`, `helm_capabilities`, `helm_classic`, `helm_instruct`)
- `rewardbench`
- `global-mmlu-lite`
- `terminal_bench_2`
- `exgentic` (used by tau/appworld/swe/browsecompplus in this dataset)

These are high-leverage because they account for a large share of missing metric identity.

### Not fully solved by adapter regeneration alone

Several benchmark families in data are not obviously sourced from current `utils/` adapters (or are manually/externally produced), including examples like:
- `apex-agents`, `apex-v1`, `bfcl`, `artificial-analysis-llms`, `arc-agi`, `sciarena`, `fibble_arena`, `wordle_arena`, `ace`, `la_leaderboard`.

For these, you need a **backfill canonicalization migration** + submission template updates, not only adapter patches.

### Practical plan

1. Patch adapters to emit explicit `metric_name` + `metric_id` and metric-free `evaluation_name`.
2. Regenerate adapter-owned families in a controlled replay branch.
3. Run one-time migration for non-adapter/manual families.
4. Turn on semantic gating and canonical-ID-only serving.
