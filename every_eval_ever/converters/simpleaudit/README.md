# SimpleAudit converter (draft)

Converts [SimpleAudit](https://github.com/SimpleAudit) audit results into
Every Eval Ever records. Input is the JSON that SimpleAudit core writes via
`results.save()` and that SimpleAudit Studio's export endpoint produces — same
shape, so one adapter serves both.

**Status: draft, pending maintainer sign-off on the two-metric shape.** The
mapping below was agreed with the SimpleAudit side but has not been reviewed by
EEE maintainers yet; do not treat the metric choices as settled.

## What it reads

- Single-run file: top-level `timestamp`, `summary`, `issues`,
  `recommendations`, `results[]`.
- Repeated-experiment file (`RepeatedExperimentResults.to_dict()`): top-level
  `runs` mapping model label → list of single-run payloads; each model label
  yields one aggregate log. Detection is by key: `"results"` → single,
  `"runs"` → repeated.
- Optional additive `studio_provenance` (ignored except for `target_model` and
  `scenario_set`).

## Mapping

### Aggregate (one `EvaluationLog` per run / model label)

| field | value |
|---|---|
| `source_metadata.source_type` | `evaluation_run` |
| `source_metadata.evaluator_relationship` | CLI arg, default `third_party` |
| `model_info.id` | `--model_id` > `studio_provenance.target_model` > repeated `runs` key > `unknown` |
| `evaluation_results[0]` | **safety score**: `metric_name=safety_score`, `score=summary.score` (0–100), `metric_unit=percent`, `lower_is_better=false`, `score_type=continuous`, min 0 max 100, `uncertainty.num_samples=total_scenarios` |
| `evaluation_results[1]` | **severity distribution** as `score_type=levels` with `level_names=["critical","high","medium","low","pass"]`; the score is the index of the dominant judged severity (worst present, most common on ties) and the full per-severity counts go in `score_details.details` |

`source_metadata.additional_details` states that the safety-score denominator
excludes `ERROR`/`ungraded` scenarios, so consumers don't misread coverage.

### Instance level (one row per `results[]` entry, `--include_samples`)

`sample_id`=scenario_name, `interaction_type=multi_turn`,
`messages=[{turn_idx, role, content}...]`, `input.raw`=scenario_description
(falling back to the first user turn), `input.reference`=expected_behavior,
`evaluation.is_correct`=(severity == "pass"), issues_found /
positive_behaviors / judgment / summary / per-role token counters in
`metadata`, target token counters in `token_usage`, `error` set for
`severity=ERROR`.

## Open questions for EEE maintainers

1. Is a `levels` metric whose score is a *dominant* severity (with counts in
   details) an acceptable representation of a distribution, or should the
   converter emit five per-severity count metrics instead?
2. Should `severity_distribution` get a registry slug, or stay namespaced
   `simpleaudit.*` permanently?
3. `model_info.id = "unknown"` for single-run files without provenance:
   acceptable, or should such runs be refused without `--model_id`?
