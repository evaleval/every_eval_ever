# Evidence for model deployment metadata

Use this reference to decide the two model axes for the exact evaluated model and run.
The live schema and validator define the allowed strings; this reference defines the
evidence standard.

## Classify what is missing

Inspect the raw JSON, not only a parsed `EvaluationLog`. The library can materialize
compatibility placeholders and make an omitted key look like an explicit `unknown`.

- If `model_info.additional_details` is absent, null, or not an object, treat both axes
  as uninvestigated. Research them, then create an object without losing other model
  metadata.
- If the object exists but one or both keys are absent, research only what is
  unestablished and preserve every unrelated key.
- If a key explicitly contains `unknown`, verify whether the contributor documented an
  evidence sweep. Otherwise treat it as an unchecked placeholder.
- If the relevant primary sources genuinely omit the fact, keep `unknown` and list the
  sources checked. Missing source evidence is a valid result; missing record metadata
  is not evidence.

## Keep the axes independent

`deployment_type` records who controlled the inference deployment used for this
evaluation. A third-party/provider endpoint is externally managed; a runtime operated
by the evaluator is self-deployed.

`model_availability` records whether the exact evaluated model's weights are available.
Downloadable weights are open weights even when their license is restrictive. API or
product access without downloadable weights is closed weights.

Common valid combinations include hosted open-weight models and self-deployed
open-weight models. Weight availability alone never proves how this evaluation served
the model.

## Evidence priority

Prefer evidence closest to the submitted run:

1. Raw result metadata, run configuration, logs, or an exact inference endpoint.
2. The generating adapter and its pinned source snapshot.
3. The evaluator's methods, paper appendix, or source repository.
4. The exact model card or official model release.
5. Official provider API documentation or a dated release announcement.

A model card usually establishes availability, not deployment. A leaderboard row or
developer folder usually establishes neither. When sources conflict, prefer the
run-level source, preserve the conflict in the decision log, and lower confidence.

## Decision signals

Use these as evidence tests, not name-based mappings:

| Observed evidence | Supported decision |
|---|---|
| Run config names a provider or third-party inference API used for generation | Externally managed deployment |
| Run config or methods state that the evaluator loaded the checkpoint with vLLM, TGI, Transformers, or its own runtime | Self-deployed deployment |
| Exact checkpoint has an accessible weight repository or official downloadable release | Open weights |
| Official source offers only hosted/API access for the exact model and no weights | Closed weights |
| Only the model family, developer, path, or an old enum value is known | No decision; continue research or use `unknown` |

Do not mechanically translate stale vocabulary until checking the old validator's
meaning and the submitted run. Lexical similarity is not source evidence.

## Research procedure

1. Extract `model_info.id`, `name`, `developer`, `inference_platform`,
   `inference_engine`, model-level `additional_details`, generation configuration, and
   source links from every affected record.
2. Inspect the PR's adapter or source conversion. A label containing another model's
   name may describe a reward model, judge, or training recipe rather than the served
   policy model.
3. Search the exact variant, including dates, checkpoints, fine-tune suffixes, and
   quantization. Do not substitute a nearby family member.
4. Pin web evidence to a revision, release date, or retrieval date. Prefer official
   model cards, repositories, papers, and provider docs over aggregators.
5. Record one row per evidence-equivalent model group:

| Raw label | Canonical id | Deployment | Availability | Evidence and revision | Confidence |
|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | high/medium/low |

6. Use `unknown` only after listing the relevant surfaces checked and why they did not
   establish the axis. Do not turn lack of quick evidence into a provider-wide guess.

## Batch-edit guardrails

- Group by verified serving method and exact model identity, never just developer.
- Re-open at least one file from every group after a mechanical edit.
- Search for old, missing, and new values after editing; confirm the counts match the
  evidence table.
- Validate all affected files through the CLI at their final paths.
- Retain a short source note in `additional_details` or the PR decision log when the
  decision is not evident from typed fields.
