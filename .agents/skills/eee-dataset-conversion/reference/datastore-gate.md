# The merge gate — what fails a record after the schema passes

*Scope: the checks that live in code, not in the JSON schema. `fields.md` says what a
field means; this file says what will reject it. Everything here is a hard error
unless marked warning.*

**Two gates run, and they can disagree.**
1. **Local**: `uv run python -m every_eval_ever validate <files>` — runs the schema plus the
   semantic checks (`validator/validation_core.py`, `REGISTERED_CHECKS`). The in-library
   `validate_file(path)` used by unit tests defaults to semantic checks off, so a
   green test does *not* mean the gate is green. Semantic checks also need the file at its
   final `data/<collection>/<dev>/<model>/` path.
2. **The datastore bot**: comment `/eee validate changed` on the HF PR. It runs its own
   deployed version, which may be pinned to an older compatibility version with older
   enum vocabularies. Treat a bot complaint that contradicts this file as a version skew,
   not as something to silently paper over — fix what it asks, and say so in the PR.

Re-derive this list from `REGISTERED_CHECKS`, `_DEPLOYMENT_TYPES`,
`_MODEL_AVAILABILITY_TYPES`, `_AGGREGATE_FILE_RE`/`_INSTANCE_FILE_RE`, and
`converters/common/publication.py` whenever it looks stale.

## §path — `check_path_structure`
- Exactly five parts: `data/<collection>/<developer>/<model>/<filename>`.
- Filename must match `{UUID4}.json` or `{UUID4}_samples.jsonl` — a canonical v4
  uuid. An md5, a slug, or a hand-written name is rejected.
- `<collection>`, `<developer>`, `<model>` must be portable filesystem names: no
  `<>:"\|?*`, no control chars, no trailing `.`/space, no Windows reserved name
  (`CON`, `PRN`, `AUX`, `NUL`, `COM1-9`, `LPT1-9`), and none of them may be `data`.
- **Never build these components by string-joining source values** — a model id with `/`,
  `..`, or an absolute prefix escapes your output dir. Use
  `helpers.datastore_output_dir` / `datastore_repo_file_path` / `datastore_path_components`
  (they flatten `/` to `_` and reject unusable identities), `require_identity` for a value
  that must be known, `sanitize_filename` for free text, `require_uuid4` for the uuid.

## §companion — `check_companion_exists` / `check_instance_companion`
- `detailed_evaluation_results.file_path` = the full repo-relative path
  `data/<collection>/<dev>/<model>/<uuid>_samples.jsonl`. Not a basename, not absolute,
  not a different folder or uuid. Build it with `datastore_repo_file_path`.
- The declared companion must actually exist, and a `_samples.jsonl` must have its
  sibling aggregate — one uuid, one directory, both directions.
- Every sample row's `evaluation_id` and `model_id` must equal the aggregate's.
- `total_rows` must equal the real line count; ≥1 row; no blank rows.
- `hash_algorithm` must be `sha256` and `checksum` must match the published bytes.

## §score — `check_score_metadata` (registered as error)
- `score_details.score` must be a finite number.
- **The score must lie inside `[min_score, max_score]`** and `min_score <= max_score`.
  This is where the percent-vs-proportion mismatch surfaces: declaring `0.0–1.0` while
  the source reports `73.4` is now a hard failure, not a silent unit bug. Declare the
  bounds the *source's* numbers live in, and convert deliberately if you rescale.
- `score_type: continuous` requires both bounds; a supplied-but-unparseable bound is
  also an error. `±inf` is accepted (serialized as the JSON strings
  `"Infinity"`/`"-Infinity"`); `null` is "not provided", never "unbounded".
- Uncertainty must be finite where present: `standard_deviation`,
  `standard_error.value`, `confidence_interval.{lower,upper,confidence_level}`.

## §deployment — `check_model_deployment`
- `model_info.additional_details.deployment_type` ∈ `self_deployed | externally_managed |
  unknown`; `model_availability` ∈ `open_weights | closed_weights | unknown`.
- The library **auto-fills both to `"unknown"`**, so the library validate can't tell you
  that you never set them — the CLI errors on a missing key or a non-enum value.
- Set the real values (a closed API model → `externally_managed` + `closed_weights`);
  shipping an unconsidered `"unknown"` is the single most common datastore warning.
- Older records and the deployed bot may use a different vocabulary
  (`api|local|unknown`, `closed_source|open_weights_deployment|other`). Read the enums
  from the installed validator, not from an existing record or an old bot message.

## §publish — `publish_evaluation_logs`
**Mind which root each entry point wants — they differ, and a mismatch is silent until
the path check rejects the depth:** `publish_evaluation_logs(base_output_dir=…)` takes the
**`data` dir**, while `EvaluationLogOutput(base_dir=…)` (for `save_evaluation_logs`) and
`default_failure_report_path(…)` take **`data/<collection>`**, one level down.

Use the publisher instead of writing files yourself; it enforces, before creating anything:
- the log re-validates as an `EvaluationLog`;
- `evaluation_results[0].source_data` exists (it determines the collection
  directory — pass `collection_override` when the first result's `dataset_name` isn't
  the collection you want);
- two distinct collection/model identities may not route to the same directory;
- it refuses to overwrite an existing file (so re-running into a populated output dir
  fails loudly instead of minting a duplicate logical record under a fresh uuid);
- staged samples are re-read, re-checksummed, and re-parsed line by line;
- strict JSON out (`allow_nan=False`);
- any failure rolls back every file and directory it created.

## §partial conversions
A source row you cannot represent must be accounted for, not dropped:
- keep every valid record, collect `SourceRecordFailure(source_ref, reason,
  source_record)` for each rejected row, and `SourceRecordExclusion` for rows that are
  intentionally not evaluations (a published random baseline);
- return a `SourceConversionResult`, write it with `save_failure_report(result,
  default_failure_report_path(output_dir))` → `adapter_reports/<collection>_failures.json`
  (outside `data/`, and never passed to the validator);
- `result.raise_if_incomplete()` → the command exits non-zero on failures so
  automation can tell a partial refresh from a complete one. Exclusions don't fail.
- The numbers in that report are what substantiates the PR's coverage line.

## §also
- `additional_details` (and every string-map: instance `metadata`, `tool_calls[].arguments`,
  `performance.additional_details`) is `dict[str, str]` — `json.dumps` anything else first.
  Pydantic types it `Any`, so `validate` may pass what the JSON schema rejects.
- LLM-judged metrics: `metric_config.llm_scoring` requires `judges` (≥1, each with a full
  `model_info`) and `input_prompt` (the judging prompt template). A rubric/judge
  benchmark that omits the judge's `model_info` fails every file — see `fields.md`
  §llm_scoring.
- `source_data.url` needs ≥1 entry (`min_length=1`); an empty list fails.
- Duplicates across a collection: `uv run python -m every_eval_ever.check_duplicate_entries
  <files>` before submitting a refresh.
