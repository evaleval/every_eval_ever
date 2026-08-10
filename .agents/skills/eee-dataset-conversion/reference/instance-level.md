# Instance level — the `_samples.jsonl` sidecar

*Scope: everything about the per-item instance record and its link to the aggregate.*

## json vs jsonl
| | Aggregate `.json` | Instance `<uuid>_samples.jsonl` |
|---|---|---|
| Schema | `eval.schema.json` (`EvaluationLog`) | `instance_level_eval.schema.json` (`InstanceLevelEvaluationLog`) |
| Grain | one file per (model[, relationship]) | one JSON object per line, one per item |
| Link | — | aggregate's `detailed_evaluation_results`: `format`, `file_path`, `hash_algorithm`, `checksum`, `total_rows` |
| When | always | only if you have per-item data |

- Instances are a sidecar next to the aggregate, not a replacement.
- Gate them behind `--include-instances` — they're large.
- If the raw per-item data is already publicly hosted, consider **not** re-hosting
  (ship aggregates + a generator pointer / on-demand mode instead).

## The record is CLOSED (`extra='forbid'`)
Unknown top-level keys hard-fail. There is **no top-level `additional_details`** on
the instance record (the aggregate only offers `additional_details` on nested
sub-objects, never at top level either) — route instance extras into `metadata`.

**Required on every line:** `schema_version`, `evaluation_id`, `model_id`,
`evaluation_name`, `sample_id`, `interaction_type`, `input`, `answer_attribution`,
`evaluation`.

## Field traps

### Identity & linking
- `schema_version` — import `SCHEMA_VERSION` from `every_eval_ever.helpers` (the same
  value the aggregate uses); never hardcode a separate "instance version".
- `evaluation_id` — required FK, byte-identical to the aggregate log's
  `evaluation_id`. A mismatch orphans the sidecar.
- `model_id` — a flat, required HF `developer/model` string on the instance —
  NOT the nested `model_info.id`, though it must equal it.
- `evaluation_result_id` — optional but the schema's recommended join key to a
  specific aggregate `evaluation_results[].evaluation_result_id`. Set it on the
  aggregate result AND the matching instances — critical in the one-log-per-model
  layout, where `evaluation_id` alone can't say which result a line belongs to.
- `evaluation_name` — required on the instance too. Fan-out: a sample scored by
  K aggregate metrics produces K instance records (one per `evaluation_result_id`),
  never one record spanning many results. Each line still validates, so `validate`
  can't catch a violation. Roll-up vs leaf: when a sample feeds both an *overall*
  result and a *subtask* leaf, decide whether the overall also gets linked instances —
  the schema leans toward every result having some, so either attach instances to the
  overall too (≈doubles rows) or leave a comment that leaf-only is intentional.
- `sample_id` — required dataset id (e.g. `gsm8k_0001`); `sample_hash` is a
  *separate*, optional cross-model fallback — they coexist.

### Content (input / output / answer_attribution)
- `input.raw` — the prompt only (user/system turns). *Worst bug: a source
  `conversation` that includes the assistant turn leaks the answer into `input.raw`
  and breaks cross-model hashing — filter to input roles.* `input.formatted`
  (optional) is the model-facing string (chat template / few-shot); for few-shot
  evals put the bare question in `raw`, the templated prompt in `formatted`.
- `output.raw` — the model's full generation (a list). `input.reference`
  is also a list. If the source has multiple scorers, keep `output.raw`,
  `extraction_method`, and `extracted_value` pointing at the same one — don't take
  the scorer name from one place and the output text from another.
- `answer_attribution` — a REQUIRED list of objects, each needing all five:
  `turn_idx` (≥0; 0 for single_turn), `source` (`"output.raw"`), `extracted_value`,
  `extraction_method` (the scorer you *actually ran*), `is_terminal` (bool). If the
  source didn't persist a parsed value, re-run the scorer's extractor and set
  `extraction_method` to what you ran — but keep `evaluation.is_correct` from the
  *source* score.

### Hashing (`sample_hash`)
Optional cross-model/adapter join key — use exactly this recipe so hashes match
across adapters (matches `every_eval_ever/adapters/openeval`; historically the
converters computed it inconsistently). Full `reference` list; `[]` when empty. Only
meaningful once
`input.raw` is answer-free:
```python
sample_hash = hashlib.sha256(
    json.dumps({"raw": input.raw, "reference": input.reference},
               sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
```

### Scoring (`evaluation`, `token_usage`)
- `evaluation` = `{score, is_correct}`, both required. `score` is an unconstrained
  float (per-item contribution; not range-checked, independent of the aggregate
  score). `is_correct` is only meaningful for binary 0/1 metrics.
- `token_usage` — optional but all-or-nothing: if present, all three of
  `input_tokens`/`output_tokens`/`total_tokens` are required — with partial data,
  omit the whole object.

### interaction_type + multi_turn / agentic
- `interaction_type` = `single_turn|multi_turn|agentic`, enforced by a hard XOR
  validator: `single_turn` ⟹ `output` set AND `messages` null; `multi_turn`/
  `agentic` ⟹ `messages` set AND `output` null.
- **multi_turn/agentic:** build ordered `messages=[Message(turn_idx, role, content,
  tool_calls, tool_call_id)]`; each `ToolCall` needs `id`+`name` and string-only
  `arguments`; `tool_call_id` is a list; set `evaluation.num_turns`/`tool_calls_count`.

## Write-order — stage, then publish (do NOT hand-roll it)
The apparent chicken-and-egg (you need the uuid to name the sidecar, and the sidecar's
bytes to checksum it) is solved by minting the uuid yourself first and letting the
repo's publisher do the writing:
1. `file_uuid = str(uuid.uuid4())` — you own it; the publisher rejects anything that
   isn't a canonical UUIDv4.
2. Write the jsonl into a staging dir at
   `datastore_output_dir(staged_root, collection, model_id, developer)/<uuid>_samples.jsonl`,
   accumulating `hashlib.sha256()` over the exact bytes you write.
3. `log.detailed_evaluation_results = DetailedEvaluationResults(format=jsonl,
   file_path=datastore_repo_file_path(collection, model_id, developer,
   f"{file_uuid}_samples.jsonl"), hash_algorithm=sha256, checksum=..., total_rows=n)`.
4. `publish_evaluation_logs([log], base_output_dir, [file_uuid],
   staged_output_dir=staged_root, collection_override=collection)` — from
   `every_eval_ever.converters.common.publication`. It re-validates the log, re-reads
   and re-checksums the staged samples, re-parses every line, refuses to overwrite
   an existing file, refuses two identities that route to the same directory, and rolls
   back everything it created if any of that fails.

`file_path` is the full repository-relative path
`data/<collection>/<developer>/<model>/<uuid>_samples.jsonl` — **not** the basename.
Build it with `helpers.datastore_repo_file_path(...)`; a basename (or any other spelling)
is a hard error from both the publisher and `validate`'s companion check.

**The collection directory comes from `evaluation_results[0].source_data.dataset_name`**
unless you pass `collection_override`. Pass the override whenever the first result's
dataset name isn't the collection you want (see `fields.md` §collection).

What the gate then re-derives and compares (all hard errors — see
`reference/datastore-gate.md`): every sample's `evaluation_id` and `model_id` equal
the aggregate's · `total_rows` equals the real line count · at least one row · no blank
rows · `hash_algorithm=sha256` and the checksum matches the published bytes · aggregate
and sidecar share one uuid and one directory. See `templates/instance_sidecar.py`.
