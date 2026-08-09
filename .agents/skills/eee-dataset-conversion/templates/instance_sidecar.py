"""Template: instance `_samples.jsonl` sidecar (single_turn Q&A shown).

Verified against every_eval_ever/adapters/openeval + instance_level_eval.schema.json.
Encodes the messages/output XOR, answer_attribution-as-list, token_usage
all-or-nothing, the exact sample_hash recipe, and the
mint-uuid -> stage samples -> publish sequence that the repo's publisher enforces.
"""
import hashlib
import json
import uuid

from every_eval_ever.converters.common.publication import (
    publish_evaluation_logs,
)
from every_eval_ever.eval_types import (
    DetailedEvaluationResults,
    Format,
    HashAlgorithm,
)
from every_eval_ever.helpers import SCHEMA_VERSION
from every_eval_ever.helpers.io import (
    datastore_output_dir,
    datastore_repo_file_path,
)
from every_eval_ever.instance_level_types import (
    AnswerAttributionItem,
    Evaluation,
    Input,
    InstanceLevelEvaluationLog,
    InteractionType,
    Output,
    TokenUsage,
)

SRC = "<src>"
COLLECTION = "<src>"          # same collection as the aggregate adapter


def _sample_hash(raw: str, reference: list[str]) -> str:   # ONE recipe — every adapter MUST match
    payload = json.dumps({"raw": raw, "reference": reference},   # FULL list, not first elem / not a str
                         sort_keys=True, separators=(",", ":"))  # canonical JSON; [] when empty
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()   # == adapters/openeval.sample_hash


def _instance(item, evaluation_id, evaluation_name, evaluation_result_id, model_id):
    tok = None                                             # token_usage is ALL-OR-NOTHING:
    if item.in_tok is not None and item.out_tok is not None:   # build it only if you have all three
        tok = TokenUsage(input_tokens=item.in_tok, output_tokens=item.out_tok,
                         total_tokens=item.in_tok + item.out_tok)
    return InstanceLevelEvaluationLog(
        schema_version=SCHEMA_VERSION,                     # same version as the aggregate
        evaluation_id=evaluation_id,                       # REQUIRED FK: byte-identical to aggregate's
        evaluation_result_id=evaluation_result_id,         # FK to THIS evaluation_results[] row
        evaluation_name=evaluation_name,                   # REQUIRED on the instance too (fallback FK)
        model_id=model_id,                                 # REQUIRED flat HF id == model_info.id
        sample_id=item.sample_id,                          # REQUIRED dataset id (e.g. gsm8k_0001)
        sample_hash=_sample_hash(item.raw, item.reference),# optional cross-model fallback for sample_id
        interaction_type=InteractionType.single_turn,      # -> output set, messages MUST stay null
        input=Input(raw=item.raw,                          # bare question: model-INDEPENDENT, answer-FREE
                    reference=item.reference,              # list[str], NOT a str
                    formatted=item.prompt),                # optional: chat-templated / few-shot string
        output=Output(raw=item.output),                    # list[str]; null for multi_turn/agentic
        answer_attribution=[AnswerAttributionItem(         # REQUIRED list; ALL 5 fields per item
            turn_idx=0, source="output.raw",               # 0 for single_turn
            extracted_value=item.parsed,                   # parsed answer (re-run scorer if source lacks it)
            extraction_method=item.scorer,                 # PLACEHOLDER: the scorer you ACTUALLY ran (e.g. "match")
            is_terminal=True)],                            # true = final answer
        evaluation=Evaluation(score=item.score,            # unconstrained float; 0.0/1.0 for binary
                              is_correct=item.is_correct),  # from the SOURCE score; binary-only meaningful
        token_usage=tok,                                   # None unless all three counts present
        metadata={"subject": str(item.subject)})           # extras go HERE; str values only


def export_with_instances(log, developer, model_name, items, out_root, staged_root,
                          collection=COLLECTION):
    """Mint the uuid, stage the sidecar, then let the publisher write both files.

    `out_root` and `staged_root` are both `.../data` directories. The publisher
    re-validates the log, re-reads and re-checksums the staged samples, re-parses every
    line (each row's evaluation_id AND model_id must equal the aggregate's), refuses to
    overwrite an existing file, and rolls back anything it created on failure.
    """
    file_uuid = str(uuid.uuid4())                          # WE own it; publisher demands UUIDv4
    model_id = log.model_info.id
    sample_name = f"{file_uuid}_samples.jsonl"
    staged_dir = datastore_output_dir(staged_root, collection, model_id, developer)
    staged_dir.mkdir(parents=True, exist_ok=True)

    # READ the join keys off the aggregate instead of recomputing them: the instance FK
    # must equal the aggregate's evaluation_result_id, and re-deriving it by formula means
    # the two silently diverge the day either side's id scheme changes.
    result_ids = {r.evaluation_name: r.evaluation_result_id for r in log.evaluation_results}

    lines, digest = [], hashlib.sha256()
    for item in items:
        # DEFAULT one-log-per-model grain: look up the result PER ITEM so each line attaches
        # to the right aggregate result. An item with no matching result is an orphan FK --
        # fail loudly rather than emit a line pointing at nothing.
        name = f"{SRC}.{item.benchmark}"
        if name not in result_ids:
            raise ValueError(
                f"sample {item.sample_id!r} names {name!r}, which is not one of the "
                f"aggregate's results {sorted(result_ids)}")
        rec = _instance(item, log.evaluation_id, name, result_ids[name], model_id)
        line = (json.dumps(rec.model_dump(mode="json", exclude_none=True),
                           ensure_ascii=False) + "\n").encode("utf-8")
        digest.update(line)                                # checksum the EXACT bytes
        lines.append(line)
    # write_bytes, not write_text: text mode would translate newlines on Windows and the
    # checksum would no longer match the published file.
    (staged_dir / sample_name).write_bytes(b"".join(lines))

    log.detailed_evaluation_results = DetailedEvaluationResults(
        format=Format.jsonl,
        # FULL repository-relative path, NOT the basename -- data/<collection>/<dev>/<model>/
        # <uuid>_samples.jsonl. Anything else is a hard error from the publisher and from
        # the validator's companion check.
        file_path=datastore_repo_file_path(collection, model_id, developer, sample_name),
        hash_algorithm=HashAlgorithm.sha256,               # REQUIRED to interpret the checksum
        checksum=digest.hexdigest(), total_rows=len(lines))  # total_rows = real line count

    return publish_evaluation_logs([log], out_root, [file_uuid],
                                   staged_output_dir=staged_root,
                                   # pass the override so the collection is YOUR choice and
                                   # not whatever evaluation_results[0].source_data happens
                                   # to be named (fields.md §collection):
                                   collection_override=collection)
