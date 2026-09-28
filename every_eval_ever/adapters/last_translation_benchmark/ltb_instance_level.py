"""One LTB (example, translation) pair -> one EEE instance-level record.

Instance-level records follow instance_level_eval.schema.json and are written to
`{uuid}_samples.jsonl` next to the aggregate `{uuid}.json`.

Mapping decisions worth knowing:
  - interaction_type is "single_turn": one translation request, one response.
  - `input.raw` is the source text; `input.formatted` is the actual translation
    prompt the system was given (scripts/20b), so the record is reproducible.
  - `input.reference` is the list of *verification rules*, not a reference
    translation. In LTB the rules are the gold scoring criterion; the human
    translation is kept in `metadata.human_translation`.
  - `evaluation.score` is 1.0 only when every rule passes, matching how LTB
    computes leaderboard scores (scripts/41-score_leaderboard.py).
  - `metadata` values must all be strings per the schema.
  - `source_media` (base64 image/audio/video) is never copied into the record;
    only `metadata.has_media` / `metadata.media_type` are kept.
"""

import hashlib

from .ltb_constants import (
    INSTANCE_SCHEMA_VERSION,
    PROMPT_NOT_REPRODUCIBLE,
    TRANSLATE_PROMPT_TEMPLATE,
)
from .ltb_types import ModelInfo, Outcome


def media_type(source_media: str | None) -> str | None:
    """Mirror LTB's mime sniffing (server/utils.py)."""
    if not source_media:
        return None
    mime = source_media.split(",")[0]
    if "audio" in mime:
        return "audio"
    if "video" in mime:
        return "video"
    return "image"


def sample_hash(example: dict) -> str:
    """Stable hash over source text + verification rules.

    Lets the same example be joined across models even if ids ever shift.
    """
    rules = example.get("verification_rules") or []
    payload = (example.get("source_text") or "") + "\n" + "\n".join(rules)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def formatted_prompt(example: dict, model: ModelInfo | None = None) -> str | None:
    """Reconstruct the exact translation prompt LTB sent to this system.

    Mirrors scripts/20b-translate_by_extra_models.py::get_prompt for the
    API-served LLMs, including its "Translate the provide {context_type}"
    wording -- reproduced verbatim so the record shows what the models actually
    saw, typo and all -- and scripts/22-translate_by_MTs.py for the dedicated MT
    systems, which have their own formats.

    Returns None when the exact input cannot be reconstructed from the public
    release (chat templates, FLORES codes, API translation options). A null is
    honest; a plausible-looking wrong prompt is not.
    """
    if model is not None and model.prompt is PROMPT_NOT_REPRODUCIBLE:
        return None

    src = example.get("source_lang", "")
    tgt = example.get("target_lang", "")
    text = example.get("source_text") or ""
    mtype = media_type(example.get("source_media"))

    if model is not None and isinstance(model.prompt, str):
        # a dedicated MT system with its own format; these are all text-only
        return model.prompt.format(source_lang=src, target_lang=tgt, source_text=text)

    if not mtype:
        prompt = TRANSLATE_PROMPT_TEMPLATE.format(
            source_lang=src, target_lang=tgt, source_text=text
        )
    elif text:
        prompt = (
            f"Translate the following text from {src} to {tgt}. "
            f"Use the provided {mtype} as additional context. "
            f"Output only the translation and nothing else:\n{text}"
        )
    else:
        prompt = (
            f"Translate the provide {mtype} from {src} to {tgt}. "
            f"Output only the textual translation and nothing else."
        )

    if example.get("source_instructions"):
        prompt += f'\nAdditional instructions for this translation are: "{example["source_instructions"]}"'
    return prompt


def human_translation(example: dict) -> str | None:
    for mt in example.get("translations") or []:
        if mt.get("model") == "human":
            return mt.get("translation")
    return None


def _meta(key: str, value) -> tuple[str, str] | None:
    """Coerce a metadata value to a non-empty string, or drop it."""
    if value is None:
        return None
    if isinstance(value, bool):
        return key, "true" if value else "false"
    if isinstance(value, (list, tuple)):
        joined = "; ".join(str(v) for v in value if v is not None)
        return (key, joined) if joined else None
    text = str(value).strip()
    return (key, text) if text else None


def build_instance_record(
    outcome: Outcome,
    model: ModelInfo,
    evaluation_id: str,
    evaluation_name: str,
    evaluation_result_id: str,
) -> dict:
    example = outcome.example
    mtype = media_type(example.get("source_media"))

    metadata_pairs = [
        _meta("ltb_example_id", example.get("id")),
        _meta("source_lang", example.get("source_lang")),
        _meta("target_lang", example.get("target_lang")),
        _meta("source_lang_iso", example.get("source_lang_iso")),
        _meta("target_lang_iso", example.get("target_lang_iso")),
        _meta("language_pair", outcome.lang_pair),
        _meta("tags", list(outcome.tags)),
        _meta("linguistics", example.get("linguistics")),
        _meta("rules_passed", f"{outcome.rules_passed}/{outcome.rules_total}"),
        _meta("has_media", bool(example.get("source_media"))),
        _meta("media_type", mtype),
        _meta("source_instructions", example.get("source_instructions")),
        _meta("attribution", example.get("attribution")),
        _meta("human_translation", human_translation(example)),
    ]
    metadata = dict(p for p in metadata_pairs if p is not None)

    return {
        "schema_version": INSTANCE_SCHEMA_VERSION,
        "evaluation_id": evaluation_id,
        "evaluation_result_id": evaluation_result_id,
        "model_id": model.model_id,
        "evaluation_name": evaluation_name,
        "sample_id": str(example.get("id")),
        "sample_hash": sample_hash(example),
        "interaction_type": "single_turn",
        "input": {
            "raw": example.get("source_text") or "",
            "formatted": formatted_prompt(example, model),
            "reference": list(example.get("verification_rules") or []),
        },
        "output": {"raw": [outcome.translation]},
        "messages": None,
        "answer_attribution": [
            {
                "turn_idx": 0,
                "source": "output.raw",
                "extracted_value": "pass" if outcome.passed else "fail",
                "extraction_method": "llm_judge",
                "is_terminal": True,
            }
        ],
        "evaluation": {
            "score": 1.0 if outcome.passed else 0.0,
            "is_correct": outcome.passed,
            "num_turns": 1,
        },
        "error": None,
        "metadata": metadata,
    }
