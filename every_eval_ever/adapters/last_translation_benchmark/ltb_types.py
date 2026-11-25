"""Small value types shared by the LTB -> EEE converter."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelInfo:
    """Resolved EEE identity of one LTB translation system."""

    display_name: str          # as it appears in v1.json, e.g. "Gemini 3.1 Pro"
    model_id: str              # HF-style id, e.g. "google/gemini-3.1-pro-preview"
    developer: str             # e.g. "Google"
    platform: str | None       # inference_platform, for remote APIs
    deployment_type: str       # self_deployed | externally_managed | unknown
    availability: str          # open_weights | closed_weights | unknown
    needs_review: str | None = None
    engine: str | None = None      # inference_engine name, for locally served models
    prompt: object | None = None   # custom template, or PROMPT_NOT_REPRODUCIBLE
    prompt_note: str | None = None
    gen_args: dict | None = None   # decoding parameters actually used

    @property
    def developer_dir(self) -> str:
        return self.model_id.split("/")[0]

    @property
    def model_dir(self) -> str:
        parts = self.model_id.split("/", 1)
        return parts[1] if len(parts) > 1 else parts[0]


@dataclass
class Outcome:
    """One (model, example) verification outcome."""

    example_id: int
    passed: bool               # all rules passed
    rules_passed: int
    rules_total: int
    translation: str
    tags: tuple[str, ...]
    lang_pair: str             # e.g. "eng-deu"
    example: dict = field(repr=False, default_factory=dict)


@dataclass
class Stats:
    """Counters for the conversion report."""

    examples_total: int = 0
    models_seen: int = 0
    models_unknown: list[str] = field(default_factory=list)
    models_thin: dict[str, int] = field(default_factory=dict)
    aliases_merged: dict[str, dict] = field(default_factory=dict)
    skipped_no_translation: int = 0
    skipped_unverified: int = 0
    missing_translations: int = 0
    skipped_blocklisted: int = 0
    files_written: int = 0
    instance_rows: int = 0
