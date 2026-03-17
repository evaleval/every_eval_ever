from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, ConfigDict

from every_eval_ever.eval_types import ScoreType


class _StrictSupplementalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class SupplementalModelInfo(_StrictSupplementalModel):
    additional_details: dict[str, Any] | None = None

class SupplementalSourceData(_StrictSupplementalModel):
    additional_details: dict[str, Any] | None = None

class SupplementalAgenticEvalConfig(_StrictSupplementalModel):
    additional_details: dict[str, Any] | None = None

class SupplementalGenerationConfig(_StrictSupplementalModel):
    additional_details: dict[str, Any] | None = None

class SupplementalScoreDetails(_StrictSupplementalModel):
    details: dict[str, Any] | None = None

class SupplementalMetricConfig(_StrictSupplementalModel):
    evaluation_description: str | None = None
    lower_is_better: bool | None = None
    score_type: ScoreType | None = None
    level_names: list[str] | None = None
    level_metadata: list[str] | None = None
    has_unknown_level: bool | None = None
    min_score: float | None = None
    max_score: float | None = None
    additional_details: dict[str, Any] | None = None

class SupplementalForEvaluationResults(_StrictSupplementalModel):
    evaluation_name: str | None = None
    metric_config: SupplementalMetricConfig | None = None
    score_details: SupplementalScoreDetails | None = None

class SupplementalEvalDetails(_StrictSupplementalModel):
    model_info: SupplementalModelInfo | None = None
    source_data: SupplementalSourceData | None = None
    generation_config: SupplementalGenerationConfig | None = None
    agentic_eval_config: SupplementalAgenticEvalConfig | None = None
    evaluation_results: List[SupplementalForEvaluationResults] | None = None
