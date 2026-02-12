"""Schema construction helpers for building evaluation logs."""

import time
from typing import Any, Dict, List, Optional

from eval_types import (
    EvaluationLog,
    EvaluationResult,
    EvaluatorRelationship,
    MetricConfig,
    ModelInfo,
    ScoreDetails,
    ScoreType,
    SourceMetadata,
)

from .developer import get_developer, get_model_id


def make_metric_config(
    description: str,
    lower_is_better: bool = False,
    score_type: ScoreType = ScoreType.continuous,
    min_score: float = 0.0,
    max_score: float = 1.0,
    level_names: Optional[List[str]] = None,
    has_unknown_level: Optional[bool] = None,
) -> MetricConfig:
    """
    Create a MetricConfig with sensible defaults.

    Most evaluations use continuous scores from 0-1 where higher is better,
    so those are the defaults.

    Args:
        description: Human-readable description of the metric
        lower_is_better: Whether lower scores are better (default: False)
        score_type: Type of score (default: continuous)
        min_score: Minimum possible score (default: 0.0)
        max_score: Maximum possible score (default: 1.0)
        level_names: For level-based scores, the names of each level
        has_unknown_level: For level-based scores, whether -1 means unknown

    Returns:
        Configured MetricConfig instance
    """
    config = MetricConfig(
        evaluation_description=description,
        lower_is_better=lower_is_better,
        score_type=score_type,
    )

    if score_type == ScoreType.continuous:
        config.min_score = min_score
        config.max_score = max_score
    elif score_type == ScoreType.levels and level_names:
        config.level_names = level_names
        config.has_unknown_level = has_unknown_level

    return config


def make_evaluation_result(
    name: str,
    score: float,
    description: str,
    lower_is_better: bool = False,
    score_type: ScoreType = ScoreType.continuous,
    min_score: float = 0.0,
    max_score: float = 1.0,
    details: Optional[Dict[str, Any]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
) -> EvaluationResult:
    """
    Create an EvaluationResult with a single score.

    Convenience function that combines MetricConfig and ScoreDetails creation.

    Args:
        name: Name of the evaluation (e.g., "MMLU", "GSM8K")
        score: The score value
        description: Human-readable description of what this measures
        lower_is_better: Whether lower scores are better
        score_type: Type of score
        min_score: Minimum possible score
        max_score: Maximum possible score
        details: Additional score details
        generation_config: Configuration used to generate results

    Returns:
        Configured EvaluationResult instance
    """
    return EvaluationResult(
        evaluation_name=name,
        metric_config=make_metric_config(
            description=description,
            lower_is_better=lower_is_better,
            score_type=score_type,
            min_score=min_score,
            max_score=max_score,
        ),
        score_details=ScoreDetails(
            score=round(score, 4) if score is not None else -1,
            details=details,
        ),
        generation_config=generation_config,
    )


def make_source_metadata(
    source_name: str,
    organization_name: str,
    source_type: str = "documentation",
    evaluator_relationship: EvaluatorRelationship = EvaluatorRelationship.third_party,
    organization_url: Optional[str] = None,
) -> SourceMetadata:
    """
    Create SourceMetadata for an evaluation source.

    Args:
        source_name: Name of the source (e.g., "HELM Lite", "RewardBench")
        organization_name: Name of the organization providing the data
        source_type: Either "documentation" or "evaluation_run"
        evaluator_relationship: Relationship to model developer
        organization_url: Optional URL for the organization

    Returns:
        Configured SourceMetadata instance
    """
    return SourceMetadata(
        source_name=source_name,
        source_type=source_type,
        source_organization_name=organization_name,
        source_organization_url=organization_url,
        evaluator_relationship=evaluator_relationship,
    )


def make_model_info(
    model_name: str,
    developer: Optional[str] = None,
    inference_platform: str = "unknown",
    additional_details: Optional[Dict[str, Any]] = None,
) -> ModelInfo:
    """
    Create ModelInfo from a model name.

    Automatically extracts developer if not provided.

    Args:
        model_name: Name of the model
        developer: Optional developer override
        inference_platform: Platform used for inference
        additional_details: Extra model metadata

    Returns:
        Configured ModelInfo instance
    """
    dev = developer or get_developer(model_name)
    model_id = get_model_id(model_name, dev)

    return ModelInfo(
        name=model_name,
        id=model_id,
        developer=dev,
        inference_platform=inference_platform,
        additional_details=additional_details,
    )


def make_evaluation_log(
    source_name: str,
    model_name: str,
    evaluation_results: List[EvaluationResult],
    source_data: List[str],
    organization_name: str,
    source_type: str = "documentation",
    evaluator_relationship: EvaluatorRelationship = EvaluatorRelationship.third_party,
    organization_url: Optional[str] = None,
    developer: Optional[str] = None,
    inference_platform: str = "unknown",
    model_additional_details: Optional[Dict[str, Any]] = None,
    retrieved_timestamp: Optional[str] = None,
) -> EvaluationLog:
    """
    Create a complete EvaluationLog with all components.

    High-level convenience function that assembles all the pieces.

    Args:
        source_name: Name of the evaluation source
        model_name: Name of the model being evaluated
        evaluation_results: List of evaluation results
        source_data: URLs or dataset info for the source data
        organization_name: Organization providing the evaluation
        source_type: Either "documentation" or "evaluation_run"
        evaluator_relationship: Relationship to model developer
        organization_url: Optional URL for the organization
        developer: Optional developer override
        inference_platform: Platform used for inference
        model_additional_details: Extra model metadata
        retrieved_timestamp: Optional timestamp override

    Returns:
        Complete EvaluationLog ready to save
    """
    timestamp = retrieved_timestamp or str(time.time())
    dev = developer or get_developer(model_name)
    model_id = get_model_id(model_name, dev)

    # Build evaluation_id: source_name/model_id_sanitized/timestamp
    sanitized_model_id = model_id.replace("/", "_")
    evaluation_id = f"{source_name}/{sanitized_model_id}/{timestamp}"

    return EvaluationLog(
        schema_version="0.1.0",
        evaluation_id=evaluation_id,
        retrieved_timestamp=timestamp,
        source_data=source_data,
        source_metadata=make_source_metadata(
            source_name=source_name,
            organization_name=organization_name,
            source_type=source_type,
            evaluator_relationship=evaluator_relationship,
            organization_url=organization_url,
        ),
        model_info=make_model_info(
            model_name=model_name,
            developer=dev,
            inference_platform=inference_platform,
            additional_details=model_additional_details,
        ),
        evaluation_results=evaluation_results,
    )
