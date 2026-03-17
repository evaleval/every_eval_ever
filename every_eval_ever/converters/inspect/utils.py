import json
import re
from pathlib import Path

from pydantic import BaseModel
from typing import Any, Dict, List, Type

from every_eval_ever.eval_types import (
    AgenticEvalConfig,
    EvaluationResult,
    GenerationArgs,
    GenerationConfig,
    InferenceEngine,
    MetricConfig,
    ModelInfo
)
from every_eval_ever.converters.common.utils import get_model_organization_info
from every_eval_ever.converters.inspect.supplemental_eval_details import (
    SupplementalAgenticEvalConfig,
    SupplementalEvalDetails,
    SupplementalForEvaluationResults,
    SupplementalGenerationConfig,
    SupplementalSourceData,
)


class ModelPathHandler:
    """Base class for all model path parsing strategies."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.parts = model_path.split('/')

    def handle(self) -> ModelInfo:
        """Must be implemented by subclasses to return the parsed ModelInfo."""
        raise NotImplementedError


def normalize_claude_model_name(model_str: str) -> str:
    """
    Normalizes any Claude model identifier to the canonical format:
    'claude-3-5-sonnet@YYYYMMDD'
    """
    base_match = re.search(r'(claude-\d+-\d+-[a-z]+)', model_str)
    if not base_match:
        # Return the original string if the base model name is not found
        return model_str
    base = base_match.group(1)

    # Use '00000000' as a default date if not found
    date_match = re.search(r'@?(\d{8})', model_str)
    date = date_match.group(1) if date_match else '00000000'

    return f'{base}@{date}'


class ClosedApiHandler(ModelPathHandler):
    """Handles paths for closed API providers like OpenAI, Anthropic, Google, etc."""

    def handle(self) -> ModelInfo:
        developer = self.parts[0]
        inference_platform = self.parts[0]
        model_id = self.model_path

        # Special handling for Anthropic models running on cloud platforms (Vertex/Bedrock)
        if self.model_path.startswith(
            'anthropic/vertex'
        ) or self.model_path.startswith('anthropic/bedrock'):
            if len(self.parts) >= 3:
                # e.g., anthropic/vertex/claude-3-5-sonnet-v2@20241022
                developer = self.parts[0]
                inference_platform = self.parts[1]
                model_id = (
                    f'{developer}/{normalize_claude_model_name(self.parts[2])}'
                )

        # General handling for Azure or Vertex (non-Anthropic)
        elif 'azure' in self.model_path or 'vertex' in self.model_path:
            # Assumes the format: developer/platform/model_name
            if len(self.parts) >= 3:
                developer = self.parts[0]
                inference_platform = self.parts[1]
                model_id = f'{developer}/{self.parts[2]}'

        return ModelInfo(
            name=self.model_path,
            id=model_id,
            developer=developer,
            inference_platform=inference_platform,
        )


class BedrockParser:
    @staticmethod
    def parse(model_path: str) -> ModelInfo:
        # Remove any variant suffix after a colon (e.g. ':2')
        base_model = model_path.split('/')[1]

        # Split into developer and model parts
        parts = base_model.split('.', 1)
        developer = parts[0] if len(parts) > 1 else None
        model = parts[1] if len(parts) > 1 else parts[0]

        # Strip trailing version tags like '-v0', '-v1.2'
        model = re.sub(r'-v\d+(\.\d+)?$', '', model)

        return ModelInfo(
            name=model_path,
            id=f'{developer}/{model}' if developer else model,
            developer=developer,
            inference_platform='bedrock',
        )


class AzureAiParser:
    @staticmethod
    def parse(model_name: str) -> ModelInfo:
        parts = model_name.split('/')
        inference_platform = parts[0]
        base_model_name = parts[-1]

        inferred_org_name = get_model_organization_info(base_model_name)
        developer = (
            inferred_org_name
            if inferred_org_name and inferred_org_name != 'not_found'
            else inference_platform
        )

        return ModelInfo(
            name=model_name,
            id=f'{developer}/{base_model_name}',  # Corrected 'id' logic
            developer=developer,
            inference_platform=inference_platform,
        )


class CloudApiHandler(ModelPathHandler):
    """Handles paths for cloud API providers like AWS Bedrock and Azure AI."""

    def handle(self) -> ModelInfo:
        if self.model_path.startswith('bedrock'):
            return BedrockParser.parse(self.model_path)

        elif self.model_path.startswith('azureai'):
            return AzureAiParser.parse(self.model_path)

        return ModelInfo(
            name=self.model_path,
            id=self.model_path,
            developer='unknown',
            inference_platform='cloud_api',
        )


class HostedOpenHandler(ModelPathHandler):
    """Handles paths for hosted open models (Together, Groq, Fireworks, etc.)."""

    # Model name to developer map for Sambanova
    SAMBANOVA_DEV_MAP = {
        'llama': 'meta-llama',
        'gpt': 'openai',
        'qwen': 'qwen',
        'swallow': 'tokyotech-llm',
        'allam': 'humain-ai',
        'mistral': 'mistral',
        'deepseek': 'deepseek-ai',
    }

    def handle(self) -> ModelInfo:
        inference_platform = self.parts[0]
        model_id = self.model_path
        developer = 'unknown'

        path_lower = self.model_path.lower()

        # Group 1: Platform/Developer/Model format
        platforms = [
            'together',
            'cf',
            'openrouter',
            'openai-api',
            'hf-inference-providers',
        ]
        if any(path_lower.startswith(p) for p in platforms):
            if len(self.parts) >= 3:
                developer = self.parts[1]
                model_id = f'{developer}/{self.parts[2]}'
            else:
                developer = inference_platform  # Fallback

        # Group 2: Groq specific logic
        elif path_lower.startswith('groq'):
            if len(self.parts) >= 2:
                model_name = self.parts[1]
                is_llama = 'llama' in model_name.lower()
                developer = 'meta-llama' if is_llama else inference_platform
                model_id = f'{developer}/{model_name}'

        # Group 3: Sambanova specific logic
        elif path_lower.startswith('sambanova'):
            if len(self.parts) >= 2:
                model_name_part = self.parts[-1].lower()

                for key, dev in self.SAMBANOVA_DEV_MAP.items():
                    if key in model_name_part:
                        developer = dev
                        break
                model_id = f'{developer}/{self.parts[-1]}'

        # Group 4: Fireworks specific logic
        elif path_lower.startswith('fireworks'):
            # e.g. fireworks/accounts/fireworks/models/deepseek-r1-0528
            model_name = self.parts[-1]
            # Assuming get_model_organization_info returns an object with a 'organization' key
            inferred_org_name = get_model_organization_info(model_name)
            developer = (
                inferred_org_name
                if inferred_org_name != 'not_found'
                else 'unknown'
            )
            model_id = f'{developer}/{model_name}'

        if developer == 'unknown':
            if len(self.parts) >= 2:
                developer = self.parts[1]
                model_id = f'{developer}/{self.parts[-1]}'

        return ModelInfo(
            name=self.model_path,
            id=model_id,
            developer=developer,
            inference_platform=inference_platform,
        )


class InferenceEngineHandler(ModelPathHandler):
    """Handles paths for inference engines (vLLM, Ollama, SGLang, etc.)."""

    def handle(self) -> ModelInfo:
        inference_engine = self.parts[0]
        model_id = self.model_path
        developer = 'unknown'  # Default value

        # Group 1: /Engine/Developer/Model format (e.g., vllm/meta-llama/Llama-2-7b-chat)
        if any(self.model_path.startswith(e) for e in ['vllm', 'sglang', 'hf']):
            if len(self.parts) >= 3:
                developer = self.parts[1]
                model_id = f'{self.parts[1]}/{self.parts[2]}'
            else:
                developer = inference_engine

        # Group 2: Ollama and Llama-cpp-python format (e.g., ollama/llama2:7b)
        elif any(
            self.model_path.startswith(e)
            for e in ['ollama', 'llama-cpp-python']
        ):
            if len(self.parts) >= 2:
                developer = self.parts[0]
                model_name = self.parts[1].replace(':', '-')
                model_id = f'{self.parts[0]}/{model_name}'

        return ModelInfo(
            name=self.model_path,
            id=model_id,
            developer=developer,
            inference_engine=InferenceEngine(
                name=inference_engine  # TODO add version if possible
            ),
        )


# Mapping the provider/engine prefix to the specific Handler class
# This is where the extension point is for *new* provider categories.
MODEL_HANDLER_MAP: Dict[str, Type[ModelPathHandler]] = {
    # Closed API Providers
    'openai': ClosedApiHandler,
    'anthropic': ClosedApiHandler,
    'google': ClosedApiHandler,
    'grok': ClosedApiHandler,
    'mistral': ClosedApiHandler,
    'deepseek': ClosedApiHandler,
    'perplexity': ClosedApiHandler,
    # Cloud API Providers
    'bedrock': CloudApiHandler,
    'azure-ai': CloudApiHandler,
    # Hosted Open Providers
    'groq': HostedOpenHandler,
    'together': HostedOpenHandler,
    'fireworks': HostedOpenHandler,
    'cf': HostedOpenHandler,
    'hf-inference-providers': HostedOpenHandler,
    'sambanova': HostedOpenHandler,
    'openrouter': HostedOpenHandler,
    'openai-api': HostedOpenHandler,
    # Inference Engines
    'hf': InferenceEngineHandler,
    'vllm': InferenceEngineHandler,
    'ollama': InferenceEngineHandler,
    'llamacpp': InferenceEngineHandler,
    'sglang': InferenceEngineHandler,
}

PROVIDER_PREFIXES: List[str] = list(MODEL_HANDLER_MAP.keys())


def extract_model_info_from_model_path(model_path: str) -> ModelInfo:
    """
    Infers the ModelInfo by dispatching the model_path to the appropriate handler.

    This function is now simple and focuses only on dispatching.
    To add a new provider/engine, you only need to update the MODEL_HANDLER_MAP
    and create a corresponding Handler class.
    """

    provider_candidate = model_path.split('/')[0].lower()
    handler_class = MODEL_HANDLER_MAP.get(provider_candidate, None)

    if handler_class:
        try:
            handler = handler_class(model_path)
            return handler.handle()
        except Exception as e:
            print(
                f'Handler failed for {model_path}: {e}. Fallback into unknown model developer.'
            )
            pass

    # Fallback
    return ModelInfo(
        name=model_path,
        id=model_path,
        developer='unknown',
        inference_platform='unknown',
    )


def save_to_file(path: str, obj: BaseModel) -> bool:
    json_str = obj.model_dump_json(indent=4, exclude_none=True)

    obj_path = Path(path)
    obj_path.mkdir(parents=True, exist_ok=True)

    with open(obj_path, 'w') as json_file:
        json_file.write(json_str)


SYNTHETIC_METRIC_CONFIG_FIELDS = {
    "evaluation_description",
    "lower_is_better",
    "score_type",
    "level_names",
    "level_metadata",
    "has_unknown_level",
    "min_score",
    "max_score",
}


def parse_supplemental_eval_details(
    raw_supplemental_eval_details: Any,
) -> SupplementalEvalDetails | None:
    if raw_supplemental_eval_details is None:
        return None

    if isinstance(raw_supplemental_eval_details, SupplementalEvalDetails):
        return raw_supplemental_eval_details

    if isinstance(raw_supplemental_eval_details, dict):
        return SupplementalEvalDetails.model_validate(raw_supplemental_eval_details)

    raise ValueError(
        "metadata_args['supplemental_eval_details'] must be a dict or SupplementalEvalDetails instance."
    )


def convert_to_string_dict(data: dict[str, Any] | None) -> dict[str, str] | None:
    if data is None:
        return None
    return {
        str(key): value if isinstance(value, str) else json.dumps(value)
        for key, value in data.items()
    }


def extend_additional_details(
    existing_details: dict[str, str] | None,
    supplemental_details: dict[str, Any] | None,
) -> dict[str, str] | None:
    if supplemental_details is None:
        return existing_details

    supplemental_str = convert_to_string_dict(supplemental_details) or {}
    if existing_details is None:
        return supplemental_str or None

    merged = dict(existing_details)
    for key, value in supplemental_str.items():
        if key not in merged:
            merged[key] = value

    return merged or None


def apply_model_info_supplement(
    model_info: ModelInfo,
    supplemental_eval_details: SupplementalEvalDetails | None,
) -> None:
    if supplemental_eval_details is None or supplemental_eval_details.model_info is None:
        return

    model_info.additional_details = extend_additional_details(
        model_info.additional_details,
        supplemental_eval_details.model_info.additional_details,
    )


def apply_generation_config_supplement(
    evaluation_result: EvaluationResult,
    generation_supplement: SupplementalGenerationConfig | None,
    agentic_supplement: SupplementalAgenticEvalConfig | None,
) -> None:
    if generation_supplement is None and agentic_supplement is None:
        return

    if evaluation_result.generation_config is None:
        evaluation_result.generation_config = GenerationConfig()

    generation_config = evaluation_result.generation_config
    if generation_supplement is not None:
        generation_config.additional_details = extend_additional_details(
            generation_config.additional_details,
            generation_supplement.additional_details,
        )

    if agentic_supplement is None:
        return

    if generation_config.generation_args is None:
        generation_config.generation_args = GenerationArgs()

    if generation_config.generation_args.agentic_eval_config is None:
        generation_config.generation_args.agentic_eval_config = AgenticEvalConfig()

    generation_config.generation_args.agentic_eval_config.additional_details = (
        extend_additional_details(
            generation_config.generation_args.agentic_eval_config.additional_details,
            agentic_supplement.additional_details,
        )
    )


def apply_source_data_supplement(
    evaluation_result: EvaluationResult,
    source_data_supplement: SupplementalSourceData | None,
) -> None:
    if source_data_supplement is None:
        return

    evaluation_result.source_data.additional_details = extend_additional_details(
        evaluation_result.source_data.additional_details,
        source_data_supplement.additional_details,
    )


def apply_metric_config_supplement(
    evaluation_result: EvaluationResult,
    supplement: SupplementalForEvaluationResults,
) -> None:
    metric_supplement = supplement.metric_config
    if metric_supplement is None:
        return

    current = evaluation_result.metric_config.model_dump(mode="python")
    supplemental = metric_supplement.model_dump(mode="python", exclude_none=True)

    additional_details = supplemental.pop("additional_details", None)

    for field_name, field_value in supplemental.items():
        if (
            field_name in SYNTHETIC_METRIC_CONFIG_FIELDS
        ):
            current[field_name] = field_value

    current["additional_details"] = extend_additional_details(
        current.get("additional_details"),
        additional_details,
    )

    evaluation_result.metric_config = MetricConfig.model_validate(current)


def apply_result_supplement(
    evaluation_result: EvaluationResult,
    supplement: SupplementalForEvaluationResults | None,
) -> None:
    if supplement is None:
        return

    if supplement.score_details is not None:
        evaluation_result.score_details.details = extend_additional_details(
            evaluation_result.score_details.details,
            supplement.score_details.details,
        )

    apply_metric_config_supplement(evaluation_result, supplement)


def apply_supplemental_eval_details(
    model_info: ModelInfo,
    evaluation_results: list[EvaluationResult],
    supplemental_eval_details: SupplementalEvalDetails | None,
) -> None:
    if supplemental_eval_details is None:
        return

    apply_model_info_supplement(model_info, supplemental_eval_details)

    for evaluation_result in evaluation_results:
        apply_source_data_supplement(
            evaluation_result,
            supplemental_eval_details.source_data,
        )
        apply_generation_config_supplement(
            evaluation_result,
            supplemental_eval_details.generation_config,
            supplemental_eval_details.agentic_eval_config,
        )

    result_supplements = supplemental_eval_details.evaluation_results or []
    named_supplements = {
        supplement.evaluation_name: supplement
        for supplement in result_supplements
        if supplement.evaluation_name is not None
    }
    if len(named_supplements) != len(
        [s for s in result_supplements if s.evaluation_name is not None]
    ):
        raise ValueError(
            "Duplicate evaluation_name values in supplemental_eval_details.evaluation_results."
        )
    unnamed_supplements = [
        supplement for supplement in result_supplements if supplement.evaluation_name is None
    ]
    unnamed_idx = 0

    for evaluation_result in evaluation_results:
        supplement = named_supplements.get(evaluation_result.evaluation_name)
        if supplement is None and unnamed_idx < len(unnamed_supplements):
            supplement = unnamed_supplements[unnamed_idx]
            unnamed_idx += 1

        apply_result_supplement(evaluation_result, supplement)