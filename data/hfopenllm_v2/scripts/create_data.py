import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional


def evaluation_description(evaluation_name: str) -> str:
    if evaluation_name == "MATH Level 5":
        return "Exact Match on MATH Level 5"
    return f"Accuracy on {evaluation_name}"


def extract_company_name(model_name: str) -> Optional[str]:
    """
    Extract company name from model name based on known patterns.
    Only applies to original models, not finetuned versions.

    Args:
        model_name: Full model name (e.g., "meta-llama/Llama-3-8B")

    Returns:
        Company name if recognized pattern found and it's an original model, else None
    """
    # Company to model name patterns mapping
    company_patterns = {
        "meta": ["llama"],
        "google": ["gemini", "gemma"],
        "openai": ["gpt"],
        "anthropic": ["claude"],
        "alibaba": ["qwen"],
        "microsoft": ["phi"],
        "mistral": ["mistral"],
    }

    model_name_lower = model_name.lower()

    # Check if this is a finetuned model (contains typical finetune indicators)
    finetune_indicators = [
        "-dpo",
        "-sft",
        "-instruct",
        "-chat",
        "-rlhf",
        "-tune",
        "finetuned",
        "ft-",
    ]

    # If it has a slash, check if the part after slash contains finetune indicators
    if "/" in model_name:
        model_part = model_name.split("/", 1)[1].lower()
        # Check for finetune indicators, but exclude the base model names themselves
        for indicator in finetune_indicators:
            if indicator in model_part:
                # Check if it's not part of an official model name
                is_official_variant = any(
                    pattern in model_part
                    for patterns in company_patterns.values()
                    for pattern in patterns
                )
                # If it has finetune indicators and looks like a third-party finetune, return None
                if not is_official_variant or any(
                    char.isdigit() and indicator in model_part for char in model_part
                ):
                    return None

    # Check for company patterns in the model name
    for company, patterns in company_patterns.items():
        for pattern in patterns:
            if pattern in model_name_lower:
                return company

    return None


def convert_to_evalhub_format(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a model evaluation dict to evalHub format.

    Args:
        input_data: Dict containing model and evaluation information

    Returns:
        Dict in evalHub format
    """
    model_name = input_data["model"]["name"]

    # Create evaluation results list
    evaluation_results = []

    # Map evaluations to the new format
    evaluation_mapping = {
        "ifeval": "IFEval",
        "bbh": "BBH",
        "math": "MATH Level 5",
        "gpqa": "GPQA",
        "musr": "MUSR",
        "mmlu_pro": "MMLU-PRO",
    }

    for eval_key, eval_data in input_data.get("evaluations", {}).items():
        evaluation_result = {
            "evaluation_name": eval_data.get(
                "name", evaluation_mapping.get(eval_key, eval_key)
            ),
            "metric_config": {
                "evaluation_description": evaluation_description(
                    eval_data.get("name", eval_key)
                ),
                "lower_is_better": False,
                "score_type": "continuous",
                "min_score": 0,
                "max_score": 1,
            },
            "score_details": {"score": eval_data.get("value", 0.0)},
        }
        evaluation_results.append(evaluation_result)

    # Create additional_details dict
    additional_details = {}
    if "precision" in input_data["model"]:
        additional_details["precision"] = input_data["model"]["precision"]
    if "architecture" in input_data["model"]:
        additional_details["architecture"] = input_data["model"]["architecture"]
    if "params_billions" in input_data.get("metadata", {}):
        additional_details["params_billions"] = input_data["metadata"][
            "params_billions"
        ]

    # Extract developer name from model name
    # First, try to extract company name if it's an original model
    company_name = extract_company_name(model_name)

    if company_name:
        developer = company_name
    elif "/" in model_name:
        developer = model_name.split("/")[0]
    else:
        developer = "Unknown"

    # Create the evalHub format
    output_data = {
        "schema_version": "0.0.1",
        "evaluation_id": f"hfopenllm_v2/{model_name.replace('/', '_')}/{time.time()}",
        "retrieved_timestamp": time.time(),
        "source_data_urls": [
            "https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted"
        ],
        "evaluation_source": {
            "evaluation_source_name": "HF Open LLM v2",
            "evaluation_source_type": "leaderboard",
        },
        "source_metadata": {
            "source_organization_name": "Hugging Face",
            "evaluator_relationship": "third_party",
        },
        "model_info": {
            "name": model_name,
            "developer": developer,
            "inference_platform": "unknown",
        },
        "evaluation_results": evaluation_results,
    }

    # Add additional_details only if it has content
    if additional_details:
        output_data["additional_details"] = additional_details

    return output_data


def process_models(
    models_data: List[Dict[str, Any]], output_dir: str = "/Users/random/every_eval_ever/data/hfopenllm_v2"
):
    """
    Process a list of model evaluation dicts and save them in evalHub format.
    Follows the structure: {leaderboard_name}/{developer_name}/{model_name}/{uuid}.json

    Args:
        models_data: List of dicts containing model and evaluation information
        output_dir: Base directory (should be the leaderboard name folder)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_data in models_data:
        try:
            # Convert to evalHub format
            converted_data = convert_to_evalhub_format(model_data)

            # Get model name and parse developer/model
            model_name = model_data["model"]["name"]

            # Extract developer (will use company name if applicable)
            company_name = extract_company_name(model_name)

            if company_name:
                developer = company_name
                model = model_name  # Keep full model name
            elif "/" in model_name:
                developer, model = model_name.split("/", 1)
            else:
                developer = "Unknown"
                model = model_name

            # Create folder structure: {leaderboard}/{developer}/{model}/
            model_dir = output_path / developer / model
            model_dir.mkdir(parents=True, exist_ok=True)

            # Generate UUID for the filename
            file_uuid = str(uuid.uuid4())
            output_file = model_dir / f"{file_uuid}.json"

            # Save to file
            with open(output_file, "w") as f:
                json.dump(converted_data, f, indent=2)

            print(f"✓ Converted: {model_name} -> {output_file}")

        except Exception as e:
            print(
                f"✗ Error processing {model_data.get('model', {}).get('name', 'unknown')}: {e}"
            )


# Example usage
if __name__ == "__main__":
    # Example: Single model conversion
    example_model = {
        "id": "0-hero/Matter-0.2-7B-DPO_bfloat16_26a66f0d862e2024ce4ad0a09c37052ac36e8af6_True",
        "model": {
            "name": "0-hero/Matter-0.2-7B-DPO",
            "sha": "26a66f0d862e2024ce4ad0a09c37052ac36e8af6",
            "precision": "bfloat16",
            "type": "chatmodels",
            "weight_type": "Original",
            "architecture": "MistralForCausalLM",
            "average_score": 8.90636130175029,
            "has_chat_template": True,
        },
        "evaluations": {
            "ifeval": {
                "name": "IFEval",
                "value": 0.3302792147058693,
                "normalized_score": 33.02792147058693,
            },
            "bbh": {
                "name": "BBH",
                "value": 0.3596254301656297,
                "normalized_score": 10.055525080241035,
            },
            "math": {
                "name": "MATH Level 5",
                "value": 0.014350453172205438,
                "normalized_score": 1.4350453172205437,
            },
            "gpqa": {
                "name": "GPQA",
                "value": 0.25922818791946306,
                "normalized_score": 1.230425055928408,
            },
            "musr": {
                "name": "MUSR",
                "value": 0.381375,
                "normalized_score": 5.871874999999999,
            },
            "mmlu_pro": {
                "name": "MMLU-PRO",
                "value": 0.1163563829787234,
                "normalized_score": 1.8173758865248217,
            },
        },
        "features": {
            "is_not_available_on_hub": True,
            "is_merged": False,
            "is_moe": False,
            "is_flagged": False,
            "is_official_provider": False,
        },
        "metadata": {
            "upload_date": "2024-04-13",
            "submission_date": "2024-08-05",
            "generation": 0,
            "base_model": "0-hero/Matter-0.2-7B-DPO",
            "hub_license": "apache-2.0",
            "hub_hearts": 3,
            "params_billions": 7.242,
            "co2_cost": 1.219174164123715,
        },
    }

    # Process single model
    data_path = "/Users/random/every_eval_ever/data/formatted"
    all_models = []
    with open(data_path, "r") as f:
        all_models = json.load(f)

    process_models(all_models)
    # process_models([example_model])

    # Or load from a JSON file containing a list of models:
    # with open('models_data.json', 'r') as f:
    #     models_list = json.load(f)
    # process_models(models_list)
