import hashlib
from datetime import datetime
from huggingface_hub import HfApi
from typing import Dict

def convert_timestamp_to_unix_format(timestamp: str) -> str:
    dt = datetime.fromisoformat(timestamp)
    return str(dt.timestamp())

def get_current_unix_timestamp() -> str:
    return str(datetime.now().timestamp())

def get_model_organization_info(model_base_name: str) -> Dict:
    """
    Searches the Hugging Face Hub for a model based on its base name 
    and attempts to find the organization that published the most relevant/original version.

    Args:
        model_base_name: The model name without an organization (e.g., 'deepseek-coder-6.7b-base').

    Returns:
        A dictionary containing the best-guess organization and full repository ID, 
        or an error message.
    """
    
    api = HfApi()
    
    try:
        models = api.list_models(
            search=model_base_name,
            sort="downloads",
            direction=-1,
            limit=50
        )
        models_list = list(models)
    except Exception as e:
        return f"Failed to connect to Hugging Face Hub: {e}"

    if not models_list:
        return 'not_found'

    # Heuristic to find the 'Original' Organization:
    # The original model is usually the one with the shortest repo_id 
    # that includes the base model name (e.g., 'deepseek-ai/deepseek-coder-6.7b-base').
    # We also prioritize the one with the highest downloads.
    
    best_match = models_list[0] # Start with the most downloaded model

    for model in models_list:
        repo_id = model.modelId
        
        parts = repo_id.split('/')
        if len(parts) != 2:
             continue
        
        org, name = parts
        
        # A good heuristic: the model name part (name) should exactly match the base name,
        # or be a very close variant (e.g., -instruct) with the highest download count.
        if model_base_name in name and name == model_base_name:
            best_match = model
            break

    full_repo_id = best_match.modelId
    organization = full_repo_id.split('/')[0]

    return organization

def sha256_file(path, chunk_size=8192):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def sha256_string(text: str, chunk_size=8192):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()