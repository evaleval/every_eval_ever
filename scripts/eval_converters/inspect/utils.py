import hashlib
import re

from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Type, List

from eval_types import (
    InferenceEngine,
    ModelInfo
)
from scripts.eval_converters.common.utils import get_model_organization_info


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
    date = date_match.group(1) if date_match else "00000000"

    return f"{base}@{date}"

class ClosedApiHandler(ModelPathHandler):
    """Handles paths for closed API providers like OpenAI, Anthropic, Google, etc."""
    def handle(self) -> ModelInfo:
        developer = self.parts[0]
        inference_platform = self.parts[0]
        model_id = self.model_path
        
        # Special handling for Anthropic models running on cloud platforms (Vertex/Bedrock)
        if self.model_path.startswith('anthropic/vertex') or self.model_path.startswith('anthropic/bedrock'):
            if len(self.parts) >= 3:
                # e.g., anthropic/vertex/claude-3-5-sonnet-v2@20241022
                developer = self.parts[0]
                inference_platform = self.parts[1]
                model_id = f'{developer}/{normalize_claude_model_name(self.parts[2])}'

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
            inference_platform=inference_platform
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
            id=f"{developer}/{model}" if developer else model,
            developer=developer,
            inference_platform='bedrock'
        )

class AzureAiParser:

    @staticmethod
    def parse(model_name: str) -> ModelInfo:
        parts = model_name.split('/')
        inference_platform = parts[0]
        base_model_name = parts[-1]
        
        inferred_org_name = get_model_organization_info(base_model_name)
        developer = inferred_org_name if inferred_org_name and inferred_org_name != 'not_found' else inference_platform
        
        return ModelInfo(
            name=model_name,
            id=f'{developer}/{base_model_name}', # Corrected 'id' logic
            developer=developer,
            inference_platform=inference_platform
        )

class CloudApiHandler(ModelPathHandler):
    """Handles paths for cloud API providers like AWS Bedrock and Azure AI."""
    def handle(self) -> ModelInfo:
        if self.model_path.startswith('bedrock'):
            return BedrockParser.parse(self.model_path) 
        
        elif self.model_path.startswith('azureai'):
            return AzureAiParser.parse(self.model_path)
            
        return ModelInfo(name=self.model_path, id=self.model_path, developer='unknown', inference_platform='cloud_api')

class HostedOpenHandler(ModelPathHandler):
    """Handles paths for hosted open models (Together, Groq, Fireworks, etc.)."""

    # Model name to developer map for Sambanova
    SAMBANOVA_DEV_MAP = {
        'llama': 'meta-llama', 'gpt': 'openai', 'qwen': 'qwen', 
        'swallow': 'tokyotech-llm', 'allam': 'humain-ai', 
        'mistral': 'mistral', 'deepseek': 'deepseek-ai'
    }

    def handle(self) -> ModelInfo:
        inference_platform = self.parts[0]
        model_id = self.model_path
        developer = 'unknown'
        
        path_lower = self.model_path.lower()

        # Group 1: Platform/Developer/Model format
        platforms = ['together', 'cf', 'openrouter', 'openai-api', 'hf-inference-providers']
        if any(path_lower.startswith(p) for p in platforms):
            if len(self.parts) >= 3:
                developer = self.parts[1]
                model_id = f'{developer}/{self.parts[2]}'
            else:
                developer = inference_platform # Fallback

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
            developer = inferred_org_name if inferred_org_name != 'not_found' else 'unknown'
            model_id = f'{developer}/{model_name}'
        
        if developer == 'unknown':
             if len(self.parts) >= 2:
                developer = self.parts[1]
                model_id = f'{developer}/{self.parts[-1]}'

        return ModelInfo(
            name=self.model_path,
            id=model_id,
            developer=developer,
            inference_platform=inference_platform
        )

class InferenceEngineHandler(ModelPathHandler):
    """Handles paths for inference engines (vLLM, Ollama, SGLang, etc.)."""
    def handle(self) -> ModelInfo:
        inference_engine = self.parts[0]
        model_id = self.model_path
        developer = 'unknown' # Default value

        # Group 1: /Engine/Developer/Model format (e.g., vllm/meta-llama/Llama-2-7b-chat)
        if any(self.model_path.startswith(e) for e in ['vllm', 'sglang', 'hf']):
            if len(self.parts) >= 3:
                developer = self.parts[1]
                model_id = f'{self.parts[1]}/{self.parts[2]}'
            else:
                developer = inference_engine
        
        # Group 2: Ollama and Llama-cpp-python format (e.g., ollama/llama2:7b)
        elif any(self.model_path.startswith(e) for e in ['ollama', 'llama-cpp-python']):
            if len(self.parts) >= 2:
                developer = self.parts[0]
                model_name = self.parts[1].replace(':', '-')
                model_id = f'{self.parts[0]}/{model_name}'
        
        return ModelInfo(
            name=self.model_path,
            id=model_id,
            developer=developer,
            inference_engine=InferenceEngine(
                name=inference_engine # TODO add version if possible
            )
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
    'sglang': InferenceEngineHandler
}

PROVIDER_PREFIXES: List[str] = list(MODEL_HANDLER_MAP.keys())

def extract_model_info_from_model_path(model_path: str) -> ModelInfo:
    """
    Infers the ModelInfo by dispatching the model_path to the appropriate handler.
    
    This function is now simple and focuses only on dispatching.
    To add a new provider/engine, you only need to update the MODEL_HANDLER_MAP 
    and create a corresponding Handler class.
    """
    handler_class = None
    for prefix in PROVIDER_PREFIXES:
        model_provider = model_path.split('/')[0]
        if model_provider.lower() == prefix:
            handler_class = MODEL_HANDLER_MAP[prefix]
            break

    if handler_class:
        handler = handler_class(model_path)
        return handler.handle()

    # Fallback
    return ModelInfo(
        name=model_path, 
        id=model_path, 
        developer='unknown', 
        inference_platform='unknown'
    )

def sha256_file(path, chunk_size=8192):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def sha256_string(text: str, chunk_size=8192):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def save_to_file(path: str, obj: BaseModel) -> bool:
    json_str = obj.model_dump_json(indent=4, exclude_none=True)

    obj_path = Path(path)
    obj_path.mkdir(parents=True, exist_ok=True)

    with open(obj_path, 'w') as json_file:
        json_file.write(json_str)