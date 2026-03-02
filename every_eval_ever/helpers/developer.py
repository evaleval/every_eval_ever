"""Unified developer/organization extraction from model names."""

from typing import Optional

DEVELOPER_PATTERNS = {
    # OpenAI models
    "gpt": "openai",
    "text-davinci": "openai",
    "text-curie": "openai",
    "text-babbage": "openai",
    "text-ada": "openai",
    "davinci": "openai",
    "curie": "openai",
    "babbage": "openai",
    "ada": "openai",
    "o1": "openai",
    "o3": "openai",

    # Anthropic models
    "claude": "anthropic",

    # Google models
    "gemini": "google",
    "gemma": "google",
    "palm": "google",
    "t5": "google",
    "ul2": "google",
    "text-bison": "google",
    "text-unicorn": "google",

    # Meta models
    "llama": "meta",
    "opt": "meta",

    # Mistral models
    "mistral": "mistralai",
    "mixtral": "mistralai",

    # Alibaba models
    "qwen": "alibaba",

    # Microsoft models
    "phi": "microsoft",
    "tnlg": "microsoft",

    # AI21 models
    "j1": "ai21",
    "j2": "ai21",
    "jamba": "ai21",
    "jurassic": "ai21",

    # Cohere models
    "command": "cohere",
    "cohere": "cohere",

    # Other providers
    "falcon": "tiiuae",
    "bloom": "bigscience",
    "t0pp": "bigscience",
    "pythia": "eleutherai",
    "gpt-j": "eleutherai",
    "gpt-neox": "eleutherai",
    "luminous": "aleph-alpha",
    "mpt": "mosaicml",
    "redpajama": "together",
    "vicuna": "lmsys",
    "alpaca": "stanford",
    "palmyra": "writer",
    "instructpalmyra": "writer",
    "yalm": "yandex",
    "glm": "zhipu-ai",
    "deepseek": "deepseek",
    "yi": "01-ai",
    "solar": "upstage",
    "arctic": "snowflake",
    "dbrx": "databricks",
    "olmo": "allenai",
    "nova": "amazon",
    "grok": "xai",
}


def get_developer(model_name: str) -> str:
    """
    Extract developer/organization name from a model name.

    Uses a two-step approach:
    1. If model_name contains '/', use the prefix as the developer
    2. Otherwise, pattern match against known model families

    Args:
        model_name: The model name (e.g., "meta-llama/Llama-3-8B" or "gpt-4")

    Returns:
        Developer name (lowercase), or "unknown" if not recognized

    Examples:
        >>> get_developer("meta-llama/Llama-3-8B")
        "meta-llama"
        >>> get_developer("gpt-4-turbo")
        "openai"
        >>> get_developer("claude-3-opus")
        "anthropic"
        >>> get_developer("some-random-model")
        "unknown"
    """
    if not model_name:
        return "unknown"

    # If already has org prefix (e.g., "meta-llama/Llama-3-8B"), use it
    if "/" in model_name:
        return model_name.split("/")[0]

    # Pattern match against known model families
    lower_name = model_name.lower()
    for pattern, developer in DEVELOPER_PATTERNS.items():
        if lower_name.startswith(pattern) or f"-{pattern}" in lower_name:
            return developer

    return "unknown"


def get_model_id(model_name: str, developer: Optional[str] = None) -> str:
    """
    Generate a standardized model ID in the format 'developer/model'.

    Args:
        model_name: The model name
        developer: Optional developer override; if not provided, will be extracted

    Returns:
        Model ID in 'developer/model' format

    Examples:
        >>> get_model_id("Llama-3-8B", "meta")
        "meta/Llama-3-8B"
        >>> get_model_id("openai/gpt-4")
        "openai/gpt-4"
    """
    if "/" in model_name:
        return model_name

    dev = developer or get_developer(model_name)
    return f"{dev}/{model_name}"
