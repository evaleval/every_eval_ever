"""Unified developer/organization extraction from model names."""

from typing import Optional

DEVELOPER_PATTERNS = {
    # OpenAI models
    'gpt': 'openai',
    'text-davinci': 'openai',
    'text-curie': 'openai',
    'text-babbage': 'openai',
    'text-ada': 'openai',
    'davinci': 'openai',
    'curie': 'openai',
    'babbage': 'openai',
    'ada': 'openai',
    'o1': 'openai',
    'o3': 'openai',
    'o4': 'openai',
    # Anthropic models
    'claude': 'anthropic',
    # Google models
    'gemini': 'google',
    'gemma': 'google',
    'palm': 'google',
    't5': 'google',
    'ul2': 'google',
    'text-bison': 'google',
    'text-unicorn': 'google',
    # Meta models
    'llama': 'meta-llama',
    'opt': 'facebook',  # OPT ships under facebook/, not meta-llama/
    # Mistral models
    'mistral': 'mistralai',
    'mixtral': 'mistralai',
    'devstral': 'mistralai',
    'ministral': 'mistralai',
    'codestral': 'mistralai',
    # Alibaba models
    'qwen': 'Qwen',
    # Microsoft models
    'phi': 'microsoft',
    'tnlg': 'microsoft',
    # AI21 models
    'j1': 'ai21labs',
    'j2': 'ai21labs',
    'jamba': 'ai21labs',
    'jurassic': 'ai21labs',
    # Cohere models
    'command': 'CohereForAI',
    'cohere': 'CohereForAI',
    'aya': 'CohereForAI',
    'granite': 'ibm',
    # Other providers
    'falcon': 'tiiuae',
    'bloom': 'bigscience',
    't0pp': 'bigscience',
    'pythia': 'EleutherAI',
    'gpt-j': 'EleutherAI',
    'gpt-neox': 'EleutherAI',
    'luminous': 'Aleph-Alpha',
    'mpt': 'mosaicml',
    'redpajama': 'togethercomputer',
    'vicuna': 'lmsys',
    'alpaca': 'tatsu-lab',
    'palmyra': 'Writer',
    'instructpalmyra': 'Writer',
    'yalm': 'yandex',
    'glm': 'zai-org',
    'deepseek': 'deepseek-ai',
    'yi': '01-ai',
    'solar': 'upstage',
    'arctic': 'Snowflake',
    'dbrx': 'databricks',
    'olmo': 'allenai',
    'nova': 'amazon',
    'grok': 'xai',
    'kimi': 'moonshotai',
    'sarvam': 'sarvamai',
}


def get_developer(model_name: str) -> str:
    """
    Extract developer/organization name from a model name.

    Uses a two-step approach:
    1. If model_name contains '/', use the prefix as the developer
    2. Otherwise, pattern match against known model families

    Args:
        model_name: The model name (e.g., "meta-llama/Llama-3-8B" or "gpt-4")

    Both steps answer with the **publishing namespace**, never the parent
    company, because that is what the datastore's developer folder is (see
    ``datastore_path_components``). So the two agree for one model however a
    source spells it: ``Qwen/Qwen3-32B`` and ``qwen3-32b`` both give ``Qwen``.
    ``test_developer.py`` pins that agreement pair by pair — a new pattern whose
    value is a company rather than a namespace fails there.

    A bare name resolved through the eval-card-registry is still better where an
    adapter can afford the lookup, because this table only knows the families
    listed in it.

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
        return 'unknown'

    # If already has org prefix (e.g., "meta-llama/Llama-3-8B"), use it
    if '/' in model_name:
        return model_name.split('/')[0]

    # Pattern match against known model families
    lower_name = model_name.lower()
    for pattern, developer in DEVELOPER_PATTERNS.items():
        if lower_name.startswith(pattern) or f'-{pattern}' in lower_name:
            return developer

    return 'unknown'


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
    if '/' in model_name:
        return model_name

    dev = developer or get_developer(model_name)
    return f'{dev}/{model_name}'
