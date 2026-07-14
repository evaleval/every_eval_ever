"""Reviewed model-availability registry for Artificial Analysis identifiers.

Entries in ``VERIFIED_OPEN_WEIGHT_MODELS`` are exact source slugs whose
developers publish model weights through an official repository or release.
The registry is intentionally exact: unlisted siblings remain unknown.
"""

from __future__ import annotations

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN_DEPLOYMENT = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'
UNKNOWN_AVAILABILITY = 'unknown'
INFERENCE_PLATFORM = 'unknown'
INFERENCE_ENGINE_NAME = 'unknown'
INFERENCE_ENGINE_VERSION = 'unknown'

# First-party evidence for family rules and exact reviewed entries. These
# sources establish weight availability, but not which endpoint Artificial
# Analysis evaluated. Open-weight entries therefore retain unknown deployment.
MODEL_AVAILABILITY_SOURCES = {
    'anthropic_claude': (
        'https://docs.anthropic.com/en/docs/about-claude/models/overview'
    ),
    'amazon_nova': (
        'https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html'
    ),
    'openai': 'https://openai.com/index/introducing-gpt-oss/',
    'google_gemini_gemma': 'https://ai.google.dev/',
    'meta_llama': 'https://ai.meta.com/blog/meta-llama-3-1/',
    'meta_muse': (
        'https://about.fb.com/news/2026/04/'
        'introducing-muse-spark-meta-superintelligence-labs/'
    ),
    'xai_grok': 'https://docs.x.ai/developers/models',
    'alibaba_qwen': 'https://huggingface.co/Qwen/models',
    'minimax': 'https://huggingface.co/MiniMaxAI/models',
    'cohere_command': (
        'https://huggingface.co/CohereLabs/c4ai-command-a-03-2025'
    ),
    'cohere_tiny_aya': 'https://huggingface.co/CohereLabs/tiny-aya-global',
    'arcee_trinity': ('https://huggingface.co/arcee-ai/Trinity-Large-Thinking'),
    'microsoft_phi': 'https://huggingface.co/microsoft/models',
    'azure_phi3': (
        'https://ai.azure.com/catalog/models/Phi-3-mini-128k-instruct'
    ),
    'nous_hermes': 'https://huggingface.co/NousResearch/models',
    'ai2_olmo_molmo': 'https://huggingface.co/allenai/models',
    'ai2_tulu': 'https://allenai.org/blog/tulu-3-405b',
    'bytedance_seed_oss': (
        'https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct'
    ),
    'deepseek': 'https://huggingface.co/deepseek-ai/models',
    'deepcogito': 'https://huggingface.co/deepcogito/cogito-671b-v2.1',
    'databricks_dbrx': (
        'https://www.databricks.com/blog/'
        'introducing-dbrx-new-state-art-open-llm'
    ),
    'ibm_granite': 'https://huggingface.co/ibm-granite/models',
    'kimi': 'https://huggingface.co/moonshotai/models',
    'liquidai': 'https://huggingface.co/LiquidAI/models',
    'mistral': 'https://huggingface.co/mistralai/models',
    'mbzuai_k2': 'https://huggingface.co/collections/LLM360/k2-v2',
    'nvidia_nemotron': 'https://huggingface.co/nvidia/models',
    'snowflake_arctic': (
        'https://huggingface.co/Snowflake/snowflake-arctic-instruct'
    ),
    'tii_falcon': 'https://huggingface.co/tiiuae/Falcon-H1R-7B',
    'prime_intellect': 'https://huggingface.co/PrimeIntellect/INTELLECT-3',
    'sarvam': 'https://huggingface.co/sarvamai/models',
    'zai_glm': 'https://huggingface.co/zai-org/models',
    'baidu_ernie': 'https://huggingface.co/baidu/models',
    'ai21_jamba': 'https://huggingface.co/ai21labs/models',
    'inclusionai': 'https://huggingface.co/inclusionAI/models',
    'lg_exaone': 'https://huggingface.co/LGAI-EXAONE/models',
    'longcat': (
        'https://huggingface.co/meituan-longcat/LongCat-Flash-Lite'
    ),
    'motif': (
        'https://huggingface.co/Motif-Technologies/Motif-2-12.7B-Instruct'
    ),
    'nanbeige': 'https://huggingface.co/Nanbeige/Nanbeige4.1-3B',
    'naver': (
        'https://huggingface.co/naver-hyperclovax/'
        'HyperCLOVAX-SEED-Think-32B'
    ),
    'openchat': 'https://huggingface.co/openchat/openchat_3.5',
    'perplexity_r1': (
        'https://www.perplexity.ai/hub/blog/open-sourcing-r1-1776'
    ),
    'reka': 'https://huggingface.co/RekaAI/reka-flash-3',
    'servicenow_apriel': 'https://huggingface.co/ServiceNow-AI/models',
    'stepfun': 'https://huggingface.co/stepfun-ai/models',
    'swiss_apertus': 'https://huggingface.co/swiss-ai/models',
    'trillion': 'https://huggingface.co/trillionlabs/models',
    'upstage_solar': 'https://huggingface.co/upstage/models',
    'xiaomi_mimo': 'https://huggingface.co/XiaomiMiMo/models',
    'api_inception': 'https://docs.inceptionlabs.ai/get-started/models',
    'api_cohere_command_legacy': (
        'https://docs.cohere.com/docs/deprecations'
    ),
    'api_korea_telecom_midm': (
        'https://artificialanalysis.ai/models/mi-dm-k-2-5-pro-dec28'
    ),
    'api_mistral_premier': 'https://docs.mistral.ai/models/overview',
    'api_perplexity_sonar': (
        'https://docs.perplexity.ai/docs/sonar/quickstart'
    ),
    'api_reka_flash': 'https://docs.reka.ai/chat/models',
    'api_upstage_solar': 'https://www.upstage.ai/news/solar-pro-2',
    'api_xiaomi_mimo': 'https://mimo.mi.com/docs/en-US/quick-start/model',
}


VERIFIED_OPEN_WEIGHT_MODELS: dict[str, frozenset[str]] = {
    'ai21-labs': frozenset(
        {
            'jamba-1-5-large',
            'jamba-1-5-mini',
            'jamba-1-6-large',
            'jamba-1-6-mini',
            'jamba-1-7-large',
            'jamba-1-7-mini',
            'jamba-reasoning-3b',
        }
    ),
    'alibaba': frozenset(
        {
            'qwq-32b',
            'qwq-32b-preview',
            'qwen-chat-14b',
            'qwen-chat-72b',
            'qwen1.5-110b-chat',
            'qwen2-5-72b-instruct',
            'qwen2-5-coder-32b-instruct',
            'qwen2-5-coder-7b-instruct',
            'qwen2-72b-instruct',
            'qwen2.5-32b-instruct',
            'qwen3-0.6b-instruct',
            'qwen3-0.6b-instruct-reasoning',
            'qwen3-1.7b-instruct',
            'qwen3-1.7b-instruct-reasoning',
            'qwen3-14b-instruct',
            'qwen3-14b-instruct-reasoning',
            'qwen3-235b-a22b-instruct',
            'qwen3-235b-a22b-instruct-2507',
            'qwen3-235b-a22b-instruct-2507-reasoning',
            'qwen3-235b-a22b-instruct-reasoning',
            'qwen3-30b-a3b-2507',
            'qwen3-30b-a3b-2507-reasoning',
            'qwen3-30b-a3b-instruct',
            'qwen3-30b-a3b-instruct-reasoning',
            'qwen3-32b-instruct',
            'qwen3-32b-instruct-reasoning',
            'qwen3-4b-2507-instruct',
            'qwen3-4b-2507-instruct-reasoning',
            'qwen3-4b-instruct',
            'qwen3-4b-instruct-reasoning',
            'qwen3-5-0-8b',
            'qwen3-5-0-8b-non-reasoning',
            'qwen3-5-122b-a10b',
            'qwen3-5-122b-a10b-non-reasoning',
            'qwen3-5-27b',
            'qwen3-5-27b-non-reasoning',
            'qwen3-5-2b',
            'qwen3-5-2b-non-reasoning',
            'qwen3-5-35b-a3b',
            'qwen3-5-35b-a3b-non-reasoning',
            'qwen3-5-397b-a17b',
            'qwen3-5-397b-a17b-non-reasoning',
            'qwen3-5-4b',
            'qwen3-5-4b-non-reasoning',
            'qwen3-5-9b',
            'qwen3-5-9b-non-reasoning',
            'qwen3-coder-30b-a3b-instruct',
            'qwen3-coder-480b-a35b-instruct',
            'qwen3-coder-next',
            'qwen3-next-80b-a3b-instruct',
            'qwen3-next-80b-a3b-reasoning',
            'qwen3-omni-30b-a3b-instruct',
            'qwen3-omni-30b-a3b-reasoning',
            'qwen3-vl-235b-a22b-instruct',
            'qwen3-vl-235b-a22b-reasoning',
            'qwen3-vl-30b-a3b-instruct',
            'qwen3-vl-30b-a3b-reasoning',
            'qwen3-vl-32b-instruct',
            'qwen3-vl-32b-reasoning',
            'qwen3-vl-4b-instruct',
            'qwen3-vl-4b-reasoning',
            'qwen3-vl-8b-instruct',
            'qwen3-vl-8b-reasoning',
        }
    ),
    'azure': frozenset({'phi-3-mini', 'phi-4-multimodal'}),
    'baidu': frozenset({'ernie-4-5-300b-a47b'}),
    'cohere': frozenset({'tiny-aya-global'}),
    'databricks': frozenset({'dbrx'}),
    'deepseek': frozenset(
        {
            'deepseek-coder-v2',
            'deepseek-coder-v2-lite',
            'deepseek-llm-67b-chat',
            'deepseek-r1',
            'deepseek-r1-0120',
            'deepseek-r1-distill-llama-70b',
            'deepseek-r1-distill-llama-8b',
            'deepseek-r1-distill-qwen-1-5b',
            'deepseek-r1-distill-qwen-14b',
            'deepseek-r1-distill-qwen-32b',
            'deepseek-r1-qwen3-8b',
            'deepseek-v2',
            'deepseek-v2-5',
            'deepseek-v2-5-sep-2024',
            'deepseek-v3',
            'deepseek-v3-0324',
            'deepseek-v3-1',
            'deepseek-v3-1-reasoning',
            'deepseek-v3-1-terminus',
            'deepseek-v3-1-terminus-reasoning',
            'deepseek-v3-2',
            'deepseek-v3-2-0925',
            'deepseek-v3-2-reasoning',
            'deepseek-v3-2-reasoning-0925',
            'deepseek-v3-2-speciale',
        }
    ),
    'deepcogito': frozenset({'cogito-v2-1-reasoning'}),
    'ibm': frozenset(
        {
            'granite-3-3-8b-instruct',
            'granite-4-0-350m',
            'granite-4-0-h-350m',
            'granite-4-0-h-nano-1b',
            'granite-4-0-h-small',
            'granite-4-0-micro',
            'granite-4-0-nano-1b',
        }
    ),
    'inclusionai': frozenset(
        {
            'ling-1t',
            'ling-flash-2-0',
            'ling-mini-2-0',
            'ring-1t',
            'ring-flash-2-0',
        }
    ),
    'kimi': frozenset(
        {
            'kimi-k2',
            'kimi-k2-0905',
            'kimi-k2-5',
            'kimi-k2-5-non-reasoning',
            'kimi-k2-thinking',
            'kimi-linear-48b-a3b-instruct',
        }
    ),
    'liquidai': frozenset(
        {
            'lfm2-1-2b',
            'lfm2-2-6b',
            'lfm2-24b-a2b',
            'lfm2-5-1-2b-instruct',
            'lfm2-5-1-2b-thinking',
            'lfm2-5-vl-1-6b',
            'lfm2-8b-a1b',
        }
    ),
    'lg': frozenset(
        {
            'exaone-4-0-1-2b',
            'exaone-4-0-1-2b-reasoning',
            'exaone-4-0-32b',
            'exaone-4-0-32b-reasoning',
            'k-exaone',
            'k-exaone-non-reasoning',
        }
    ),
    'longcat': frozenset({'longcat-flash-lite'}),
    'mistral': frozenset(
        {
            'devstral-2',
            'devstral-small',
            'devstral-small-2',
            'devstral-small-2505',
            'ministral-3-14b',
            'ministral-3-3b',
            'ministral-3-8b',
            'mistral-7b-instruct',
            'mistral-8x22b-instruct',
            'mistral-large-3',
            'mistral-large-2',
            'mistral-large-2407',
            'magistral-small',
            'magistral-small-2509',
            'mistral-small-3',
            'mistral-small-3-1',
            'mistral-small-3-2',
            'mistral-small-4',
            'mistral-small-4-non-reasoning',
            'mixtral-8x7b-instruct',
            'pixtral-large-2411',
        }
    ),
    'mbzuai': frozenset(
        {'k2-think-v2', 'k2-v2', 'k2-v2-low', 'k2-v2-medium'}
    ),
    'motif-technologies': frozenset({'motif-2-12-7b'}),
    'nanbeige': frozenset({'nanbeige4-1-3b'}),
    'naver': frozenset({'hyperclova-x-seed-think-32b'}),
    'nvidia': frozenset(
        {
            'llama-3-1-nemotron-instruct-70b',
            'llama-3-1-nemotron-nano-4b-reasoning',
            'llama-3-1-nemotron-ultra-253b-v1-reasoning',
            'llama-3-3-nemotron-super-49b',
            'llama-3-3-nemotron-super-49b-reasoning',
            'llama-nemotron-super-49b-v1-5',
            'llama-nemotron-super-49b-v1-5-reasoning',
            'nemotron-cascade-2-30b-a3b',
            'nvidia-nemotron-3-nano-30b-a3b',
            'nvidia-nemotron-3-nano-30b-a3b-reasoning',
            'nvidia-nemotron-3-nano-4b',
            'nvidia-nemotron-3-super-120b-a12b',
            'nvidia-nemotron-nano-12b-v2-vl',
            'nvidia-nemotron-nano-12b-v2-vl-reasoning',
            'nvidia-nemotron-nano-9b-v2',
            'nvidia-nemotron-nano-9b-v2-reasoning',
        }
    ),
    'prime-intellect': frozenset({'intellect-3'}),
    'openchat': frozenset({'openchat-35'}),
    'perplexity': frozenset({'r1-1776'}),
    'reka-ai': frozenset({'reka-flash-3'}),
    'sarvam': frozenset(
        {'sarvam-105b', 'sarvam-30b', 'sarvam-m-reasoning'}
    ),
    'snowflake': frozenset({'arctic-instruct'}),
    'servicenow': frozenset(
        {'apriel-v1-5-15b-thinker', 'apriel-v1-6-15b-thinker'}
    ),
    'stepfun': frozenset({'step-3-5-flash', 'step-3-vl-10b'}),
    'swiss-ai-initiative': frozenset(
        {'apertus-70b-instruct', 'apertus-8b-instruct'}
    ),
    'tii-uae': frozenset({'falcon-h1r-7b'}),
    'trillionlabs': frozenset(
        {'tri-21b-think-preview', 'tri-21b-think-v0-5'}
    ),
    'upstage': frozenset({'solar-mini', 'solar-open-100b-reasoning'}),
    'xiaomi': frozenset(
        {'mimo-v2-0206', 'mimo-v2-flash', 'mimo-v2-flash-reasoning'}
    ),
    'zai': frozenset(
        {
            'glm-4-5-air',
            'glm-4-5v',
            'glm-4-5v-reasoning',
            'glm-4-6',
            'glm-4-6-reasoning',
            'glm-4-6v',
            'glm-4-6v-reasoning',
            'glm-4-7',
            'glm-4-7-flash',
            'glm-4-7-flash-non-reasoning',
            'glm-4-7-non-reasoning',
            'glm-4.5',
            'glm-5',
            'glm-5-1',
            'glm-5-1-non-reasoning',
            'glm-5-non-reasoning',
        }
    ),
}


API_ONLY_CLOSED_WEIGHT_MODELS: dict[str, frozenset[str]] = {
    'alibaba': frozenset(
        {
            'qwen3-5-omni-flash',
            'qwen3-5-omni-plus',
            'qwen3-6-plus',
        }
    ),
    'bytedance_seed': frozenset({'doubao-seed-code'}),
    'cohere': frozenset({'command-r-03-2024'}),
    'inception': frozenset({'mercury-2'}),
    'korea-telecom': frozenset(
        {'mi-dm-k-2-5-pro-dec28', 'midm-250-pro-rsnsft'}
    ),
    'kwaikat': frozenset({'kat-coder-pro-v1', 'kat-coder-pro-v2'}),
    'liquidai': frozenset({'lfm-40b'}),
    'mistral': frozenset(
        {
            'devstral-medium',
            'magistral-medium',
            'magistral-medium-2509',
            'mistral-large',
            'mistral-medium',
            'mistral-medium-3',
            'mistral-medium-3-1',
            'mistral-saba',
            'mistral-small',
            'mistral-small-2402',
        }
    ),
    'perplexity': frozenset(
        {'sonar', 'sonar-pro', 'sonar-reasoning', 'sonar-reasoning-pro'}
    ),
    'reka-ai': frozenset({'reka-flash'}),
    'upstage': frozenset(
        {
            'solar-pro-2',
            'solar-pro-2-preview',
            'solar-pro-2-preview-reasoning',
            'solar-pro-2-reasoning',
            'solar-pro-3',
        }
    ),
    'xai': frozenset({'grok-beta', 'grok-code-fast-1'}),
    'xiaomi': frozenset(
        {'mimo-v2-omni', 'mimo-v2-omni-0327', 'mimo-v2-pro'}
    ),
    'zai': frozenset({'glm-5-turbo', 'glm-5v-turbo'}),
}


def is_verified_open_weight(creator_slug: str, model_slug: str) -> bool:
    """Return whether an exact source identifier has reviewed official weights."""
    return model_slug in VERIFIED_OPEN_WEIGHT_MODELS.get(
        creator_slug, frozenset()
    )


def is_api_only_closed_weight(creator_slug: str, model_slug: str) -> bool:
    """Return whether an exact source identifier is reviewed as API-only."""
    return model_slug in API_ONLY_CLOSED_WEIGHT_MODELS.get(
        creator_slug, frozenset()
    )
