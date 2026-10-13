"""Shared constants for the Last Translation Benchmark -> Every Eval Ever converter.

Everything that is *metadata about LTB itself* (as opposed to per-record logic)
lives here, so it can be reviewed and corrected in one place.

Sources for these values:
  - LTB README + paper:  https://arxiv.org/abs/2609.04173
  - LTB repo:            https://github.com/zouharvi/last-translation-benchmark
  - LTB data on HF:      https://hf.co/datasets/zouhar/last-translation-benchmark
  - Verification prompt: server/utils.py::get_prompt_verify
  - Model id maps:       scripts/20b-translate_by_extra_models.py (API-served LLMs),
                         scripts/22-translate_by_MTs.py (dedicated MT systems),
                         server/routers.py::MODEL_LIBRARY (live translate button)
"""

# ---------------------------------------------------------------------------
# Schema versions
# ---------------------------------------------------------------------------
# Bump these together with the schemas in evaleval/every_eval_ever.
# 0.3.0 is a strict superset of 0.2.2 for everything this converter emits,
# so output validates against both.
SCHEMA_VERSION = "0.3.0"
INSTANCE_SCHEMA_VERSION = f"instance_level_eval_{SCHEMA_VERSION}"

# Folder name under data/ in the EEE datastore.
BENCHMARK_NAME = "last-translation-benchmark"

# Dataset release being converted.
DATASET_VERSION = "LTBv1"

# Tags in v1.json that we emit aggregate results for, in order.
# The first one is the "primary" tag: instance-level rows are linked to its
# overall result, and per-language-pair results are computed for it.
TAGS = ["LTBv1", "LTBv1-eval"]

LTB_REPO_URL = "https://github.com/zouharvi/last-translation-benchmark"
LTB_SITE_URL = "https://last-translation-benchmark.vilda.net"
LTB_HF_REPO = "zouhar/last-translation-benchmark"
LTB_PAPER_URL = "https://arxiv.org/abs/2609.04173"
LTB_JSON_URL = "http://last-translation-benchmark.vilda.net/LTBv1.json"

# ---------------------------------------------------------------------------
# source_metadata
# ---------------------------------------------------------------------------
# source_type is "evaluation_run" (not "documentation"): v1.json carries the raw
# per-item model outputs and their per-item verification verdicts, not just an
# aggregate leaderboard number.
SOURCE_METADATA = {
    "source_name": "Last Translation Benchmark",
    "source_type": "evaluation_run",
    "source_organization_name": "Last Translation Benchmark",
    "source_organization_url": LTB_SITE_URL,
    "evaluator_relationship": "third_party",
    "additional_details": {
        "paper": LTB_PAPER_URL,
        "repository": LTB_REPO_URL,
        "dataset_version": DATASET_VERSION,
        "dataset_url": LTB_JSON_URL,
        "data_license": "CC BY 4.0",
        "code_license": "MIT",
        "cutoff": "accepted contributions prior to 2026-09-01",
    },
}

# ---------------------------------------------------------------------------
# eval_library
# ---------------------------------------------------------------------------
EVAL_LIBRARY = {
    "name": "last-translation-benchmark",
    "version": DATASET_VERSION,
    "additional_details": {
        "repository": LTB_REPO_URL,
        "scoring_script": "scripts/41-score_leaderboard.py",
        "release_script": "scripts/03a-prepare_release.py",
    },
}

# ---------------------------------------------------------------------------
# LLM judge used for verification
# ---------------------------------------------------------------------------
# v1.json stores `translations[].verified`, which is
# `verified_extra["Gemini 3.1 Pro"]` -- one boolean per verification rule,
# produced by the prompt below (server/utils.py::get_prompt_verify).
JUDGE_DISPLAY_NAME = "Gemini 3.1 Pro"
JUDGE_MODEL_INFO = {
    "name": "Gemini 3.1 Pro",
    "id": "google/gemini-3.1-pro-preview",
    "developer": "Google",
    "inference_platform": "openrouter",
    "additional_details": {
        "deployment_type": "externally_managed",
        "model_availability": "closed_weights",
    },
}

VERIFY_PROMPT_TEMPLATE = (
    "Your goal is to verify whether a translation fulfills a criterion.\n\n"
    "Criterion: {rule}\n\n"
    "Input: {source_text}\n\n"
    "Translation to verify: {translation}\n\n"
    "Output only pass or fail and nothing else."
)

# Appended by LTB when the example carries media.
VERIFY_PROMPT_MEDIA_SUFFIX = "\n\nUse the provided {context_type} as additional context."

# Prompt used by the LLM *judges* (0-100 quality), server/utils.py::get_prompt_judge.
# Distinct from the verifier prompt above: judges rate overall quality, verifiers
# check a specific rule.
JUDGE_PROMPT_TEMPLATE = (
    "Your goal is to evaluate the quality of a translation. Translation quality is "
    "evaluated as follows:\n\n"
    "85-100% (Very Good): Complete meaning transfer; perfectly natural; no or minimal proofreading.\n"
    "65-80% (Good): Near complete transfer, minor inaccuracies; mostly natural, minor awkwardness; "
    "needs light proofreading.\n"
    "45-60% (Acceptable): Main ideas conveyed, noticeable inaccuracies or omissions; uneven "
    "naturalness, awkward phrasing; usable only after substantial revision.\n"
    "25-40% (Borderline): Partial transfer; frequent misinterpretation or omission confusing the "
    "message; often unnatural; requires major rewrite.\n"
    "0-20% (Not acceptable): Violation of meaning; large portions mistranslated, missing, or "
    "incoherent; unusable without complete retranslation.\n\n"
    "Input: {source_text}\n\n"
    "Translation to evaluate: {translation}\n\n"
    "Output a single number between 0 and 100, representing the quality of the translation, "
    "and nothing else."
)

# Prompt the translation systems themselves were given (scripts/20b, and the
# generic branch of scripts/22-translate_by_MTs.py::get_prompt).
TRANSLATE_PROMPT_TEMPLATE = (
    "Translate the following text from {source_lang} to {target_lang}. "
    "Output only the translation and nothing else:\n{source_text}"
)

# GemmaX2 is a completion model and gets a different format (scripts/22).
GEMMAX2_PROMPT_TEMPLATE = (
    "Translate this from {source_lang} to {target_lang}:\n"
    "{source_lang}: {source_text}\n{target_lang}:"
)

# Sentinel: this system's exact input cannot be reconstructed from the public
# release, so `input.formatted` is left null rather than guessed at. The reason
# goes into generation_config.additional_details.prompt_format.
PROMPT_NOT_REPRODUCIBLE = object()

NLLB_PROMPT_NOTE = (
    "no text prompt: NLLB is an encoder-decoder run through transformers with "
    "tokenizer.src_lang set and forced_bos_token_id set to the FLORES-200 target code"
)

# Decoding parameters, from scripts/22-translate_by_MTs.py.
VLLM_GEN_ARGS = {"temperature": 0, "max_tokens": 1024}
COHERE_GEN_ARGS = {"temperature": 0, "max_tokens": 4096}
NLLB_GEN_ARGS = {"max_tokens": 512}

LLM_SCORING = {
    "judges": [{"model_info": JUDGE_MODEL_INFO, "weight": 1.0}],
    "input_prompt": VERIFY_PROMPT_TEMPLATE,
    "aggregation_method": "majority_vote",
    "additional_details": {
        "verdict_parsing": "last whitespace token of the response; 'pass' -> True, 'fail' -> False, "
        "otherwise substring search, else False",
        "example_passes": "an example counts as passed only if ALL of its verification rules pass",
        "note": "LTB also verifies with other judges (verified_extra); the public release "
        "uses Gemini 3.1 Pro and only that verdict is converted here",
    },
}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
METRIC_PASS_RATE = {
    "metric_id": "ltb.verification_pass_rate",
    "metric_name": "Verification pass rate",
    "metric_kind": "pass_rate",
    "metric_unit": "proportion",
    "lower_is_better": False,
    "score_type": "continuous",
    "min_score": 0.0,
    "max_score": 1.0,
}

METRIC_RULE_PASS_RATE = {
    "metric_id": "ltb.rule_pass_rate",
    "metric_name": "Verification rule pass rate",
    "metric_kind": "pass_rate",
    "metric_unit": "proportion",
    "lower_is_better": False,
    "score_type": "continuous",
    "min_score": 0.0,
    "max_score": 1.0,
}

# ---------------------------------------------------------------------------
# Model registry: LTB display name -> EEE model_info
# ---------------------------------------------------------------------------
# `id` drives the datastore path: data/{BENCHMARK_NAME}/{id.split("/")[0]}/{id.split("/")[1]}/
#
# Authoritative sources in the LTB repo, cross-checked against each other:
#   - scripts/20b-translate_by_extra_models.py  (API-served LLMs)
#   - scripts/22-translate_by_MTs.py            (dedicated MT systems)
#   - server/routers.py::MODEL_LIBRARY          (live "translate" button, incl. audio)
#
# Two entries stay flagged `needs_review`: Google Translate and Lara are commercial
# APIs with no HuggingFace repo, so their ids are stable stand-ins rather than
# resolvable model references.
#
# deployment_type:    self_deployed | externally_managed | unknown
# model_availability: open_weights  | closed_weights     | unknown
_R = "needs_review"

MODEL_REGISTRY: dict[str, dict] = {
    # --- LLMs served through OpenRouter (authoritative ids from scripts/20b) ---
    "Gemma 4": {
        "id": "google/gemma-4-31b-it", "developer": "Google",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "open_weights",
    },
    "Llama 4 Maverick": {
        "id": "meta-llama/llama-4-maverick", "developer": "Meta",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "open_weights",
    },
    "GPT-5.4 Mini": {
        "id": "openai/gpt-5.4-mini", "developer": "OpenAI",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "GPT-5.6 Sol": {
        "id": "openai/gpt-5.6-sol", "developer": "OpenAI",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "GPT-5.6 Luna": {
        "id": "openai/gpt-5.6-luna", "developer": "OpenAI",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "GPT-5.6 Terra": {
        "id": "openai/gpt-5.6-terra", "developer": "OpenAI",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "gpt-oss-20b": {
        "id": "openai/gpt-oss-20b", "developer": "OpenAI",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "open_weights",
    },
    "Claude Haiku 4.5": {
        "id": "anthropic/claude-haiku-4.5", "developer": "Anthropic",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "Claude Sonnet 4.5": {
        "id": "anthropic/claude-sonnet-4.5", "developer": "Anthropic",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "Command A": {
        "id": "cohere/command-a", "developer": "Cohere",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "open_weights",
    },
    "Command A+": {
        "id": "cohere/command-a-plus-05-2026", "developer": "Cohere",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "TinyAya Global": {
        "id": "cohere/tiny-aya-global", "developer": "Cohere",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "open_weights",
    },
    "Qwen 3.7 Plus": {
        "id": "qwen/qwen3.7-plus", "developer": "Alibaba",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "Qwen 3.7 Flash": {
        "id": "qwen/qwen3.7-flash", "developer": "Alibaba",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "Gemini 3.1 Pro": {
        "id": "google/gemini-3.1-pro-preview", "developer": "Google",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "Gemini 3.5 Flash Lite": {
        "id": "google/gemini-3.5-flash-lite", "developer": "Google",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "Gemini 2.5 Flash": {
        "id": "google/gemini-2.5-flash", "developer": "Google",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "Gemini 2.5 Pro": {
        "id": "google/gemini-2.5-pro", "developer": "Google",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "Gemini 3.5 Flash": {
        "id": "google/gemini-3.5-flash", "developer": "Google",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "Gemini 3.7 Flash": {
        "id": "google/gemini-3.7-flash", "developer": "Google",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "GPT-4.1 Nano": {
        "id": "openai/gpt-4.1-nano", "developer": "OpenAI",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "Llama 4 Scout": {
        "id": "meta-llama/llama-4-scout", "developer": "Meta",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "open_weights",
    },
    "Kimi K3": {
        "id": "moonshotai/kimi-k3", "developer": "Moonshot AI",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "open_weights",
    },
    "Nemotron 3 Ultra": {
        "id": "nvidia/nemotron-3-ultra-550b-a55b", "developer": "NVIDIA",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "open_weights",
    },
    "Deepseek V4 Pro": {
        "id": "deepseek/deepseek-v4-pro", "developer": "DeepSeek",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "open_weights",
    },

    # --- Commercial translation APIs ---
    "Google Translate": {
        "id": "google/google-translate", "developer": "Google",
        "platform": "google-cloud-translation", "deployment": "externally_managed",
        "availability": "closed_weights",
        _R: "no HF repo; id is a stable stand-in for the Cloud Translation API",
    },
    "Lara": {
        "id": "translated/lara", "developer": "Translated",
        "platform": "translated-lara-api", "deployment": "externally_managed",
        "availability": "closed_weights",
        _R: "no HF repo; id is a stable stand-in for the Lara API",
    },

    # --- Speech-capable models (used on the audio/video examples) ---
    "GPT Audio": {
        "id": "openai/gpt-audio", "developer": "OpenAI",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "GPT Audio Mini": {
        "id": "openai/gpt-audio-mini", "developer": "OpenAI",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "Voxtral Small": {
        "id": "mistralai/voxtral-small-24b-2507", "developer": "Mistral AI",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "open_weights",
    },

    # --- In server/routers.py::MODEL_LIBRARY but absent from LTBv1; kept so a
    #     future release converts without touching the registry ---
    "GPT-6 Astra": {
        "id": "openai/gpt-6-astra", "developer": "OpenAI",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },
    "Gemini 3.8 Flash": {
        "id": "google/gemini-3.8-flash", "developer": "Google",
        "platform": "openrouter", "deployment": "externally_managed", "availability": "closed_weights",
    },

    # --- Dedicated MT models ---
    # Ids, serving stack, decoding parameters and prompt formats all come from
    # scripts/22-translate_by_MTs.py (confirmed by @zouharvi on issue #249), and
    # every HF id below was checked to resolve on the Hub.
    #
    # `engine` (local serving) is used instead of `platform` per the EEE
    # contributor guide: inference_platform for remote APIs, inference_engine for
    # local runs.
    "TranslateGemma": {
        "id": "google/translategemma-27b-it", "developer": "Google",
        "engine": "vllm", "deployment": "self_deployed", "availability": "open_weights",
        "gen_args": VLLM_GEN_ARGS,
        "prompt": PROMPT_NOT_REPRODUCIBLE,
        "prompt_note": "rendered with the model's own chat template, passing "
                       "source_lang_code/target_lang_code rather than a text prompt",
    },
    "Tower+": {
        "id": "Unbabel/Tower-Plus-9B", "developer": "Unbabel",
        "engine": "vllm", "deployment": "self_deployed", "availability": "open_weights",
        "gen_args": VLLM_GEN_ARGS,
    },
    "HY-MT2": {
        "id": "tencent/Hy-MT2-30B-A3B", "developer": "Tencent",
        "engine": "vllm", "deployment": "self_deployed", "availability": "open_weights",
        "gen_args": VLLM_GEN_ARGS,
    },
    "Seed-X-PPO-7B": {
        "id": "ByteDance-Seed/Seed-X-PPO-7B", "developer": "ByteDance",
        "engine": "vllm", "deployment": "self_deployed", "availability": "open_weights",
        "gen_args": VLLM_GEN_ARGS,
        "prompt": PROMPT_NOT_REPRODUCIBLE,
        "prompt_note": "completion prompt 'Translate the following {source_lang} sentence into "
                       "{target_lang}:\\n{source_text} <{target_iso639_1}>'; the ISO 639-1 tag is "
                       "resolved from a lookup table not present in the public release",
    },
    "GemmaX2-28-9B": {
        "id": "ModelSpace/GemmaX2-28-9B-v0.1", "developer": "ModelSpace",
        "engine": "vllm", "deployment": "self_deployed", "availability": "open_weights",
        "gen_args": VLLM_GEN_ARGS,
        "prompt": GEMMAX2_PROMPT_TEMPLATE,
    },
    "NLLB 3.3B": {
        "id": "facebook/nllb-200-3.3B", "developer": "Meta",
        "engine": "transformers", "deployment": "self_deployed", "availability": "open_weights",
        "gen_args": NLLB_GEN_ARGS,
        "prompt": PROMPT_NOT_REPRODUCIBLE,
        "prompt_note": NLLB_PROMPT_NOTE,
    },
    "NLLB 54B": {
        "id": "facebook/nllb-moe-54b", "developer": "Meta",
        "engine": "transformers", "deployment": "self_deployed", "availability": "open_weights",
        "gen_args": NLLB_GEN_ARGS,
        "prompt": PROMPT_NOT_REPRODUCIBLE,
        "prompt_note": NLLB_PROMPT_NOTE,
    },
    "Command A Translate": {
        "id": "cohere/command-a-translate-08-2025", "developer": "Cohere",
        "platform": "cohere", "deployment": "externally_managed", "availability": "closed_weights",
        "gen_args": COHERE_GEN_ARGS,
    },
    "QwenMT": {
        "id": "qwen/qwen-mt-plus", "developer": "Alibaba",
        "platform": "alibaba-dashscope", "deployment": "externally_managed",
        "availability": "closed_weights",
        "prompt": PROMPT_NOT_REPRODUCIBLE,
        "prompt_note": "DashScope translation endpoint: the source text is sent as the message "
                       "content with source_lang/target_lang passed in translation_options, so "
                       "there is no natural-language prompt",
    },
    # NOTE: "Cohere Command A" is not a registry entry. It is the same model as
    # "Command A" run twice, and MODEL_ALIASES folds it into that one. See below.

    # --- Human reference (only emitted with --include-human) ---
    "human": {
        "id": "ltb/human-reference", "developer": "Last Translation Benchmark",
        "platform": "human", "deployment": "unknown", "availability": "unknown",
        _R: "not a model: human contributor translations, kept as a reference row",
    },
}

NEEDS_REVIEW_KEY = _R

# ---------------------------------------------------------------------------
# Aliases: two LTB display names, one model run twice
# ---------------------------------------------------------------------------
# {alias display name: canonical display name}. The alias's examples are folded
# into the canonical record, and where both scored the same example the
# canonical one wins.
#
# `Cohere Command A` is `Command A` served directly by Cohere instead of through
# OpenRouter, confirmed by the LTB authors on issue #249. It is the same model
# run twice, so it becomes one record rather than two. Nothing is lost by
# preferring `Command A`: all 2272 `Cohere Command A` examples are already inside
# the 3244 `Command A` ones, so the merged record is exactly the `Command A` run.
#
# A display name in here must NOT also be in MODEL_REGISTRY; the alias is
# resolved before the registry is consulted.
MODEL_ALIASES: dict[str, str] = {
    "Cohere Command A": "Command A",
}

# Translations whose `model` value starts with one of these is dropped, mirroring
# scripts/03a-prepare_release.py.
MODEL_PREFIX_BLOCKLIST = ("SKIP: ", "PRIVILEGE-")
