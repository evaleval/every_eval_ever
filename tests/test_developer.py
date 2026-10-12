"""`get_developer` must answer with the publishing namespace, not the company.

The datastore's developer folder is `model_info.id`'s prefix
(`datastore-gate.md` §path), so a bare model name has to resolve to the same
namespace its slashed id would carry. When it folded to the parent company
instead, one model reached two directories depending on how a source happened to
spell it, which is #272 and a meaningful share of the 996 publisher directories
in the published datastore.
"""

from __future__ import annotations

import pytest

from every_eval_ever.helpers.developer import (
    DEVELOPER_PATTERNS,
    get_developer,
)

#: (slashed id, bare name) for one model. Every namespace here is HuggingFace's
#: own spelling, taken from its organization record rather than inferred.
SAME_MODEL_TWO_SPELLINGS = [
    ('Qwen/Qwen3-32B', 'qwen3-32b'),
    ('meta-llama/Llama-3.1-8B-Instruct', 'llama-3.1-8b-instruct'),
    ('facebook/opt-1.3b', 'opt-1.3b'),
    ('mistralai/Mistral-Large-2411', 'mistral-large-2411'),
    ('deepseek-ai/DeepSeek-V3', 'deepseek-v3'),
    ('zai-org/GLM-4.6', 'glm-4.6'),
    ('EleutherAI/pythia-1b', 'pythia-1b'),
    ('CohereForAI/c4ai-command-r-plus', 'command-r-plus'),
    ('ai21labs/AI21-Jamba-1.5-Large', 'jamba-1.5-large'),
    ('Snowflake/snowflake-arctic-instruct', 'arctic-instruct'),
    ('togethercomputer/RedPajama-INCITE-7B-Base', 'redpajama-incite-7b-base'),
    ('allenai/OLMo-2-1124-7B', 'olmo-2-1124-7b'),
    ('microsoft/phi-4', 'phi-4'),
    ('google/gemma-3-27b-it', 'gemma-3-27b-it'),
    ('mistralai/Ministral-8B-Instruct-2410', 'ministral-8b-2410'),
    ('mistralai/Codestral-22B-v0.1', 'codestral-2501'),
    ('sarvamai/sarvam-m', 'sarvam-m'),
]


@pytest.mark.parametrize('slashed,bare', SAME_MODEL_TWO_SPELLINGS)
def test_one_model_gets_one_developer_however_it_is_spelled(
    slashed: str, bare: str
) -> None:
    assert get_developer(bare) == get_developer(slashed) == slashed.split('/')[0]


def test_a_closed_model_keeps_its_company_because_it_has_no_namespace():
    """A model with no HF repo is addressed `{org}/{slug}`, so the org is right."""
    for bare, developer in (
        ('gpt-4.1-mini', 'openai'),
        ('claude-opus-4-5', 'anthropic'),
        ('grok-4', 'xai'),
        ('nova-pro', 'amazon'),
    ):
        assert get_developer(bare) == developer


def test_every_pattern_answers_with_one_path_component():
    """A value with a slash would be flattened into a different directory."""
    for pattern, developer in DEVELOPER_PATTERNS.items():
        assert '/' not in developer, pattern
        assert developer == developer.strip(), pattern
        assert developer, pattern


def test_an_unknown_model_is_not_guessed_at():
    assert get_developer('some-model-nobody-registered') == 'unknown'
    assert get_developer('') == 'unknown'
