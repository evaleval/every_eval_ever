"""Strict source-license provenance for BFCL leaderboard models."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN = 'unknown'
OPEN_WEIGHTS = 'open_weights'
CLOSED_WEIGHTS = 'closed_weights'

BFCL_METHOD_URL = 'https://gorilla.cs.berkeley.edu/leaderboard.html'

REVIEWED_OPEN_LICENSES = frozenset(
    {
        'Apache 2.0',
        'Apache-2.0',
        'CC-BY-NC 4.0 License (w/ Acceptable Use Addendum)',
        'MIT',
        'Meta Llama 3 Community',
        'Meta Llama 4 Community',
        'Mistral AI Research License',
        'apache-2.0',
        'cc-by-nc-4.0',
        'falcon-llm-license',
        'gemma-terms-of-use',
        'katanemo-research',
        'modified-mit',
        'nvidia-open-model-license',
        'qwen-research',
    }
)

# BFCL labels these evaluated endpoints Proprietary even though first-party
# weights exist. Keep exact identifiers so another proprietary row cannot
# silently inherit open-weight status.
OPEN_PROPRIETARY_OVERRIDES = frozenset(
    {
        'mistralai/mistral-small-2506-fc',
        'mistralai/mistral-small-2506-prompt',
        'mistralai/open-mistral-nemo-2407-fc',
        'mistralai/open-mistral-nemo-2407-prompt',
    }
)

MODEL_AVAILABILITY_SOURCES = {
    'mistral_small_2506': (
        'https://huggingface.co/mistralai/'
        'Mistral-Small-3.2-24B-Instruct-2506'
    ),
    'mistral_nemo_2407': (
        'https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407'
    ),
}


@dataclass(frozen=True)
class BFCLProvenance:
    deployment_type: str
    model_availability: str
    inference_platform: str
    inference_engine_name: str
    inference_engine_version: str


def bfcl_provenance(
    model_id: str, source_license: str, model_link: str
) -> BFCLProvenance:
    """Classify one BFCL row from reviewed source fields."""
    normalized = model_id.strip().casefold()
    developer, separator, leaf = normalized.partition('/')
    if not separator or not developer or not leaf:
        raise ValueError(f'invalid BFCL model id: {model_id!r}')
    if not isinstance(source_license, str) or not source_license.strip():
        raise ValueError('BFCL source license must be non-blank text')
    if not isinstance(model_link, str) or not model_link.strip():
        raise ValueError('BFCL model link must be non-blank text')
    link = urlsplit(model_link.strip())
    if link.scheme != 'https' or not link.netloc:
        raise ValueError(f'BFCL model link must be absolute HTTPS: {model_link!r}')

    license_name = source_license.strip()
    if license_name == 'Proprietary':
        if normalized in OPEN_PROPRIETARY_OVERRIDES:
            return BFCLProvenance(
                UNKNOWN, OPEN_WEIGHTS, UNKNOWN, UNKNOWN, UNKNOWN
            )
        return BFCLProvenance(
            EXTERNALLY_MANAGED,
            CLOSED_WEIGHTS,
            developer,
            UNKNOWN,
            UNKNOWN,
        )
    if license_name in REVIEWED_OPEN_LICENSES:
        return BFCLProvenance(
            UNKNOWN, OPEN_WEIGHTS, UNKNOWN, UNKNOWN, UNKNOWN
        )
    raise ValueError(
        f'unreviewed BFCL license {source_license!r} for {model_id!r}'
    )
