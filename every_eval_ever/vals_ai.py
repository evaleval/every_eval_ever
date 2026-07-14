"""Shared provenance and public model-registry support for Vals.ai."""

from __future__ import annotations

import re
from collections.abc import Callable
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

SOURCE_NAME = 'Vals.ai'
SOURCE_ORGANIZATION_URL = 'https://www.vals.ai'
BENCHMARKS_URL = f'{SOURCE_ORGANIZATION_URL}/benchmarks'
USER_AGENT = 'every-eval-ever vals-ai adapter'
EXTERNALLY_MANAGED = 'externally_managed'
UNKNOWN_DEPLOYMENT = 'unknown'
INFERENCE_ENGINE_NAME = 'unknown'
INFERENCE_ENGINE_VERSION = 'unknown'

_MODEL_TABLE_ASSET_RE = re.compile(
    r'component-url=["\'](?P<path>/[^"\']*ModelTable[^"\']*\.js)["\']'
)
_CONSTANTS_IMPORT_RE = re.compile(
    r'from["\']\./(?P<path>constants\.[^"\']+\.js)["\']'
)
_MODEL_ENTRY_RE = re.compile(
    r'"(?P<key>[^"]+)":\{company:"(?P<company>[^"]+)",'
    r'label:"[^"]*",release_date:(?:"[^"]*"|null),'
    r'open_source:!(?P<negated>[01])'
)


def fetch_text(url: str) -> str:
    request = Request(url, headers={'User-Agent': USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read().decode('utf-8')
    except URLError as exc:
        raise RuntimeError(f'Failed to fetch {url}: {exc}') from exc


def parse_model_registry(script: str) -> dict[str, dict[str, str | bool]]:
    registry = {
        match.group('key'): {
            'company': match.group('company'),
            'open_weights': match.group('negated') == '0',
        }
        for match in _MODEL_ENTRY_RE.finditer(script)
    }
    if not registry:
        raise ValueError('Vals.ai model registry contained no model entries')
    return registry


def fetch_model_registry(
    base_url: str = SOURCE_ORGANIZATION_URL,
    *,
    fetcher: Callable[[str], str] = fetch_text,
) -> dict[str, dict[str, str | bool]]:
    base_url = base_url.rstrip('/')
    models_html = fetcher(f'{base_url}/models')
    model_table_match = _MODEL_TABLE_ASSET_RE.search(models_html)
    if model_table_match is None:
        raise ValueError(
            'Vals.ai models page did not identify ModelTable asset'
        )
    model_table_url = urljoin(base_url, model_table_match.group('path'))
    model_table_script = fetcher(model_table_url)
    constants_match = _CONSTANTS_IMPORT_RE.search(model_table_script)
    if constants_match is None:
        raise ValueError('Vals.ai ModelTable did not identify constants asset')
    constants_url = urljoin(model_table_url, constants_match.group('path'))
    return parse_model_registry(fetcher(constants_url))


def provider_to_platform(provider: str | None) -> str:
    if not isinstance(provider, str) or not provider.strip():
        return 'unknown'
    platform = re.sub(r'[^a-z0-9]+', '_', provider.casefold()).strip('_')
    return platform or 'unknown'


def deployment_type_for_provider(provider: str | None) -> str:
    """Only claim external deployment when the result names a provider."""
    if not isinstance(provider, str) or not provider.strip():
        return UNKNOWN_DEPLOYMENT
    if provider.strip().casefold() == 'unknown':
        return UNKNOWN_DEPLOYMENT
    return EXTERNALLY_MANAGED


def model_availability(
    registry: dict[str, dict[str, str | bool]], vals_model_id: str
) -> str:
    entry = registry.get(vals_model_id)
    if entry is None:
        return 'unknown'
    open_weights = entry.get('open_weights')
    if open_weights is True:
        return 'open_weights'
    if open_weights is False:
        return 'closed_weights'
    raise ValueError(
        f'Vals.ai registry has invalid open_weights for {vals_model_id!r}'
    )
