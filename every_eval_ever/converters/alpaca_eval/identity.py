"""Resolve AlpacaEval leaderboard slugs to model identities.

A leaderboard row is keyed by an AlpacaEval *slug* (``Yi-34B-Chat``,
``aligner-2b_gpt-4-turbo-2024-04-09``), which is neither a HuggingFace repo id
nor a developer name. Upstream, however, ships one config file per slug
(``models_configs/<slug>/configs.yaml``) recording the real model name it was
served under, a reference link, how it was served and with which generation
settings. That file is the evidence this module resolves identity from, in a
fixed ladder of rungs — each rung is a *source of truth*, not a guess, and the
rung that fired is recorded on the record so a reviewer can check it.

Rungs, in order:

1. ``upstream_model_name`` — ``completions_kwargs.model_name`` is a repo id
   (``FuseAI/FuseChat-Llama-3.1-8B-Instruct``).
2. ``vendor_api`` — the entry was served through a first-party vendor API
   (``anthropic_completions``, OpenAI with no custom ``base_url``, …), so the
   vendor is the developer.
3. ``hf_model_link`` — the reference link is a HuggingFace *model* page whose
   repo name corresponds to the slug.
4. ``hf_link_org`` — the link is a HuggingFace model page for a *different*
   artefact (e.g. the reward model of a best-of-n entry): trust the org, keep
   the slug as the model name.
5. ``github_link`` / 6. ``vendor_site`` — the link is a project repository or a
   vendor website: trust the owner/host, keep the slug as the model name.
7. ``name_pattern`` — fall back to the repo-wide developer patterns
   (``helpers.get_developer``).

Unresolvable rows return ``None`` so the caller can record a failure instead of
inventing an identity.

Two canonicalization steps then correct a repo id the source spells in a way
HuggingFace does not — casing (:func:`canonical_repo_casing`) and renames
(:data:`HF_CANONICAL_NAME`, refreshed by ``refresh_hf_canonical_ids.py``). Both
change only how a repo is spelled, and record the source's spelling as
``model_id_as_referenced``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional
from urllib.parse import urlparse

from every_eval_ever.helpers.developer import get_developer, get_model_id
from every_eval_ever.helpers.eval_card_registry import hf_namespace_of

# ---------------------------------------------------------------------------
# Evidence vocabularies
# ---------------------------------------------------------------------------

#: First path segment of a huggingface.co URL that is *not* an organization.
_HF_RESERVED = frozenset(
    {
        'spaces',
        'datasets',
        'papers',
        'paper',
        'collections',
        'blog',
        'docs',
        'models',
        'organizations',
        'join',
        'pricing',
        'settings',
        'chat',
        'learn',
    }
)

#: Hosts that document a model without identifying its developer.
_PAPER_HOSTS = frozenset(
    {
        'arxiv.org',
        'ar5iv.org',
        'alphaxiv.org',
        'openreview.net',
        'aclanthology.org',
        'semanticscholar.org',
        'paperswithcode.com',
        'papers.nips.cc',
        'proceedings.mlr.press',
        'proceedings.neurips.cc',
        'dl.acm.org',
        'doi.org',
        'medium.com',
        'sites.google.com',
        'notion.so',
        'github.io',
    }
)

_GENERIC_TLDS = frozenset(
    {
        'com',
        'org',
        'net',
        'io',
        'co',
        'xyz',
        'dev',
        'app',
        'tech',
        'cloud',
        'me',
        'info',
        'cn',
        'us',
    }
)

#: ``fn_completions`` values that mean "a first-party vendor API served this".
_VENDOR_FNS = {
    'anthropic_completions': 'anthropic',
    'cohere_completions': 'cohere',
    'google_completions': 'google',
    'jina_chat_completions': 'jinaai',
}

#: ``fn_completions`` values that mean "weights were run locally", and the
#: inference engine each implies.
_LOCAL_FNS = {
    'vllm_local_completions': 'vllm',
    'huggingface_local_completions': 'transformers',
}

#: ``fn_completions`` values that name their own hosting platform.
_HOSTED_FNS = {
    'openai_completions': 'openai',
    'replicate_completions': 'replicate',
    'huggingface_completions': 'huggingface',
}

#: ``fn_completions`` values that name no caller. Upstream spells "absent" three
#: ways: the key missing, an empty value, and the literal ``null``
#: (``Samba-CoE-v0.1``).
_NO_COMPLETIONS_FN = frozenset({'', 'null', 'none'})

_NON_ALNUM = re.compile(r'[^a-z0-9]+')

#: Identity rungs whose ``model_id`` **is** a HuggingFace repo id, and so the
#: only ones looked up in the rename map. The other rungs *construct* an id from
#: an organization and a slug, and such an id colliding with a real repo is not
#: evidence that the repo is what ran: HuggingFace redirects
#: ``cohere/command-nightly`` to ``CohereLabs/Command-nightly``, but that row was
#: served by Cohere's API under a rolling alias.
HF_GROUNDED_SOURCES = frozenset({'hf_model_link', 'upstream_model_name'})

HF_CANONICAL_NAME = 'hf_canonical_ids.json'

#: Where ``refresh_hf_canonical_ids.py`` writes the rename map in a source
#: checkout. Reading goes through :func:`hf_canonical_ids` so an installed or
#: zipped package works too.
HF_CANONICAL_PATH = (
    Path(__file__).resolve().parent / 'data' / HF_CANONICAL_NAME
)

#: Reference links for entries whose upstream config omits ``link`` but whose
#: upstream pull request states the developer. Keys are **exact** slugs, since a
#: substring rule would misfire (``-evo`` also matches
#: ``llama-2-chat-7b-evol70k-neft``). These feed the normal ladder rather than
#: bypassing it.
_LINK_EVIDENCE = {
    # "Evo-7b is fine tuned based on Llama 2. The team worked on this model
    # are from https://evolusion.ai/." — tatsu-lab/alpaca_eval#144
    'evo-7b': (
        'https://evolusion.ai/',
        'https://github.com/tatsu-lab/alpaca_eval/pull/144',
    ),
}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelIdentity:
    """A leaderboard entry's resolved identity plus the evidence behind it."""

    slug: str
    model_id: str
    developer: str
    identity_source: str
    model_availability: str = 'unknown'
    deployment_type: str = 'unknown'
    inference_platform: Optional[str] = None
    inference_engine: Optional[str] = None
    reference_link: Optional[str] = None
    pretty_name: Optional[str] = None
    upstream_model_name: Optional[str] = None
    #: Set when ``reference_link`` came from :data:`_LINK_EVIDENCE` rather than
    #: from the upstream config itself.
    link_evidence: Optional[str] = None
    #: The repo id the source referenced, set only when it is no longer the id
    #: HuggingFace serves and :func:`hf_canonical_ids` replaced it.
    model_id_as_referenced: Optional[str] = None
    #: See :func:`local_generate_evidence`.
    deployment_evidence: Optional[str] = None


# ---------------------------------------------------------------------------
# Upstream config accessors (upstream rows are hand-written YAML: guard types)
# ---------------------------------------------------------------------------


def _mapping(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def completions_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the ``completions_kwargs`` mapping of an upstream model config."""
    return _mapping(_mapping(config).get('completions_kwargs'))


def upstream_model_name(config: Dict[str, Any]) -> str:
    """Return the model name the entry was served under, if recorded."""
    value = completions_kwargs(config).get('model_name')
    return value.strip() if isinstance(value, str) else ''


def completions_fn(config: Dict[str, Any]) -> str:
    """Return the upstream ``fn_completions`` value (may be empty)."""
    value = _mapping(config).get('fn_completions')
    return value.strip() if isinstance(value, str) else ''


def reference_link(config: Dict[str, Any]) -> str:
    """Return the entry's reference link (may be empty)."""
    value = _mapping(config).get('link')
    return value.strip() if isinstance(value, str) else ''


def base_url(config: Dict[str, Any]) -> str:
    """Return the custom API base URL the entry was served from, if any.

    Upstream configs spell this three ways depending on when they were written:
    ``client_kwargs.base_url`` (current), a top-level ``base_url``, and the
    pre-1.0 openai-python ``openai_api_base``, which the OpenChat entries point
    at ``127.0.0.1``.
    """
    kwargs = completions_kwargs(config)
    for holder, key in (
        (_mapping(kwargs.get('client_kwargs')), 'base_url'),
        (kwargs, 'base_url'),
        (kwargs, 'openai_api_base'),
        (kwargs, 'api_base'),
    ):
        value = holder.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ''


# ---------------------------------------------------------------------------
# Link parsing
# ---------------------------------------------------------------------------


def _host_and_parts(link: str) -> tuple:
    if not link:
        return '', []
    parsed = urlparse(link if '//' in link else f'https://{link}')
    host = (parsed.netloc or '').lower()
    if host.startswith('www.'):
        host = host[4:]
    parts = [p for p in (parsed.path or '').split('/') if p]
    return host, parts


def hf_repo_from_link(link: str) -> Optional[str]:
    """Return ``org/repo`` when *link* is a HuggingFace **model** page."""
    host, parts = _host_and_parts(link)
    if host not in {'huggingface.co', 'hf.co'} or len(parts) < 2:
        return None
    if parts[0].lower() in _HF_RESERVED:
        return None
    return f'{parts[0]}/{parts[1]}'


def github_owner_from_link(link: str) -> Optional[str]:
    """Return the repository owner when *link* is a GitHub project URL."""
    host, parts = _host_and_parts(link)
    if host != 'github.com' or not parts:
        return None
    return parts[0]


def site_developer_from_link(link: str) -> Optional[str]:
    """Derive a developer slug from a vendor's own website.

    ``boson.ai`` → ``bosonai``, ``mistral.ai`` → ``mistralai`` (then normalised
    by the shared developer patterns), ``together.xyz`` → ``together``. Paper
    and generic publishing hosts identify no developer and return ``None``.
    """
    host, _ = _host_and_parts(link)
    if not host or host in _PAPER_HOSTS:
        return None
    if any(host == h or host.endswith(f'.{h}') for h in _PAPER_HOSTS):
        return None
    labels = [label for label in host.split('.') if label]
    # Drop subdomains such as api./coe-1.cloud., keep the registrable name.
    while len(labels) > 2:
        labels.pop(0)
    if len(labels) == 2 and labels[1] in _GENERIC_TLDS:
        labels = labels[:1]
    if not labels:
        return None
    # ``boson.ai`` reads as one word (bosonai) — that is also how such vendors
    # name themselves on HuggingFace (mistralai, bosonai).
    slug = (
        f'{labels[0]}{labels[1]}'
        if len(labels) == 2 and labels[1] == 'ai'
        else '-'.join(labels)
    )
    slug = _NON_ALNUM.sub('-', slug.lower()).strip('-')
    if not slug:
        return None
    normalized = get_developer(slug)
    return normalized if normalized != 'unknown' else slug


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------


def _is_repo_id(value: str) -> bool:
    """Whether *value* looks like an ``org/model`` model repository id.

    ``:`` disqualifies it: a colon marks a *serving* reference rather than a
    model, and adopting one publishes the host as the developer — upstream
    serves Llama 2 70B as ``replicate/llama70b-v2-chat:e951f18…``, whose owner
    is Replicate, not Meta. Such rows fall through to the link-based rungs; the
    served reference is still recorded as ``upstream_model_name``.
    """
    if not value or value.startswith(('.', '/', 'http')) or ' ' in value:
        return False
    if ':' in value:
        return False
    parts = value.split('/')
    return len(parts) == 2 and all(parts)


def _norm(value: str) -> str:
    return _NON_ALNUM.sub('', value.lower())


def _repo_matches_slug(repo: str, slug: str) -> bool:
    """Whether an HF repo name plausibly names the same model as *slug*.

    Best-of-n entries link the *reward* model they re-ranked with
    (``ultralm-13b-best-of-16`` → ``openbmb/UltraRM-13b``); adopting that repo
    id would publish the wrong model. Requiring one name to be a prefix of the
    other keeps ``Storm-7B-best-of-64`` → ``jieliu/Storm-7B`` while rejecting
    the reward-model case.
    """
    repo_name = _norm(repo.split('/')[-1])
    slug_name = _norm(slug)
    if not repo_name or not slug_name:
        return False
    return repo_name.startswith(slug_name) or slug_name.startswith(repo_name)


def _is_local_url(url: str) -> bool:
    host, _ = _host_and_parts(url)
    host = host.split(':')[0]
    return host in {'localhost', '127.0.0.1', '0.0.0.0', '::1', '[::1]'}


def _served_locally(fn: str, custom_base_url: str) -> bool:
    """Whether the entry ran on hardware the submitter controlled."""
    return fn in _LOCAL_FNS or (
        bool(custom_base_url) and _is_local_url(custom_base_url)
    )


def _vendor_from_fn(fn: str, custom_base_url: str) -> Optional[str]:
    if fn in _VENDOR_FNS:
        return _VENDOR_FNS[fn]
    if fn == 'openai_completions' and not custom_base_url:
        return 'openai'
    return None


def local_generate_evidence(config: Dict[str, Any]) -> Optional[str]:
    """The ``completions_kwargs`` key showing weights ran in-process, or ``None``.

    For the 28 configs recording no usable ``fn_completions``, these two kwargs
    still decide ``deployment_type``, because AlpacaEval's local and API callers
    take different parameter names:

    - ``model_kwargs.torch_dtype`` — an argument to ``from_pretrained``, with no
      remote API to pass it to.
    - ``max_new_tokens`` / ``max_length`` — the ``transformers.generate()``
      spelling, where the API spelling is ``max_tokens``.

    Says nothing about ``model_availability``: running weights yourself does not
    make them public.
    """
    kwargs = completions_kwargs(config)
    if _mapping(kwargs.get('model_kwargs')).get('torch_dtype'):
        return 'model_kwargs.torch_dtype'
    for key in ('max_new_tokens', 'max_length'):
        if kwargs.get(key) is not None:
            return key
    return None


def _deployment(fn: str, custom_base_url: str, config: Dict[str, Any]) -> tuple:
    """``(deployment_type, inference_platform, inference_engine, evidence)``.

    ``evidence`` is set only where ``fn_completions`` was unreadable and
    :func:`local_generate_evidence` decided the deployment instead.
    """
    if _served_locally(fn, custom_base_url):
        return 'self_deployed', 'local', _LOCAL_FNS.get(fn), None
    if custom_base_url:
        host, _ = _host_and_parts(custom_base_url)
        return 'externally_managed', host or None, None, None
    if fn in _VENDOR_FNS:
        return 'externally_managed', _VENDOR_FNS[fn], None, None
    if fn in _HOSTED_FNS:
        return 'externally_managed', _HOSTED_FNS[fn], None, None
    if fn.lower() in _NO_COMPLETIONS_FN:
        evidence = local_generate_evidence(config)
        if evidence:
            # The engine stays unset: the kwargs say the weights were held
            # locally, not whether transformers or vLLM ran them.
            return 'self_deployed', 'local', None, evidence
    return 'unknown', None, None, None


def _availability(fn: str, link: str, custom_base_url: str) -> str:
    if _served_locally(fn, custom_base_url):
        # Running the weights yourself is only possible if they are available.
        return 'open_weights'
    if hf_repo_from_link(link):
        return 'open_weights'
    if _vendor_from_fn(fn, custom_base_url):
        return 'closed_weights'
    return 'unknown'


def canonical_repo_casing(configs: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """Map lowercased repo ids to the spelling a HuggingFace URL corroborates.

    Pass the result to :func:`resolve_identity` as ``casing`` to fix hand-typed
    repo ids whose casing is wrong. Across all 227 upstream configs the 126
    linked repos disagree with each other on casing zero times, so first-seen
    wins is unambiguous in practice.
    """
    casing: Dict[str, str] = {}
    for config in configs:
        repo = hf_repo_from_link(reference_link(_mapping(config)))
        if repo:
            casing.setdefault(repo.lower(), repo)
    return casing


def hf_canonical_map(payload: Dict[str, Any]) -> Dict[str, str]:
    """Extract the lookup map from a parsed ``hf_canonical_ids.json`` payload.

    Keys are lowercased because a repo id is case-insensitive on HuggingFace,
    so a differently-cased spelling of a renamed repo must still be caught.
    """
    renames = _mapping(payload).get('renamed_repos')
    return {
        key.lower(): value
        for key, value in _mapping(renames).items()
        if isinstance(key, str) and isinstance(value, str)
    }


def hf_org_namespace(org_id: str) -> Optional[str]:
    """The HuggingFace namespace *org_id* publishes under, or ``None``.

    Thin seam over the registry so a test can substitute a map and so
    :func:`resolve_identity` can be told to consult nothing.
    """
    return hf_namespace_of(org_id)


@lru_cache(maxsize=1)
def hf_canonical_ids() -> Mapping[str, str]:
    """Return the vendored ``referenced repo id -> current repo id`` map.

    Renames are vendored rather than resolved live for the same reason the
    registry snapshot is: a conversion stays deterministic and needs no network,
    and the map is reviewable in the diff. It is authoritative only as of its
    ``retrieved_date`` — rerun ``refresh_hf_canonical_ids.py --check`` to see
    whether HuggingFace has moved anything since.
    """
    resource = resources.files(
        'every_eval_ever.converters.alpaca_eval'
    ).joinpath('data', HF_CANONICAL_NAME)
    payload = json.loads(resource.read_text(encoding='utf-8'))
    return MappingProxyType(hf_canonical_map(payload))


def resolve_identity(
    slug: str,
    config: Optional[Dict[str, Any]],
    casing: Optional[Dict[str, str]] = None,
    hf_canonical: Optional[Mapping[str, str]] = None,
    hf_namespaces: Optional[Mapping[str, str]] = None,
) -> Optional[ModelIdentity]:
    """Resolve one leaderboard slug against its upstream model config.

    Args:
        slug: The leaderboard row's model slug.
        config: The parsed ``models_configs/<slug>/configs.yaml`` body, or
            ``None`` when upstream has no config for the slug.
        casing: Optional output of :func:`canonical_repo_casing`, letting a row
            borrow a sibling entry's link as casing evidence (see below).
        hf_canonical: ``referenced repo id -> current repo id`` overrides,
            defaulting to the vendored :func:`hf_canonical_ids` map. Pass ``{}``
            to publish repo ids exactly as the source spells them.
        hf_namespaces: ``canonical org id -> hf_org`` overrides for the
            ``vendor_site`` rung, defaulting to the registry snapshot. Pass
            ``{}`` to publish the website's own name as the namespace.

    Returns:
        A :class:`ModelIdentity`, or ``None`` when no rung applies (the caller
        should record a failure rather than publish a guessed identity).
    """
    slug = (slug or '').strip()
    if not slug:
        return None
    config = _mapping(config)
    model_name = upstream_model_name(config)
    fn = completions_fn(config)
    link = reference_link(config)
    link_evidence = None
    if not link and slug in _LINK_EVIDENCE:
        link, link_evidence = _LINK_EVIDENCE[slug]
    custom_base_url = base_url(config)
    pretty = config.get('pretty_name')
    deployment_type, platform, engine, deployment_evidence = _deployment(
        fn, custom_base_url, config
    )
    availability = _availability(fn, link, custom_base_url)
    hf_repo = hf_repo_from_link(link)
    third_party_host = bool(custom_base_url) and not _is_local_url(
        custom_base_url
    )

    # Repo ids in ``completions_kwargs`` are hand-typed and sometimes miscased
    # (``01-ai/Yi-34b-Chat`` does not exist on HuggingFace), while a reference
    # link is a URL that resolves, so a link's spelling wins. The entry's own
    # link corroborates its own spelling; *casing* adds sibling entries' links,
    # which is what recovers the spelling for a row whose link points elsewhere
    # (``pairrm-Yi-34B-Chat`` links its reward model, not the model served).
    casing = dict(casing or {})
    if hf_repo:
        casing.setdefault(hf_repo.lower(), hf_repo)
    renames = hf_canonical_ids() if hf_canonical is None else hf_canonical
    namespaces = hf_namespaces

    def _identity(
        model_id: str, developer: str, source: str
    ) -> ModelIdentity:
        canonical = casing.get(model_id.lower())
        if canonical and canonical != model_id:
            model_id = canonical
            developer = canonical.split('/')[0]

        # A renamed repo redirects, so the old id still works while joining with
        # nothing. ``developer`` does not follow the new namespace: a redirect
        # cannot tell an organization renaming itself from a repo transferred to
        # someone else.
        referenced = None
        if source in HF_GROUNDED_SOURCES:
            current = renames.get(model_id.lower())
            if current and current != model_id:
                referenced, model_id = model_id, current

        # A vendor's own website names the *organization* (``ai.meta.com`` ->
        # ``meta``), and building a repo id out of that publishes a namespace
        # that does not exist while the real one does (``meta`` hosts no Llama
        # repo, ``meta-llama`` hosts ``Llama-2-70b-chat-hf``). The registry
        # records the namespace per org, so this is a lookup rather than a
        # pattern guess, and the website's spelling stays in provenance. Casing
        # is left to the next ``refresh_hf_canonical_ids`` sweep, which follows
        # HuggingFace's redirect and vendors the spelling it lands on.
        if source == 'vendor_site':
            namespace, _, repo_name = model_id.partition('/')
            lifted = (
                namespaces.get(namespace)
                if namespaces is not None
                else hf_org_namespace(namespace)
            )
            if lifted and lifted != namespace and repo_name:
                referenced = referenced or model_id
                model_id, developer = f'{lifted}/{repo_name}', lifted

        return ModelIdentity(
            slug=slug,
            model_id=model_id,
            developer=developer,
            identity_source=source,
            model_availability=availability,
            deployment_type=deployment_type,
            inference_platform=platform,
            inference_engine=engine,
            reference_link=link or None,
            pretty_name=pretty if isinstance(pretty, str) else None,
            upstream_model_name=model_name or None,
            link_evidence=link_evidence,
            model_id_as_referenced=referenced,
            deployment_evidence=deployment_evidence,
        )

    # 0. A hosting provider's namespace and spelling are not the model's
    # identity, so for an entry served on a third-party API its own HuggingFace
    # link outranks the served name. This folds host-side aliases
    # (``Llama-3-70b-chat-hf``) and serving tiers (``…-405B-Instruct-Turbo``)
    # back onto the canonical repo; the slug and ``upstream_model_name`` still
    # record which variant ran.
    if (
        third_party_host
        and hf_repo
        and (
            _repo_matches_slug(hf_repo, slug)
            or (
                model_name
                and _repo_matches_slug(hf_repo, model_name.split('/')[-1])
            )
        )
    ):
        return _identity(hf_repo, hf_repo.split('/')[0], 'hf_model_link')

    # 1. Upstream recorded a real repo id.
    if _is_repo_id(model_name):
        return _identity(
            model_name, model_name.split('/')[0], 'upstream_model_name'
        )

    # 2. Served by a first-party vendor API.
    vendor = _vendor_from_fn(fn, custom_base_url)
    if vendor:
        return _identity(
            get_model_id(model_name or slug, vendor), vendor, 'vendor_api'
        )

    # 3./4. HuggingFace model page.
    if hf_repo:
        org = hf_repo.split('/')[0]
        if _repo_matches_slug(hf_repo, slug):
            return _identity(hf_repo, org, 'hf_model_link')
        return _identity(get_model_id(slug, org), org, 'hf_link_org')

    # 5. Project repository.
    owner = github_owner_from_link(link)
    if owner:
        developer = owner.lower()
        return _identity(get_model_id(slug, developer), developer, 'github_link')

    # 6. Vendor website.
    site_developer = site_developer_from_link(link)
    if site_developer:
        return _identity(
            get_model_id(slug, site_developer), site_developer, 'vendor_site'
        )

    # 7. Repo-wide developer patterns. Local checkout paths ("./evo-7b") carry
    # no organization, so pattern-match the slug instead of the path.
    plain_name = model_name if model_name and '/' not in model_name else slug
    developer = get_developer(plain_name)
    if developer != 'unknown':
        return _identity(
            get_model_id(plain_name, developer), developer, 'name_pattern'
        )

    return None
