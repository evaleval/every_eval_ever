"""Tests for the shared eval-card-registry vocabulary and resolver.

Two layers, tested separately: what the refresh tool *derives* from the
registry's list endpoints (pure functions over raw records), and what the
bundled snapshot then answers (the vocabulary consumers actually read).
"""

import pytest

from every_eval_ever.helpers.eval_card_registry import (
    Registry,
    gaps,
    hf_namespace_of,
    iter_org_identities,
    normalize,
    second_name_of,
    snapshot_meta,
)
from every_eval_ever.tools.refresh_eval_card_registry import (
    org_alias_spellings,
    org_hf_namespaces,
    org_identities,
    org_identity_spellings,
    org_second_names,
)


def _org(org_id, hf_org=None, review_status='reviewed'):
    return {'id': org_id, 'hf_org': hf_org, 'review_status': review_status}


def _alias(raw_value, canonical_id, status='confirmed'):
    return {
        'raw_value': raw_value,
        'canonical_id': canonical_id,
        'status': status,
    }


# ---------------------------------------------------------------------------
# Deriving the vocabulary
# ---------------------------------------------------------------------------


def test_a_recorded_namespace_is_an_identity_not_an_alias():
    """``meta-llama`` is Meta, so it must not read as a second name."""
    identities = org_identities([_org('meta', 'meta-llama'), _org('cohere')])

    assert identities['metallama'] == 'meta'
    assert identities['meta'] == 'meta'
    assert identities['cohere'] == 'cohere'


def test_namespaces_are_recorded_only_where_the_registry_declares_one():
    """An organization with no HuggingFace presence must stay absent.

    Mapping it to itself would let a caller publish a namespace nobody declared,
    which is the pattern guess this direction exists to avoid.
    """
    namespaces = org_hf_namespaces(
        [_org('meta', 'meta-llama'), _org('mistralai', 'mistralai'),
         _org('amazon'), _org('acme', '')]
    )

    assert namespaces == {'meta': 'meta-llama', 'mistralai': 'mistralai'}


def test_a_spelling_two_organizations_answer_to_names_neither():
    """``DeepAuto-AI`` and ``deepautoai`` are both canonical ids.

    Awarding the shared spelling to one makes the other resolve to an
    organization that is not itself, and this mapping decides a published
    ``model_info.developer``.
    """
    orgs = [_org('DeepAuto-AI'), _org('deepautoai')]

    assert org_identities(orgs) == org_identities(orgs[::-1]) == {}


def test_a_namespace_does_not_claim_a_contested_spelling():
    identities = org_identities(
        [_org('DeepAuto-AI'), _org('deepautoai'), _org('other', 'deepauto.ai')]
    )

    assert 'deepautoai' not in identities


def test_identity_spellings_keep_the_punctuation_the_registry_records():
    """``meta-llama`` is a declared namespace; ``metallama`` is nobody's name."""
    spellings = org_identity_spellings(
        [_org('meta', 'meta-llama'), _org('Qwen'), _org('DeepAuto-AI')]
    )

    assert spellings['meta-llama'] == 'meta'
    assert 'metallama' not in spellings
    # Case is still folded: the registry aims for HuggingFace-true casing and
    # HuggingFace is not consistent, so these are one identifier.
    assert spellings['qwen'] == 'Qwen'
    # Punctuation twins stay distinct here, so neither has to be dropped.
    assert spellings['deepauto-ai'] == 'DeepAuto-AI'


def test_a_spelling_two_organizations_record_is_dropped_here_too():
    orgs = [_org('acme', 'shared-ns'), _org('other', 'Shared-NS')]

    assert org_identity_spellings(orgs) == {
        'acme': 'acme',
        'other': 'other',
    }


def test_alias_spellings_keep_an_alias_that_restates_its_own_name():
    """The read-side ``second_name_of`` drops these; resolution wants them."""
    spellings = org_identity_spellings([_org('mistralai'), _org('allenai')])
    aliases = org_alias_spellings(
        [
            _alias('Mistral AI', 'mistralai'),
            _alias('AI2', 'allenai'),
            _alias('Guess', 'allenai', status='pending'),
            _alias('Stale', 'an-org-the-registry-dropped'),
        ],
        spellings,
    )

    assert aliases == {'mistral ai': 'mistralai', 'ai2': 'allenai'}


def test_second_names_keep_only_a_genuinely_different_name():
    identities = org_identities(
        [_org('mistralai'), _org('meta', 'meta-llama'), _org('zai')]
    )
    second_names = org_second_names(
        [
            _alias('Mistral', 'mistralai'),
            # Restates an identity: a canonical id, then a namespace.
            _alias('Mistral AI', 'mistralai'),
            _alias('meta-llama', 'meta'),
            # Unconfirmed, and pointing at an organization not in the list.
            _alias('Mistral Large', 'mistralai', status='pending'),
            _alias('Kimi', 'moonshotai'),
            # One normalized spelling claimed by two organizations.
            _alias('GLM', 'zai'),
            _alias('glm', 'meta'),
        ],
        identities,
    )

    assert second_names == {'mistral': 'mistralai'}


def test_second_names_yield_to_a_canonical_id_of_another_organization():
    """The registry sometimes has two canonical ids for one publisher.

    ``AI21 Labs`` is a confirmed alias of ``ai21`` while ``ai21-labs`` is its
    own canonical id. Answering with either would assert an ordering between
    them that the registry has not made, so the alias is dropped.
    """
    identities = org_identities([_org('ai21'), _org('ai21-labs')])

    assert org_second_names([_alias('AI21 Labs', 'ai21')], identities) == {}


# ---------------------------------------------------------------------------
# Reading the bundled snapshot
# ---------------------------------------------------------------------------


def test_normalize_collapses_case_and_punctuation():
    assert (
        normalize('Moonshot AI')
        == normalize('moonshot-ai')
        == normalize('moonshot_ai')
        == 'moonshotai'
    )
    assert normalize('  Z.AI  ') == 'zai'
    assert normalize('Win Rate') == normalize('win_rate') == 'winrate'


def test_hf_namespace_of_reads_the_curated_namespace():
    """The direction from organization to publishing namespace.

    ``meta`` names the company and ``meta-llama`` is where its repos live, so a
    source that offers only a website needs this to reach a repo id that
    resolves. An organization the registry records no namespace for gets
    ``None``, so a caller keeps whatever it already had.
    """
    assert hf_namespace_of('meta') == 'meta-llama'
    assert hf_namespace_of(' alibaba ') == 'qwen'
    assert hf_namespace_of('amazon') is None
    assert hf_namespace_of('an-organization-no-registry-records') is None


def test_second_name_of_answers_only_for_a_different_name():
    assert second_name_of('mistral') == 'mistralai'
    assert second_name_of('AI2') == 'allenai'
    # A canonical id, and a HuggingFace namespace the registry records for one.
    assert second_name_of('mistralai') is None
    assert second_name_of('meta-llama') is None
    # Unknown to the registry, and not a string at all.
    assert second_name_of('mosaicml') is None
    assert second_name_of(None) is None


def test_second_names_and_identities_never_overlap():
    """A spelling cannot be both a name of record and a second name for one."""
    identities = {key for key, _ in iter_org_identities()}

    assert identities
    assert not [key for key in identities if second_name_of(key)]


def test_org_resolution_reads_namespaces_and_second_names_alike():
    """A converter asks "which organization is this" and gets one answer."""
    registry = Registry()

    assert registry.org('meta-llama').canonical_id == 'meta'
    assert registry.org('meta-llama').strategy == 'snapshot_identifier'
    assert registry.org('AI2').canonical_id == 'allenai'
    assert registry.org('AI2').strategy == 'snapshot_alias_identifier'
    assert registry.org('meta-llama').reviewed


def test_a_recorded_spelling_outranks_a_punctuation_collapse():
    """``meta-llama`` is a namespace Meta declares; ``metallama`` is nobody's."""
    registry = Registry()

    assert registry.org('meta-llama').strategy == 'snapshot_identifier'
    assert registry.org('META-LLAMA').strategy == 'snapshot_identifier'
    assert registry.org('metallama').strategy == 'snapshot_normalized'
    assert registry.org('Moonshot AI').strategy == 'snapshot_alias_identifier'
    assert registry.org('moonshot-ai').strategy == 'snapshot_normalized'
    # Same answer either way; only the strength of the claim differs.
    assert registry.org('metallama').canonical_id == 'meta'


def test_no_alpaca_eval_publisher_needs_the_normalized_tiers():
    """Every organization on either leaderboard matches a recorded identifier.

    The punctuation-insensitive fallback exists but decides nothing today. If a
    future upstream row lands on it, this fails and the spelling gets looked at
    rather than resolved by coincidence.
    """
    registry = Registry()
    namespaces = (
        'meta-llama',
        'Qwen',
        'HuggingFaceH4',
        'NousResearch',
        'baichuan-inc',
        'deepseek-ai',
        'zai-org',
        'Nanbeige',
        'WizardLM',
        'WizardLMTeam',
        'mistralai',
        'lmsys',
        'openai',
        'anthropic',
    )
    weak = {'snapshot_normalized', 'snapshot_alias_normalized'}
    resolved = {
        namespace: registry.org(namespace) for namespace in namespaces
    }

    assert not [
        namespace
        for namespace, res in resolved.items()
        if res.resolved and res.strategy in weak
    ]


def test_a_canonical_id_always_resolves_to_itself():
    """Including the four whose normalized spelling another id also answers to.

    Those spellings are unowned in the snapshot, so an id like ``deepautoai``
    reaches itself only by name. Resolving it to its punctuation twin would
    publish one organization's records under another's directory.
    """
    registry = Registry()
    for org_id in ('deepautoai', 'DeepAuto-AI', 'mistralai', 'meta', 'zai'):
        resolution = registry.org(org_id)
        assert resolution.canonical_id == org_id, org_id

    assert registry.org('deepautoai').strategy == 'snapshot_exact'
    assert second_name_of('deepautoai') is None


def test_snapshot_records_its_own_provenance():
    """A vendored snapshot without provenance cannot be audited."""
    meta = snapshot_meta()

    assert 'read-only' in meta['source']
    assert meta['retrieved_date']
    assert meta['counts']['orgs'] > 0
    assert 'refresh_eval_card_registry' in meta['note']
    # The gaps are recorded rather than silently absent, so a consumer can
    # report them and a refresh can be diffed.
    assert 'metric:avg_length' in gaps()


# ---------------------------------------------------------------------------
# The opt-in live path
# ---------------------------------------------------------------------------


def test_registry_never_resolves_in_a_mode_that_creates_canonicals():
    """The live path must not write to a shared registry.

    ``POST /resolve`` defaults to ``mode="resolve"``, which auto-creates a draft
    canonical for anything it cannot place. Only ``mode="exact"`` is
    side-effect-free, so that is the only mode this module may send.
    """
    sent = {}

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {'canonical_id': None, 'created_new': False}

    def _post(url, json=None, timeout=None):
        sent.update(json or {})
        return _Response()

    registry = Registry(live=True)
    module = pytest.importorskip('requests')
    original = module.post
    module.post = _post
    try:
        registry.metric('a_column_the_registry_has_never_heard_of')
    finally:
        module.post = original

    assert sent['mode'] == 'exact'


def test_live_registry_failure_is_never_fatal():
    """A registry outage degrades provenance; it does not fail a conversion."""

    def _post(url, json=None, timeout=None):
        raise OSError('registry unreachable')

    registry = Registry(live=True)
    module = pytest.importorskip('requests')
    original = module.post
    module.post = _post
    try:
        resolution = registry.metric('a_column_with_no_canonical')
    finally:
        module.post = original

    assert resolution.canonical_id is None
    assert resolution.strategy == 'registry_unavailable'
    assert 'registry unreachable' in registry.live_error


def test_one_outage_does_not_relabel_every_later_miss():
    """A clean miss and an outage are different facts about a record.

    ``live_error`` is a run-level aggregate, so a resolution's strategy must not
    be decided from it.
    """
    calls = []

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {'canonical_id': None, 'created_new': False}

    def _post(url, json=None, timeout=None):
        calls.append(json['raw_value'])
        if len(calls) == 1:
            raise OSError('registry unreachable')
        return _Response()

    registry = Registry(live=True)
    module = pytest.importorskip('requests')
    original = module.post
    module.post = _post
    try:
        first = registry.metric('a_column_the_registry_never_answers_for')
        second = registry.metric('another_column_with_no_canonical')
    finally:
        module.post = original

    assert first.strategy == 'registry_unavailable'
    assert second.strategy == 'no_canonical'
    # The outage is still reported once, for the run as a whole.
    assert 'registry unreachable' in registry.live_error


def test_live_mode_asks_again_about_a_gap_the_snapshot_recorded():
    """A recorded gap is an answer with a date on it, not a permanent one.

    Minting the canonical is a PR to the registry; the snapshot only learns about
    it at the next refresh, and live mode is what a maintainer turns on to not
    wait for that.
    """
    metric_gaps = [
        query.split(':', 1)[1]
        for query in gaps()
        if query.startswith('metric:')
    ]
    if not metric_gaps:
        pytest.skip('the snapshot currently records no metric gaps')
    asked = []

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {'canonical_id': 'metric:win_rate', 'created_new': False}

    def _post(url, json=None, timeout=None):
        asked.append(json['raw_value'])
        return _Response()

    module = pytest.importorskip('requests')
    original = module.post
    module.post = _post
    try:
        live = Registry(live=True).metric(metric_gaps[0])
        offline = Registry().metric(metric_gaps[0])
    finally:
        module.post = original

    assert asked == [metric_gaps[0]]
    assert live.canonical_id == 'metric:win_rate'
    assert live.strategy == 'live_exact'
    # Offline, a recorded gap stays one and nothing is contacted.
    assert (offline.canonical_id, offline.strategy) == (None, 'no_canonical')
