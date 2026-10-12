# Distinguish record omissions from unavailable metadata

Use this protocol before claiming that any reproducibility or model field is missing.
It applies to deployment metadata, temperature, token limits, and other fields that an
adapter or submitter may have failed to surface.

## Classify the observation

| Status | Meaning | Required action |
|---|---|---|
| `record_absent` | The submitted record omits the field; source research has not finished | Say "not surfaced in the submitted record" and investigate |
| `research_incomplete` | One or more relevant primary surfaces or exact identities remain unchecked | Continue research; do not propose `unknown` or claim source missingness |
| `available_not_surfaced` | A reliable source contains the value but the record does not | Backfill with approval and fix/follow up on the adapter or submission path |
| `conflicting_sources` | Relevant primary sources disagree and run-level evidence does not resolve them | Document the conflict; use the field's unknown/absent representation only with approval |
| `unavailable_after_search` | The stop rule is complete and searched first-party sources do not establish the value | Document the bounded claim and use the field's unknown/absent representation with approval |

These are research statuses, not schema values. For deployment axes, the schema value
corresponding to the last two unresolved outcomes is `unknown`.

## Search primary surfaces

Search the exact evaluation, model variant, aliases, and relevant historical date.
Inspect each surface when it exists; mark it not applicable with a reason rather than
silently skipping it:

1. Raw submitted record, instance companion, run logs, and generation configuration.
2. Pinned adapter, scraper, submission payload, and evaluator client configuration.
3. First-party eval card, README, methodology page, leaderboard detail page, and data
   card.
4. Evaluation paper, appendix, supplementary material, source repository, and pinned
   configuration files.
5. Exact model card, release, weight-file listing, license/access page, and aliases.
6. Official provider API/SDK documentation, model catalog, release/deprecation notice,
   and an archived page when current documentation may differ from the evaluated run.

Follow every concrete primary-source lead found on those surfaces. Do not count a
search-result snippet, model family, provider name, or current library default as the
submitted run's value. A harness default is evidence only when the pinned version and
run configuration establish that the default governed this run.

## Keep a search ledger

For each field and exact identity, record the surface, URL or repository revision,
retrieval/release date, query or path inspected, result, and any contradiction. Reuse a
source across models only when it explicitly covers every grouped variant.

Use concise reason codes for unresolved results:

- `no_run_provenance`
- `identity_unresolved`
- `conflicting_primary_sources`
- `historical_evidence_unavailable`
- `source_access_blocked`
- `no_primary_evidence_after_search`

State what specific artifact or statement would resolve the uncertainty.
`source_access_blocked` normally remains `research_incomplete`: an inaccessible known
primary source is still an unexamined lead, not evidence that the value is unavailable.

## Apply the stop rule

Classify a field as `unavailable_after_search` only when:

- every relevant surface above is checked or marked not applicable with a reason;
- exact aliases, versions, checkpoints, and the evaluation date are searched;
- mutable pages are pinned or an archive/repository history is attempted;
- the ledger contains no unexamined primary-source lead; and
- the claim is scoped to the searched sources and retrieval date.

If any condition is false, use `research_incomplete`. Never use a time limit or number
of search queries alone as proof of unavailable metadata.

## Use a confidence threshold

- **High:** a direct first-party run artifact or explicit exact-source statement.
- **Medium:** at least two aligned first-party sources for the exact identity, with no
  contrary run-level evidence.
- **Low:** family/provider inference, alias uncertainty, a single indirect source,
  secondary reporting, an assumed default, or unresolved primary-source conflict.

Only high- or medium-confidence categorical values may enter a repair proposal. A
low-confidence candidate becomes the field's unknown/absent representation only after
the stop rule is complete; otherwise it remains `research_incomplete`.

## Phrase findings conservatively

- Before research: "The submitted EEE record does not surface this field."
- When found elsewhere: "The field is source-available but was not surfaced by this
  adapter/submission."
- After the stop rule: "The value was not established in the searched first-party
  sources as of `<date>`."

Do not generalize bounded search results into a claim that the underlying evaluation
never documented the field. For papers and eval cards, distinguish source completeness
from the platform's extraction coverage and emphasize that living first-party and
community submissions can improve that coverage over time.
