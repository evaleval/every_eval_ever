# Canonicalization — the eval-card-registry

*Scope: making model/benchmark ids canonical. The registry-side mechanics live in
the registry repo (pointer at the bottom).*

Ids in your output must be canonical, resolved through the eval-card-registry.
Unresolved slugs auto-create `draft` canonicals and fragment the data (two ids for
one thing), so this is part of shipping an adapter.

**Four entity kinds, not one.** People resolve models and stop, then hand-write the
rest — that is where adapters diverge from each other:

| Your field | `entity_type` | Registry gives you |
|---|---|---|
| `model_info.id` | `model` | the canonical id (HF repo id when the model is on HF) |
| `evaluation_name` | `benchmark` | the benchmark slug; sub-tasks via `parent_benchmark_id` |
| `metric_config.metric_id` | `metric` | the metric slug and its canonical bounds/direction |
| `eval_library.name` | `harness` | the harness slug |

`model_info.developer` has an `org` entity behind it too. Two things about model ids
that surprise people:
- **The registry's model `id` is the real HF repo id, with HF-true casing**, when the
  model is on HF (`Qwen/Qwen3-32B`, `meta-llama/Llama-3.1-8B-Instruct`); only models
  with no HF repo use `{org_id}/{slug}` (`openai/gpt-4o`). An id that looks like
  `{company}/{api-model-name}` for a model that *is* on the Hub is usually an
  unreconciled API-catalog draft — prefer the HF-anchored canonical and flag the draft.
- **`org_id` is the normalized company, so it differs from the id prefix by design**
  (`Qwen/Qwen3-32B` sits under org `alibaba`; `canonical_orgs.hf_org` maps back). Don't
  "fix" the prefix to match the org. Note the EEE datastore path takes the developer
  folder from the id prefix, not from `model_info.developer`.

What an adapter author needs here:
- **Search the registry first; alias your raw slug to the *existing* canonical;
  create a new canonical only if the entity is genuinely absent** — a new canonical
  is a lasting namespace decision, so ask the operator before deliberately minting
  one (SKILL.md step 7). (A *batch* adapter can't gate per-id: resolve-by-default will
  auto-create drafts for the tail — surface those in the decision log rather than block,
  see next bullet.)
- **Resolve by default; flag what stays unverified.** The registry is a separate repo,
  but its resolver is a **hosted, no-auth endpoint**:
  `POST https://evaleval-entity-registry.hf.space/api/v1/resolve` with
  `{"raw_value","entity_type"}` (`entity_type` ∈ `model`/`benchmark`/`metric`/`harness`/
  `org`/…), returning `canonical_id` + `strategy`/`confidence`/`created_new`/
  `review_status`. Prefer resolving live and use `canonical_id` for the join-key
  field (`model_info.id`); record the provenance fields in `additional_details`. Give
  the adapter an opt-out flag (e.g. `--no-registry-resolve`) for speed/offline/
  determinism, and on opt-out **or any network error fall back to the path id, marked
  unverified — never fatal** (a converter must not die because a Space was asleep). Use
  `requests` (already a dep) so the flag is about speed, not a new dependency. Whatever
  the resolver couldn't confidently place — `created_new` drafts, low `confidence`,
  non-`reviewed` status — goes in the decision log for a follow-up alias PR.
- **Never key `evaluation_id` on the resolved canonical id** — the registry can re-map
  a draft later. Resolved id = join key; raw source identity = record identity. Rule and
  reasoning: `fields.md` model_info.
- **Disambiguate look-alikes** — `arc` (AI2 Reasoning Challenge, `allenai/ai2_arc`)
  is a *different* dataset from `arc-agi` (Chollet). Confirm from the paper.
- **Pin what you resolved.** The two most careful adapters vendor the resolution
  instead of re-querying at convert time: `utils/benchpress/model_id_map.json` pins the
  batch resolver's output with a `_meta` block (endpoint, date, `n_resolved`/
  `n_unresolved`, hand-verified `overrides`), and `utils/paperswithcode/
  registry_metrics.json` vendors the metric entries with the registry commit the
  bounds came from (`bound_registry_revision`), refreshed by a script. Record the
  revision next to any value you took from the registry so a reader can tell which
  registry state produced it — and so bumping it is a reviewable diff.
- **Decide what an unresolved entity does.** PwC fails closed (an unknown or
  ambiguous metric aborts the run) because a wrong bound silently corrupts every score;
  a model id can instead fall back marked unverified. Whichever you pick, say so in the
  adapter README — silence reads as "everything resolved".
- Adding aliases/canonicals is a separate PR to the registry repo (not the
  adapter repo, not the datastore). Alias bridges for forms no generator carries go in
  `seed/models/enrichments/aliases.yaml`; new canonicals go in the relevant
  `seed/*.yaml` (`benchmarks.yaml`, `metrics.yaml`, `harnesses.yaml`); known-wrong
  upstream records go in `enrichments/upstream_corrections.yaml` with a note in
  `curation/UPSTREAM_DATA_ISSUES.md`.

The registry-side mechanics — which YAML file, when the `normalized` matcher already
covers a variant, the id standards (HF-true casing, closed-model form, etc.), and the
`seed --local` + resolver verification — live in the **registry repo's own
`CONTRIBUTING.md`**, not in this skill. This split is deliberate: registry
contribution is a different task with a different audience.
