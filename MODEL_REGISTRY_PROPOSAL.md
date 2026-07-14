# Shared Model Registry Proposal

## Status

Proposal for review. This document defines a model-only registry. Metric and
dataset registries are intentionally out of scope for the first version, but
may later adopt the same resolution pattern.

## Summary

Every Eval Ever currently receives model metadata from many independent
evaluation sources. Those sources may use different names for the same model,
may omit deployment details, or may identify the same underlying model through
different hosted services. Adapter-specific normalization cannot reliably
produce consistent identities across all collections.

This proposal introduces one manually maintained external registry, loaded by
`every_eval_ever` and reused by every adapter and validator. Registry content
can change independently without an `every_eval_ever` release. The registry
distinguishes:

1. A **base model**, representing the underlying released or provider-defined
   model.
2. A **model instance**, representing a particular deployment target used for
   evaluation.
3. An **evaluation run**, which queries a registered model instance on a
   particular dataset using a particular metric and configuration.

Every distinct deployment is a distinct model instance. Multiple evaluation
runs may still reference the same instance. An API request or benchmark run is
not itself a new model instance.

The registry is authoritative. An incoming identifier that does not resolve to
exactly one registered instance is blocked until a person reviews it. There is
no fuzzy matching, inferred organization mapping, or unregistered fallback.

## Goals

- Generate consistent `model_info` across unrelated evaluation sources.
- Treat different deployments of the same base model as different evaluation
  targets.
- Preserve the relationship between those instances and their shared base
  model.
- Support open-weight, closed-weight, and manually reviewed unknown models.
- Make adding a model, alias, or deployment a small append-only-style registry
  change.
- Preserve all raw source identifiers for traceability.
- Block new or ambiguous identities until they are manually resolved.
- Give adapters, cron ingestion, PR validation, backfills, and deduplication one
  shared resolution path.

## Non-goals For Version 1

- Model family classification.
- Architecture or parameter-count metadata.
- Pricing, context length, modalities, or capabilities.
- Corporate ownership tracking.
- Metric or dataset normalization.
- Automatic discovery or approval of new registry entries.
- Treating every evaluation call as a new deployment instance.

## Core Concepts

### Base model

A base model stores metadata shared by all deployments:

- stable canonical ID;
- display name;
- developer namespace;
- weight availability;
- authoritative resolution source;
- official external identifiers.

For an open-weight model, an official first-party Hugging Face repository is
the preferred resolution source. For a closed model, the first-party provider
model catalog or API documentation is authoritative. When neither exists, a
reviewer may assign a stable manual ID with supporting evidence.

The canonical registry ID is frozen once accepted. If an upstream repository
or provider identifier changes, the new value is appended as an identifier;
the canonical ID is not renamed automatically.

### Model instance

A model instance represents the actual logical deployment target queried by an
evaluation. It references one base model and owns:

- a stable instance ID;
- deployment type;
- inference platform;
- inference engine name and version;
- source- or provider-specific identifiers;
- deployment evidence when available.

OpenRouter, Together, a first-party API, and a self-deployed runtime are
different instances even when they serve identical weights.

Two evaluations may reference the same instance when there is evidence that
they used the same logical deployment. If deployment identity is unavailable,
the registry uses a reviewed source-scoped unknown instance rather than
claiming that unrelated sources used the same deployment.

### Evaluation run

An evaluation run references a registered model instance. The run continues to
own observation-specific information such as:

- score;
- timestamps and evaluation IDs;
- model revision, when observed;
- precision, when observed;
- generation configuration;
- evaluation library;
- dataset and metric information.

These values do not belong in the static model registry.

## Proposed YAML Format

Each developer registry file contains base models and their instances:

```yaml
schema_version: "1"

models:
  - id: example/model-a
    display_name: Model A
    developer_id: example

    model_availability: open_weights
    model_availability_source_url: https://models.example.org/model-a

    resolution_source:
      type: huggingface
      id: ExampleOrg/Model-A
      url: https://huggingface.co/ExampleOrg/Model-A

    identifiers:
      - namespace: huggingface
        value: ExampleOrg/Model-A
        status: active

instances:
  - id: example/model-a@source-one-unknown
    model_id: example/model-a

    deployment_type: unknown
    inference_platform: unknown
    inference_engine:
      name: unknown
      version: unknown

    identifiers:
      - namespace: source_one
        value: example-org/model-a

  - id: example/model-a@managed-provider
    model_id: example/model-a

    deployment_type: externally_managed
    inference_platform: managed_provider
    inference_engine:
      name: unknown
      version: unknown

    evidence_url: https://provider.example.org/models/model-a

    identifiers:
      - namespace: managed_provider
        value: example/model-a

      - namespace: source_two
        value: model-a

  - id: example/model-a@source-three-runtime-x
    model_id: example/model-a

    deployment_type: self_deployed
    inference_platform: unknown
    inference_engine:
      name: runtime_x
      version: unknown

    identifiers:
      - namespace: source_three
        value: ExampleOrg/Model-A
```

Closed models use the same structure with a provider resolution source:

```yaml
models:
  - id: provider/model-b
    display_name: Model B
    developer_id: provider

    model_availability: closed_weights
    model_availability_source_url: https://provider.example.org/models

    resolution_source:
      type: provider
      id: model-b-2026-01
      url: https://provider.example.org/models

    identifiers:
      - namespace: provider
        value: model-b-2026-01

instances:
  - id: provider/model-b@first-party-api
    model_id: provider/model-b
    deployment_type: externally_managed
    inference_platform: provider
    inference_engine:
      name: unknown
      version: unknown
    identifiers:
      - namespace: provider
        value: model-b-2026-01
```

## Required Fields

Every base model requires:

- `id`
- `display_name`
- `developer_id`
- `model_availability`
- `resolution_source.type`
- `resolution_source.id`
- `resolution_source.url`
- at least one `identifier`

`model_availability` must be one of:

- `open_weights`
- `closed_weights`
- `unknown`

`model_availability_source_url` is required for known availability. A reviewed
unknown may instead include a note explaining why no stronger value is
supported.

Every model instance requires:

- `id`
- `model_id`
- `deployment_type`
- `inference_platform`
- `inference_engine.name`
- `inference_engine.version`
- at least one `identifier`

`deployment_type` must be one of:

- `self_deployed`
- `externally_managed`
- `unknown`

Optional lifecycle fields may be added without changing identity:

```yaml
status: active
replaced_by: null
notes: null
```

## Generated `model_info`

The resolver combines the base model and selected instance into a complete
`model_info` object:

```yaml
model_info:
  id: example/model-a@managed-provider
  name: Model A
  developer: example
  inference_platform: managed_provider
  inference_engine:
    name: unknown
    version: unknown
  additional_details:
    base_model_id: example/model-a
    deployment_type: externally_managed
    model_availability: open_weights
```

The adapter appends raw provenance without overriding registry-owned fields:

```yaml
additional_details:
  raw_model_id: model-a
  raw_model_namespace: source_two
```

Observed revision and precision may also be appended when present. They remain
evaluation-specific and continue to participate in semantic fingerprints.

## Resolution Contract

Adapters submit at least:

```text
source namespace
raw model identifier
```

They may additionally submit observed deployment, platform, and engine data.
The resolver normalizes case and surrounding whitespace, then performs exact
identifier matching. It returns one of three outcomes:

- `resolved`: exactly one registered instance matched;
- `unregistered`: no instance matched;
- `ambiguous`: more than one instance matched.

Only `resolved` may proceed. Both other outcomes are blocking.

Case-only differences are equivalent. Other transformations, including
punctuation changes, organization substitution, typo correction, and family
inference, are prohibited unless represented by an explicit identifier.

The same raw identifier may legitimately refer to different instances in
different namespaces. Identifier uniqueness is therefore enforced on the
normalized `(namespace, value)` pair, with any required deployment selectors,
not on `value` globally.

## Manual Resolution Workflow

When a lookup fails, ingestion produces a review candidate outside the
authoritative registry. The candidate includes:

- all raw identifiers and names;
- source namespace;
- observed deployment, platform, and engine fields;
- source URLs and local paths;
- suggested exact or case-normalized matches;
- unmodeled source fields.

The candidate cannot make validation pass. A reviewer must choose one action:

1. Add a new base model and its first instance.
2. Add a new instance for an existing base model.
3. Add an identifier to an existing instance.
4. Register a deliberately unknown source-scoped instance.
5. Reject the incoming record as invalid or unsupported.

After the registry change is reviewed, the original ingestion is rerun. The
registry is never modified automatically by an adapter, cron, or validator.

The distinction between missing and unknown is intentional:

```text
unregistered or missing → blocked
registered explicit unknown → allowed
```

## New Models And Versions

A new source identifier always blocks initially. During review, a new version
is classified as one of:

- a genuinely new base model;
- a revision of an existing base model;
- a new deployment instance;
- a new identifier for an existing instance;
- a mutable provider alias whose exact revision remains unknown.

Moving aliases such as `latest` must not silently resolve to a pinned model
version. They require an explicitly registered rolling instance unless the
source publishes the exact resolved version.

If a model introduces metadata not represented by registry version 1, the
candidate preserves that information. Reviewers may retain it as raw metadata,
map it to an existing field, add an optional registry field, or propose a new
registry schema version. Existing records are not reinterpreted silently.

## Validation Requirements

The registry loader and CI should reject:

- invalid YAML or schema violations;
- duplicate canonical model IDs;
- duplicate instance IDs;
- references to missing base models;
- invalid availability or deployment values;
- blank engine names or versions;
- ambiguous identifier mappings;
- identifiers differing only by case within the same resolution scope;
- known availability without evidence;
- known managed platforms without deployment evidence;
- unknown registry fields that may be misspellings.

The validator should independently resolve each submitted record and require
the emitted `model_info` to match the registry. Adapters cannot override
canonical IDs, developer IDs, availability, or instance execution metadata.

## Integration

The package should expose one API used everywhere:

```text
load_model_registry(repo_id, revision, ...)
resolve_model_instance(...)
build_model_info(...)
```

Consumers include:

- collection adapters;
- cron ingestion;
- local backfills;
- PR validation;
- semantic deduplication;
- `11jul_eee_validator_git`.

Registry content should be externally owned. The package owns only the loader,
resolver, supported schema contract, and validation behavior. Validators and
adapters must not keep private copies or additional model heuristics.

## External Registry Storage

The recommended authoritative store is a dedicated Hugging Face Dataset
repository. It provides Git history, immutable commit revisions, ordinary file
access, and a natural path to an optional review application.

A Hugging Face Space may provide the review UI, candidate comparison, and pull
request workflow, but the Space itself should not be the source of truth. Its
approved changes should be written to the Dataset repository.

A dedicated GitHub repository is also viable. The resolver should depend on a
versioned repository snapshot rather than on one hosting provider's mutable
default branch. The initial implementation may support one backend while
keeping repository retrieval separate from registry parsing.

The external repository can use:

```text
model-registry/
  schema.json
  models/
    example.yaml
    provider.yaml
```

Each developer file contains its base models and their instances. The loader
reads all YAML fragments from one resolved repository revision into one
in-memory index. Content additions and corrections update only this external
repository; they do not require changes to `every_eval_ever`, adapters, or the
validator.

Production consumers must resolve a registry reference to an immutable commit
before validation begins:

```yaml
registry:
  repository_type: huggingface_dataset
  repository_id: evaleval/model-registry
  revision: 0123456789abcdef
```

A workflow may begin from a reviewed branch such as `main`, but it must resolve
that branch to a commit and use the same commit for the entire run. The
resolved repository ID, commit, schema version, and content digest should be
included in validation output.

Registry retrieval is fail-closed. An unavailable repository, unsupported
schema version, digest mismatch, partial download, or missing configured
revision blocks validation. A local cache is allowed only when it exactly
matches the requested immutable revision and digest; consumers must not
silently fall back to an arbitrary stale snapshot.

The existing ingestion manifest may eventually record one registry content
digest for reproducibility. This should be a reference to the authoritative
registry state, not another generated model registry.

## Rollout

1. Create the external registry repository and its review policy.
2. Add the package loader, supported schema contract, and resolution tests
   without changing datastore acceptance.
3. Generate a read-only report of current identifiers and ambiguities.
4. Seed reviewed entries collection by collection in the external repository.
5. Update adapters to use the shared resolver and preserve raw identifiers.
6. Make unresolved or mismatched model instances blocking in the validator.
7. Backfill canonical instance IDs and rebuild semantic fingerprints once.

The blocking switch should happen only after the collections in scope have
reviewed coverage. Once enabled, there must be no legacy fallback path.

## Future Extension

Metric and dataset registries can later follow the same principles:

- stable canonical entities;
- source-scoped identifiers;
- exact resolution;
- manually reviewed unknowns;
- blocking unresolved inputs;
- one shared package implementation.

They are deliberately excluded from the first implementation so the model
identity and deployment contract can be validated independently.
