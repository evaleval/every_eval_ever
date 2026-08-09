# AGENTS.md — every_eval_ever

Entry point for coding agents in this repo. (Open standard — see https://agents.md.)
Repo-local agent skills live under `.agents/skills/`.

This repo defines the **Every Eval Ever (EEE)** schema and hosts **adapters** that
convert external eval sources into it.

## Skills (agent-invoked, loaded on demand)
| Skill | Use when |
|---|---|
| [`eee-dataset-conversion`](.agents/skills/eee-dataset-conversion/SKILL.md) | Converting a dataset/leaderboard into EEE; writing/fixing an adapter under `every_eval_ever/adapters/`; debugging why an EEE record won't validate |
| [`eee-datastore-pr-review`](.agents/skills/eee-datastore-pr-review/SKILL.md) | Reviewing or repairing an `EEE_datastore` Hugging Face PR; reproducing `/eee validate changed`; researching validator warnings and model deployment metadata |

## Layout
- `every_eval_ever/eval_types.py` + `eval.schema.json` — aggregate `EvaluationLog`.
- `every_eval_ever/instance_level_types.py` + `instance_level_eval.schema.json` — instance log.
- `every_eval_ever/adapters/<name>/adapter.py` — one-off source adapters (run via `uv run python -m every_eval_ever.adapters.<name>.adapter`).
- `every_eval_ever/converters/` — in-tree converters (`inspect`/`helm`/`lm_eval`, plus `alpaca_eval`; shared code in `common`), run via `uv run python -m every_eval_ever convert <inspect|helm|lm_eval> ...`.
- `every_eval_ever/validator/` — the schema + **semantic** merge gate (path shape, UUID4 names,
  companion pairing, score bounds, deployment axes). `REGISTERED_CHECKS` is the list.
- Validate: `uv run python -m every_eval_ever validate <files-or-glob>` (`.json`→aggregate,
  `.jsonl`→instance; directories are rejected).

## Principles (tie-breakers, for when the conventions below don't decide)
- **Report, don't interpret.** The job is the most correct data in a unified form. Record
  what the source states; never infer or tidy a value to make a record look complete. An
  unknown is data; a guess is corruption. If the source is ambiguous, ask in the PR.
- **Prefer a check to an instruction.** If a test or a CI job can enforce a rule, add that
  instead of writing the rule down — and delete the prose it replaces.
- **Don't spend a contributor's attention on what a machine can verify.** No checklist
  item for something CI could check.

## Conventions (non-negotiable)
- **The schemas are the source of truth.** When a doc and a schema disagree, the schema wins.
- **Use uv everywhere in documentation.** Install with `uv add`/`uv sync` and run
  project commands with `uv run`; do not document bare `python`, `pip`, `pytest`, or
  `ruff` commands.
- **Validating ≠ correct.** Everything must pass `validate`, but spot-check *content*
  (no answer leakage, no double-counted aggregates, `metric_name` is a metric, stable `evaluation_id`).
- **Tests/lint**: add an offline, fixture-based `tests/test_<name>_adapter.py`, guard
  optional deps so the `core` CI matrix skips cleanly, and keep `uv run ruff check` green —
  see the skill's `reference/verification.md` and `reference/gotchas.md` for the exact mechanics.
- **Docstrings say what, not why-I-changed-it.** A docstring documents what the function
  does and what a caller must know. Rationale for a change, what it replaced, and notes
  aimed at a reviewer belong in the **PR description** — a maintainer reads that once,
  instead of reading it every time they open the file. Where a future editor really would
  break something, leave a one-line `#` comment at that line, not a paragraph in the
  docstring. If it needs a paragraph, it needs the PR.
- **Publish through the repo, not by hand**: `converters.common.publication.publish_evaluation_logs`
  writes records atomically; `SourceConversionResult` + `save_failure_report` + a non-zero
  exit account for every source row you could not convert.
- A dataset contribution is usually three PRs (adapter here · ids in `eval-card-registry`
  · data in `EEE_datastore`) — cross-link them. See the skill's "three PRs" section.
- **Get agreement before building something structural.** A scoped change — one adapter,
  one file, tests — can go straight to a PR. A design change, a cross-package change, a
  large refactor, or anything that changes what existing data comes out as needs its
  approach agreed *first*, in an issue or with a maintainer — opening the PR is not how you
  start that conversation, and a structural PR that arrives cold will sit.
  Full policy: `CONTRIBUTING.md` § "How your PR gets reviewed".

## Changing the schema, the validator, or the publisher
These change what a *contribution* must look like, so the contributor-facing guidance is
part of the change — not a follow-up.
- `tests/test_skill_conversion.py` is the tripwire: it re-validates the skill's templates
  and one frozen reference conversion through the real CLI (semantic checks on). Run it.
- When it fails, fix the **skill**, not the test: the rule lives in
  `.agents/skills/eee-dataset-conversion/reference/` (gate rules in `datastore-gate.md`,
  field semantics in `fields.md`), the emitting code in `templates/`. Then regenerate the
  frozen conversion with the command in the failure message.
- Don't restate the new rule anywhere else. The test is the enforcement; prose that
  duplicates it is what went stale last time.

## Human docs
`README.md` (understand & use), `CONTRIBUTING.md` (author & submit) and
`every_eval_ever/adapters/README.md` (adapter authors) are for people. Keep agent
instructions here and in `.agents/skills/`.
