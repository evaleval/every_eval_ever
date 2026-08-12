# Contributing to Every Eval Ever

Thanks for contributing! This repo defines the **Every Eval Ever (EEE)** schema and hosts the tooling around it — validation, converters, and adapters. The evaluation data itself lives in the [Hugging Face datastore](https://huggingface.co/datasets/evaleval/EEE_datastore), split into files by individual model and stored using [`eval.schema.json`](every_eval_ever/schemas/eval.schema.json) under `data/{benchmark_name}/{developer_name}/{model_name}/`.

For what the schema *is* and how to read a record, start with the [README](README.md) — [The Schema in Practice](README.md#-the-schema-in-practice), [Instance-Level Data](README.md#-instance-level-data), and [`eval.schema.json`](every_eval_ever/schemas/eval.schema.json) itself. This guide covers authoring and submitting: how your PR gets reviewed, how to submit data, and what to do when you change the schema or the tooling.

> **Using an AI coding agent?** This repo ships an `eee-dataset-conversion` skill ([`.agents/skills/eee-dataset-conversion/`](.agents/skills/eee-dataset-conversion/), indexed from [AGENTS.md](AGENTS.md)) that walks an agent through a conversion — schema field traps, aggregate + instance records, verification, and a decision log to paste into the PR. The rules below apply to agent-authored PRs too, and the agent is expected to follow them.

## 🔍 How your PR gets reviewed

Which lane you land in depends on the change, not on who wrote it.

**Fast — reviewed and merged quickly, sometimes by an agent.** No conflicts with `main`; scoped to one adapter, one file, or the tests; understood and verified by you; under ~1000 hand-written lines. A new adapter, a small validator behaviour change, a workflow fix. A maintainer can also put any PR in this lane with the **`auto-review`** label — ask in your description if you think yours belongs here.

**Needs a human** — either of the PR or of an agent's review of it — if any of the above doesn't hold, or for: a design change · a change spanning packages · a large refactor · more than ~1000 hand-written lines · a material change in outcome, where the same input now produces different data · a large agent-authored change. In practice: schema changes, edits to base adapters, changing or dropping a validator rule, restructuring.

**Planning something structural? Agree the approach before opening a PR.** Open an issue or raise it with a maintainer. We don't have the capacity to review a large structural PR that arrives cold, so one opened ahead of that agreement will sit — which wastes your work, not ours.

*Hand-written lines* excludes the generated types (`every_eval_ever/eval_types.py`, `instance_level_types.py`) and fixtures under `tests/data/`. Generated files anywhere else count in full: bulk output landing in the package is itself a thing to review, not a line-count technicality. And the threshold triages — don't split one coherent change into three to duck under it.

## 📥 Submitting evaluation data

New data is contributed to the [Hugging Face datastore](https://huggingface.co/datasets/evaleval/EEE_datastore).

### TL;DR

1. Data must conform to [`eval.schema.json`](every_eval_ever/schemas/eval.schema.json) (current version: `0.3.0`, exported as `every_eval_ever.helpers.SCHEMA_VERSION` — read it from there rather than hardcoding).
2. The validation pipeline verifies data submitted in a pull request automatically, and can also be triggered manually by commenting ```/eee validate changed``` on the HF PR. Run [validation](README.md#-data-validation) locally first.
3. An EvalEval member will review and merge your submission.

### PR naming convention

Use these prefixes in your pull request titles:

- `[Submission]` — New evaluation data
- `[Issue #N]` — Fix for a specific GitHub issue
- `[Feature]` — New functionality not tied to an issue
- `[Docs]` — Documentation changes
- `[ACL Shared Task]` — Shared task submissions (priority review)

### UUID naming convention

Each JSON file is named with a **UUID (Universally Unique Identifier)** in the format `{uuid}.json`. The UUID is generated (using standard UUID v4) when creating a new evaluation result file. This ensures that:

- **Multiple evaluations** of the same model can exist without conflicts (each gets a unique UUID)
- **Different timestamps** are stored as separate files with different UUIDs (not as separate folders)
- A model may have multiple result files, with each file representing different iterations or runs of the leaderboard/evaluation
- UUIDs can be generated using Python's `uuid.uuid4()` function.

**Example**: The model `openai/gpt-4o-2024-11-20` might have multiple files like:

- `e70acf51-30ef-4c20-b7cc-51704d114d70.json` (evaluation run #1)
- `a1b2c3d4-5678-90ab-cdef-1234567890ab.json` (evaluation run #2)

Note: each file can contain multiple individual results related to one model. See [examples in the datastore](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data).

### How to add a new eval

1. Add a new folder under [`data/`](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data) on the Hugging Face datastore with a codename for your eval.
2. For each model, use the Hugging Face (`developer_name/model_name`) naming convention to create a 2-tier folder structure.
3. Add a JSON file with results for each model and name it `{uuid}.json`.
4. [Optional] Add scripts used to generate the data under [`every_eval_ever/adapters/`](every_eval_ever/adapters/) in this repository (see e.g. [`every_eval_ever/adapters/global_mmlu_lite/adapter.py`](every_eval_ever/adapters/global_mmlu_lite/adapter.py)).
5. [Submit] Two ways to submit your evaluation data:
   - **Option A: Drag & drop via Hugging Face** — Go to [evaleval/EEE_datastore](https://huggingface.co/datasets/evaleval/EEE_datastore) → click "Files and versions" → "Contribute" → "Upload files" → drag and drop your data → select "Open as a pull request to the main branch". See [step-by-step screenshots](https://docs.google.com/document/d/1dxTQF8ncGCzaAOIj0RX7E9Hg4THmUBzezDOYUp_XdCY/edit?usp=sharing).
   - **Option B: Upload via `huggingface_hub`** — Useful for larger submissions or many files.

     ```python
     from huggingface_hub import HfApi

     api = HfApi()

     pr_url = api.upload_folder(
         folder_path="data/my-eval",
         path_in_repo="data/my-eval",
         repo_id="evaleval/EEE_datastore",
         repo_type="dataset",
         commit_message="[Submission] Add my eval",
         commit_description="Adds evaluation data for my eval.",
         create_pr=True,  # opens a PR instead of committing directly
     )

     print(pr_url)
     ```

     To add files to that PR rather than opening another, pass the PR ref Hugging Face
     returned as `revision` (e.g. `revision="refs/pr/XX"`) to `upload_file` or a further
     `upload_folder`.

Before opening the PR, validate locally — see [Data Validation](README.md#-data-validation).

### Automated (cron) submissions

Some data arrives without a person: a daily GitHub Actions run refreshes each
supported adapter and opens **one pull request per adapter**, reused across days
(`[Submission] cron: <adapter> (automated ingestion)`). Every record it publishes
carries, in `source_metadata.additional_details`:

| Key | Value |
|---|---|
| `type_of_addition` | `cron` |
| `cron_run_date` | the UTC date the source was pulled |
| `cron_adapter` | the adapter that produced it |
| `cron_run_url` | the workflow run, when available |
| `cron_unknown_inferred_fields` | which of `deployment_type` and `model_availability` came out `unknown`; absent when both are known |

That is what turns a later correction, "redo everything this adapter published that
day", into a query rather than a scan. Nothing else about the record differs from a
hand-submitted one, and the same validator gates it.

The run keeps a snapshot of each source it fetched, so a record can be checked
against the input it came from. Operating it, adding an adapter to the schedule, and
its limitations are documented in
[`every_eval_ever/cron/README.md`](every_eval_ever/cron/README.md).

## 🧾 Filling in the schema

Conventions for authoring records. For what each field *means*, see [The Schema in Practice](README.md#-the-schema-in-practice) and [`eval.schema.json`](every_eval_ever/schemas/eval.schema.json); the schema is always the source of truth.

1. **`model_info`**: Use Hugging Face formatting (`developer_name/model_name`). If a model does not come from Hugging Face, use the exact API reference. Check [examples in data/livecodebenchpro](https://huggingface.co/datasets/evaleval/EEE_datastore/tree/main/data/livecodebenchpro). Notably, some do have a **date included in the model name**, but others **do not**. For example:

- OpenAI: `gpt-4o-2024-11-20`, `gpt-5-2025-08-07`, `o3-2025-04-16`
- Anthropic: `claude-3-7-sonnet-20250219`, `claude-3-sonnet-20240229`
- Google: `gemini-2.5-pro`, `gemini-2.5-flash`
- xAI (Grok): `grok-2-2024-08-13`, `grok-3-2025-01-15`

2. **`evaluation_id`**: Use `{benchmark_name/model_id/retrieved_timestamp}` format (e.g. `livecodebenchpro/qwen3-235b-a22b-thinking-2507/1760492095.8105888`).

3. **`inference_platform`** vs **`inference_engine`**: Where possible specify where the evaluation was run using one of these two fields.

- `inference_platform`: Use this field when the evaluation was run through a remote API (e.g. `openai`, `huggingface`, `openrouter`, `anthropic`, `xai`).
- `inference_engine`: Use this field when the evaluation was run locally. This is an object with `name` and `version` (e.g. `{"name": "vllm", "version": "0.6.0"}`).

4. The `source_type` on `source_metadata` has two options: `documentation` and `evaluation_run`. Use `documentation` when results are scraped from a leaderboard or paper. Use `evaluation_run` when the evaluation was run locally (e.g. via an eval converter).

5. **`source_data`** is specified per evaluation result (inside `evaluation_results`), with three variants:

- `source_type: "url"` — link to a web source (e.g. leaderboard API)
- `source_type: "hf_dataset"` — reference to a Hugging Face dataset (e.g. `{"hf_repo": "google/IFEval"}`)
- `source_type: "other"` — for private or proprietary datasets

6. The schema accommodates both numeric and level-based (e.g. Low, Medium, High) metrics. For level-based metrics, the actual `value` should be converted to an integer (e.g. Low = 1, Medium = 2, High = 3), and the `level_names` property should be used to specify the mapping of levels to integers.

7. **Timestamps**: The schema has three timestamp fields — use them as follows:

- `retrieved_timestamp` (required) — when this record was created, in Unix epoch format (e.g. `1760492095.8105888`)
- `evaluation_timestamp` (top-level, optional) — when the evaluation was run
- `evaluation_results[].evaluation_timestamp` (per-result, optional) — when a specific evaluation result was produced, if different results were run at different times

8. Additional details can be provided in several places in the schema. They are not required, but can be useful for detailed analysis.

- `model_info.additional_details`: any additional information about the model itself (e.g. number of parameters)
- `evaluation_results.generation_config.generation_args`: additional arguments used to generate outputs from the model
- `evaluation_results.generation_config.additional_details`: any additional information about the evaluation process that is not captured elsewhere

## 🔧 Changing the schema, the validator, or the publisher

These change what a *contribution* has to look like, so they are slow-lane by default, and the contributor-facing guidance is part of the change rather than a follow-up.

1. **Regenerate the Pydantic types** and commit the result — the commands are in [Auto-generation of Pydantic Classes](README.md#-auto-generation-of-pydantic-classes-for-schema).

2. **Run `tests/test_skill_conversion.py`.** It re-validates the [`eee-dataset-conversion` skill](.agents/skills/eee-dataset-conversion/SKILL.md) — its templates and one frozen reference conversion — through the real CLI with semantic checks on. When it goes red, fix the **skill**, not the test: the failure message names the file and gives the regeneration command. This test is the reason there is no checklist of skill files to update here; it catches the drift that a checklist used to be relied on for.

3. **Consider a `schema_version` bump.** A field-optionality change moved it `0.2.2 → 0.2.3` (see #212), and it has since moved to `0.3.0`. Whether a given change warrants a bump is a judgement call, not a rule — decide, and flag it in the PR so a reviewer can weigh in.

## 🧩 A contribution is usually three PRs

A dataset contribution normally spans three repos: the adapter here, the canonical ids in [`eval-card-registry`](https://github.com/evaleval/eval-card-registry), and the data in the [`EEE_datastore`](https://huggingface.co/datasets/evaleval/EEE_datastore). Cross-link them so a reviewer can see the whole change — each one on its own looks incomplete. The skill's "three PRs" section has the details.
