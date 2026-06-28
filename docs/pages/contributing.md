# Contributing

Most data contributions land in the Hugging Face datastore, not in this repository. This repo provides the schema, validators, converters, and docs artifact that make those datastore contributions reviewable.

## Pull Request Names

Use a clear prefix in the datastore PR title:

| Prefix | Use |
| --- | --- |
| `[Submission]` | New evaluation data. |
| `[Issue #N]` | Fix tied to an existing issue. |
| `[Feature]` | New functionality not tied to one issue. |
| `[Docs]` | Documentation-only change. |
| `[ACL Shared Task]` | Shared task submission requiring priority review. |

## UUID Files

Each aggregate record is named `{uuid}.json`, where the UUID is unique to that evaluation record. If samples are included, the companion file is `{uuid}_samples.jsonl` in the same directory.

This allows multiple runs of the same benchmark and model to coexist without timestamp folders or filename collisions.

## Model Identifiers

Use Hugging Face-style `developer/model` identifiers when a model exists on Hugging Face. For API-only models, use the exact provider identifier used by the evaluation source, such as `openai/gpt-4o-2024-11-20` or `anthropic/claude-3-7-sonnet-20250219`.

## Submission Checklist

- Folder path follows `data/{benchmark_name}/{developer_name}/{model_name}/`.
- Aggregate JSON validates with [Validation](/validation/).
- Companion JSONL validates when present.
- `source_metadata.source_type` is `documentation` for scraped leaderboards or papers and `evaluation_run` for direct runs.
- `evaluation_results[].source_data` uses the correct `url`, `hf_dataset`, or `other` variant.
- `metric_config.lower_is_better` is explicit for every metric.
- `score_type: "continuous"` includes `min_score` and `max_score`.
- Level-based metrics include `level_names` and `has_unknown_level`.
- Timestamps are explicit and use the correct field for record retrieval versus evaluation execution.
- Converter output has been reviewed for placeholder metadata before submission.

## Open A Datastore PR

You can upload files through the Hugging Face web UI or with `huggingface_hub`. Use `create_pr=True` so the changes are reviewed before merge.

```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="data/my-eval",
    path_in_repo="data/my-eval",
    repo_id="evaleval/EEE_datastore",
    repo_type="dataset",
    commit_message="[Submission] Add my eval",
    commit_description="Adds evaluation data for my eval.",
    create_pr=True,
)
```

After opening the PR, include enough context for reviewers to trace source data, metric definitions, and any converter assumptions.

The PR bot described in [Validation](/validation/) will report schema, path, and duplicate findings on datastore PRs. Treat bot errors as blockers and bot warnings as review items that need an explicit decision.
