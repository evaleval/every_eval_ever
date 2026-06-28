# Every Eval Ever Docs

Every Eval Ever is a schema, validator, converter set, and public datastore workflow for AI evaluation results.

Use this site to decide how a result should be shaped, checked, and submitted.

## Choose A Path

- Adding data by hand: start with [Getting Started](/getting-started/) and [Data Model](/data-model/).
- Checking files before or after a PR opens: use [Validation](/validation/).
- Converting framework logs: use [Converters](/converters/).
- Preparing Hugging Face Community Evals YAML: use [HF Community Evals](/hf-community-evals/).
- Opening the final datastore PR: follow [Contributing](/contributing/).

## Basic Flow

1. Shape the data as aggregate JSON, optionally with samples JSONL.
2. Validate locally with `uv run every_eval_ever validate`.
3. Open a Hugging Face datastore PR.
4. Let the PR bot report schema, path, and duplicate findings.
5. Fix explicit errors before review.

## Useful Links

- Public datastore: `evaleval/EEE_datastore`
- Public docs route: `https://evalevalai.com/projects/every-eval-ever/docs/`
- Local docs build: `uv run tools/build_docs.py --base-url /projects/every-eval-ever/docs --site-url https://evalevalai.com --output _site`
