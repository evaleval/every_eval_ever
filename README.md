# every_eval_ever

## Data Validation

This repository has a pre-commit that will validate that JSON files conform to the JSON schema. The pre-commit requires using [uv](https://docs.astral.sh/uv/) for dependency management.

To run the pre-commit on git staged files only:

```sh
uv run pre-commit run
```

To run the pre-commit on all files:

```sh
uv run pre-commit run --all-files
```

To run the pre-commit on specific files:

```sh
uv run pre-commit run --files a.json b.json c.json
```

To install the pre-commit so that it will run before `git commit` (optional):

```sh
uv run pre-commit install
```
