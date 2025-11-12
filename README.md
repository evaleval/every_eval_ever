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

To run the pre-commit specific files:

```sh
uv run pre-commit run --files /path/to/data/a.json /path/to/data/b.json
```

To optionally install the pre-commit so that it will run when before `git commit` (optional):

```sh
uv run pre-commit install
```
