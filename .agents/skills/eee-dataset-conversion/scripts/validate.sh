#!/usr/bin/env bash
# Validate EEE records (dispatches .json -> aggregate, .jsonl -> instance).
# `every_eval_ever validate` takes FILES or a fixed-depth glob, NOT a directory,
# so we expand any directory arg to its .json/.jsonl files ourselves.
# Usage: scripts/validate.sh <file-or-dir> [<file-or-dir> ...]
set -euo pipefail
[ "$#" -ge 1 ] || { echo "usage: validate.sh <file-or-dir> [...]" >&2; exit 2; }
files=()
for p in "$@"; do
  if [ -d "$p" ]; then
    while IFS= read -r -d '' f; do files+=("$f"); done \
      < <(find "$p" -type f \( -name '*.json' -o -name '*.jsonl' \) -print0)
  else
    files+=("$p")
  fi
done
[ "${#files[@]}" -ge 1 ] || { echo "no .json/.jsonl files under: $*" >&2; exit 2; }
# Validate the files at their canonical data/<collection>/<dev>/<model>/ path so the
# CLI's semantic checks (path structure, companion pairing, model-deployment fields)
# have the datastore context they need.
uv run python -m every_eval_ever validate "${files[@]}"
