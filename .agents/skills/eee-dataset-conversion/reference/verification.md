# Verification — the pre-PR gate

*Scope: the checks to run before opening a PR. One list — copy it into the PR description.*

- [ ] **Validate**: `uv run python -m every_eval_ever validate <file.json> [<file.jsonl> ...]` (or a glob like `'data/<src>/**/*.json'`) → all pass (`.json`→aggregate, `.jsonl`→instance). Pass **files or a glob, not a directory** — the CLI rejects a bare dir. Run it **at the final `data/<collection>/<dev>/<model>/` path: only there does the CLI run the semantic** checks that the library `validate()` skips — the merge gate, enumerated in `reference/datastore-gate.md`. A green in-library validate is necessary, not sufficient.
- [ ] **Offline unit test**: `uv run pytest tests/test_<name>_adapter.py` — fixture-based, no network; guard optional deps (`pyarrow`/`inspect_ai`) with `pytest.importorskip` so `core` CI skips. Assert any derived math (e.g. `standard_error`, aggregate == mean of item scores) against a hand-computed value so it can't silently drift.
- [ ] **Full suite**: `uv run pytest tests` — no regressions.
- [ ] **Lint**: `uv run ruff check every_eval_ever/adapters/<name>/ tests/test_<name>_adapter.py` — clean.
- [ ] **Live smoke run** on a slice → validate the real records.
- [ ] **Ids resolve**: model + benchmark ids resolve in the eval-card-registry (or the alias PR is prepared).
- [ ] **Gate items** (`reference/datastore-gate.md`, all hard errors): path is `data/<collection>/<dev>/<model>/{UUID4}.json` with portable components · `deployment_type`/`model_availability` set to the real values, not an inherited `"unknown"` · every score finite and inside its declared `[min_score, max_score]` (percent vs proportion!) · `continuous` has both bounds · uncertainty values finite · `metric_id` is a registry-canonical id for a global metric, or a namespaced one for a leaderboard-specific construct — never a bare `score`/`rank` · `hf_dataset` source_data carries `hf_repo` · judge metrics carry `judges[].model_info` + the real `input_prompt`.
- [ ] **Published, not hand-written**: records emitted via `save_evaluation_logs` (+ `EvaluationLogOutput`, for aggregate-only) or `publish_evaluation_logs` (when there are instance sidecars) — both batch-validate and roll back; output dir was empty beforehand; `uv run python -m every_eval_ever.check_duplicate_entries <files>` is clean.
- [ ] **Every dropped row accounted for**: `SourceConversionResult` failures/exclusions written to `adapter_reports/<collection>_failures.json` and the command exits non-zero on failures. The report's numbers match the PR's coverage line.
- [ ] **Content spot-check** (validating ≠ correct): `input.raw` doesn't leak the answer · aggregate not double-counted · `metric_name` is a metric not the eval · `source_data` is the dataset not the results · `evaluation_id` is stable (not keyed on `now`) and distinguishes same-model variants (effort/temperature/scaffold/date) · the metric you named is the metric upstream computed (`pass@k` ≠ `pass^k`) · the rows you converted are the rows the published number covers (same split/config/n).
- [ ] **Decisions & coverage logged** (SKILL.md step 7): every non-obvious choice (+ the alternative, + confidence) is in the PR; coverage is stated as "N source rows → M records, K dropped (reason)" with no silent caps; the operator was asked about any policy call (new canonical id, big data drop, ambiguous/unbounded metric, re-hosting).
- [ ] **Instances** (if any): every line has `evaluation_id` (== aggregate), `model_id`, `evaluation_name`, `sample_id`; `answer_attribution` is a 5-field list; a sample scored by K metrics = K records, one per aggregate result (no single record spanning multiple results); `interaction_type` XOR holds.
- [ ] **Sidecar link**: `detailed_evaluation_results` has `hash_algorithm=sha256`, `file_path` = the full repo-relative `data/<collection>/<dev>/<model>/<uuid>_samples.jsonl` (via `datastore_repo_file_path`, **not** a basename), checksum over the written bytes, `total_rows` = real line count; `sample_hash` uses the canonical-JSON recipe.
- [ ] **Datastore PR mechanics** (`reference/datastore-submission.md`): one collection per source · large uploads batched · iterate on the *same* PR ref · `/eee validate changed` run and its warnings cleared, not just its verdict · adapter code cross-linked, never vendored into the data PR.

## Prove the skeleton (don't just eyeball it)
Construct one record exactly as the template prescribes, publish it to a real
`data/<collection>/<dev>/<model>/` path, and run the CLI validator on the file.
Syntax-OK ≠ schema-valid, and schema-valid ≠ gate-clean: only the CLI at the final path
runs the semantic checks. `tests/test_skill_conversion.py` does exactly this for the
templates and for a frozen reference conversion on every CI run — if it fails after a
schema or validator change, the skill is what needs fixing.
