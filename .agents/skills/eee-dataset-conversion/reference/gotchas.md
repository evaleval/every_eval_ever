# Gotchas that cost real time (with fixes)

*Scope: deeper failure modes and mechanisms. For what each field means, see
`fields.md` / `instance-level.md`; this file is the "why it bit us" layer.*

- **`inf` bounds — settled: emit `±inf`, don't hand-roll it.** A genuinely unbounded
  `continuous` metric (PSNR, perplexity, WER) sets `min_score`/`max_score` to
  `float('-inf')`/`float('inf')`. The library serializes those as the JSON strings
  `"Infinity"`/`"-Infinity"` (a `field_serializer` on the bounds, every_eval_ever#212 —
  valid RFC-8259 JSON that reads back to a float), so just save through
  `save_evaluation_log`/`model_dump_json` and it round-trips + validates. Do **not**
  hand-roll `json.dumps(..., allow_nan=True)`: that writes a bare `Infinity` token,
  which the strict read path (#212) now rejects. `null` ≠ unbounded — `null` is
  "not provided" and still fails `continuous`. Give a finite bound only when a real
  nominal scale exists (`[0,1]`/`[0,100]`); don't invent one for an open-ended metric.
- **Model deployment axes — the library hides your omission.** It auto-fills both to
  `"unknown"`, so a green *library* validate says nothing about whether you set them;
  the CLI errors. Enums and the vocabulary-skew warning: `datastore-gate.md` §deployment.
- **Aggregate vs parts double-counting** — emitting an overall *and* subtasks lets a
  consumer double-count. Mark the level; when a benchmark has ≤1 subtask emit only
  the overall.
- **micro vs macro** — emit the overall (micro, item-pooled) and every subtask
  with its `n` → both derivable downstream.
- **`additional_details` non-strings** → validation fail. Applies to instance
  string-maps too (`metadata`, `tool_calls[].arguments`, `performance.additional_details`).
  Trap: `validate` is pydantic (`dict[str, Any]`), so it will **not** flag non-strings
  that break the JSON schema.
- **Don't chase instance `metrics.num_turns`** — the schema's multi_turn `allOf`
  references a `metrics` property that doesn't exist; `num_turns` lives under
  `evaluation`. A top-level `metrics` object just trips `extra='forbid'`.
- **The percent-vs-proportion bug is caught — but only by the CLI.** A unit test calling
  `validate_file(path)` runs with semantic checks off, so it will happily pass a `73.4`
  under `0.0–1.0` bounds that the CLI rejects. Validate at the final path with the CLI
  (rule: `datastore-gate.md` §score).
- **A "successful" run that silently dropped rows.** Warn-and-continue turns a partial
  conversion into a green exit, and a lost row can quietly shrink an aggregate's
  denominator (1 success + 1 unparseable log → "1/1 = 100%"). Collect
  `SourceRecordFailure`s, write the report, exit non-zero — `datastore-gate.md`
  §partial conversions. Same class: a metric whose bounds you don't know, a model whose
  identity you can't resolve. Preserve the valid rows, record the rest, fail the command.
- **Re-running into a populated output dir.** `save_evaluation_log` mints a fresh uuid
  every call, so a second run adds a *second* logical record for the same evaluation
  instead of replacing it. `publish_evaluation_logs` refuses to overwrite (loud failure —
  good); if you write by hand, clear the target dir first and run
  `uv run python -m every_eval_ever.check_duplicate_entries <files>` before submitting.
- **Naive timestamps take the converter host's timezone.** A source datetime with no
  offset run through `.timestamp()` shifts by whatever TZ the machine has, so the same
  input converts differently on two machines — and `evaluation_id` moves with it. Attach
  the source's real offset (or UTC) explicitly.
- **Non-idempotent `evaluation_id`** — keying on `now`/`retrieved_timestamp` changes
  every run. Key on a stable value; for an unparseable timestamp, derive a stable
  token from the source path, never `now`. For a remote source (HF/API), pin the
  dataset commit SHA / revision into the id so reruns match even if a live lookup
  hiccups — and warn rather than silently falling back to `now`. Reuse the *same*
  pinned revision across multiple passes (aggregate + instances) so they can't drift.
- **Reading big parquet/JSON over HTTP** — `datasets-server` may be empty; `duckdb`
  httpfs chokes on large string columns; **pyarrow `read_row_group(..., columns=[...])`**
  via `HfFileSystem` streams. Project only the small columns for aggregation.
- **CI optional deps** — a `core` test matrix installs no extras; if your adapter needs
  `pyarrow`/`inspect_ai`/etc., guard the test with `pytest.importorskip("pyarrow")` so
  `core` skips instead of failing collection. (An adapter using only stdlib +
  the core package needs no guard.) `importorskip` only covers CI — also declare the
  optional dep (a `<name>` extra) and note `--all-extras` in the adapter's README, so a
  *fresh local run* fails with a clear signal, not a cryptic top-level `ImportError`.
  **After adding/declaring any dependency, regenerate the lockfile (`uv lock`) and
  commit it** — the `locked` CI matrix installs from `uv.lock` frozen (`uv sync
  --locked`) and fails the moment it drifts from `pyproject.toml`, even though the
  `loose` jobs (which re-resolve) pass. A green `loose` + red `locked` almost always
  means a stale lockfile.
  Declaring a new extra also means adding it to the **aggregate `all` extra**, or
  `every-eval-ever[all]` silently lacks your dependency.
- **Optional source fields that aren't dicts.** `(payload.get("methodology") or {}).get(...)`
  still raises when `methodology` is a string or a list. Guard with
  `isinstance(x, dict)` — malformed upstream rows are normal, not exceptional.
- **Smoke runs write into the repo.** Default your `--output-dir` to a temp path
  (`/tmp/<src>-smoke/data/<collection>`), not `data/<collection>` in the checkout:
  generated records belong in the HF datastore PR, and a refresh of `data/` in the code
  repo should be a deliberate, separate act. Give the adapter `--save-raw-json` /
  `--input-json` so a fetched payload can be replayed offline — that's also what makes the
  fixture-based test possible without mocking HTTP.
- **ruff is configured but not enforced by CI** — `pyproject.toml` selects E/F/I (E501
  and E402 ignored); no workflow runs it, so nothing will tell you but a reviewer. Run
  `uv run ruff check` yourself; fix import order, and use `# noqa: E402` after an
  `importorskip` block.
- **Stale helpers** — `helpers.make_evaluation_log`/`make_evaluation_result` miss the
  now-required `eval_library`/per-result `source_data`; build the models by hand.
