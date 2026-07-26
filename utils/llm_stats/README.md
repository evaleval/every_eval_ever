# LLM Stats Adapter

Converts LLM Stats model, benchmark, and score API data into Every Eval Ever aggregate `EvaluationLog` JSON files.

## API Key

Create an API key from the LLM Stats developer console:

https://llm-stats.com/developer

Set it in the shell where you run the adapter:

```bash
export LLM_STATS_API_KEY='your_key_here'
```

If the environment variable is not available, pass the key directly with `--api-key`.

## Live Export

Run the adapter from the repository root:

```bash
uv run python -m utils.llm_stats.adapter \
  --output-dir /tmp/eee-llm-stats/data/llm-stats \
  --save-raw-json /tmp/eee-llm-stats/raw
```

Validate the generated EEE logs:

```bash
uv run python -m every_eval_ever validate /tmp/eee-llm-stats/data/llm-stats
```

The default API base URL is `https://api.llm-stats.com`. Override it only when LLM Stats changes or provides a different API host:

```bash
uv run python -m utils.llm_stats.adapter \
  --base-url https://api.llm-stats.com \
  --output-dir /tmp/eee-llm-stats/data/llm-stats \
  --save-raw-json /tmp/eee-llm-stats/raw
```

## Offline Replay

When `--save-raw-json` is used with a directory, the adapter writes `combined.json` plus endpoint-specific snapshots. Re-run conversion without network access from the saved payload:

```bash
uv run python -m utils.llm_stats.adapter \
  --input-json /tmp/eee-llm-stats/raw/combined.json \
  --output-dir /tmp/eee-llm-stats-replay/data/llm-stats
```

Then validate the replayed output:

```bash
uv run python -m every_eval_ever validate /tmp/eee-llm-stats-replay/data/llm-stats
```

## Evaluator Provenance

LLM Stats is recorded as the aggregator. `evaluator_relationship` describes
the relationship between the original score source and the model developer:

- explicit `self_reported=true` means `first_party`;
- explicit `self_reported=false` means `third_party`;
- otherwise, a source organization matching the model developer means
  `first_party`, and a differing organization means `third_party`;
- records without usable evidence remain `other`.

When the benchmark fallback API omits this evidence, model pages are fetched
with bounded concurrency and a bounded per-request timeout. The raw source
fields, inferred relationship, and inference reason are preserved in score
details so shared validation can verify that they match the aggregate group.
