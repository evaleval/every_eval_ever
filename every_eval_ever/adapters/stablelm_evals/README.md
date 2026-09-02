# StableLM evals

Publishes the lm-evaluation-harness outputs Stability AI committed under `evals/` in
[Stability-AI/StableLM](https://github.com/Stability-AI/StableLM), into
`data/stablelm-evals/`.

**This is a wrapper, not a converter.** The in-tree `lm_eval` converter already
reads this format. Everything here is the three things a public repository of
harness JSON needs before the converter can be pointed at it, and it is worth
copying for the next such repository.

```bash
uv run python -m every_eval_ever.adapters.stablelm_evals.adapter \
  --output-dir /tmp/stablelm-evals-smoke/data/stablelm-evals

uv run python -m every_eval_ever validate \
  '/tmp/stablelm-evals-smoke/data/stablelm-evals/*/*/*.json'
```

## The three things a wrapper adds

**Pin the source.** The revision is resolved to a commit sha before any file is
fetched, so every record cites bytes that cannot move. `--allow-unpinned-source` is
required to proceed without one.

**Repair the model identity.** The converter takes `model_info.id` from
`config.model_args`'s `pretrained=` value and refuses an id with no publishing
namespace, which is right — a placeholder developer routes unrelated models into one
directory. 28 of the 29 files here name a full repo id. One
(`evals/stablelm-3b-4e1t.json`) was run from a local checkout, so its org is
supplied through `ORGLESS_MODEL_ORG`, where the decision is visible. An org-less id
that is *not* listed there fails the row rather than getting a guess.

**Pin one collection.** Left alone the converter files each task into its own bare
collection — `data/sciq/`, `data/piqa/`, `data/lambada_openai/` — which mixes these
numbers with every other source's records for the same benchmark and loses the
provenance. `publish_evaluation_logs(collection_override=...)` keeps the source
together.

## Coverage

29 models x 11 benchmarks, 293 records carrying 530 results, 0 dropped. The 2026-09-02
conversion of commit `93eea082c4` validates 293/293 through the CLI with semantic
checks on and no warnings.

Benchmarks: `arc_challenge`, `arc_easy`, `boolq`, `hellaswag`, `lambada_openai`,
`openbookqa`, `piqa`, `sciq`, `siqa`, `truthfulqa_mc`, `winogrande`.

Models, all open-weights: Pythia (2.8b-deduped, 6.9b, 12b), GPT-J-6B, GPT-NeoX-20B,
BLOOM (3b, 7b1), OPT (2.7b, 6.7b), LLaMA-1 via `huggyllama`, Llama-2 (7b, 13b),
MPT-7B, Falcon-7B, RedPajama-INCITE-7B-Base, Qwen-7B and Qwen-7B-Chat, Cerebras
BTLM-3B-8K, OpenLLaMA x3, Mistral-7B via `kittn`, phi-1.5, Baichuan2-7B-Base, and
five StableLM releases.

Two of those are the reason this source is worth having. It carries SciQ for 29
models and LAMBADA-OpenAI for 29, both of which the datastore held almost nothing
for.

## What is not converted

`evals/open_llm_leaderboard/` holds per-task files for two models rather than one
file per model — a different shape this wrapper does not read. Excluded by prefix
and reported on every run, so the 8 files are visible rather than silently absent.

## Notes on the source

These runs predate lm-eval v0.4: `versions` uses the integer form and metric keys
carry no `,filter` suffix, so `eval_library.version` is recorded as `0.3`. The
converter's handling of that format was wrong until #285 — standard errors were
published as scores — so records generated before that fix should be regenerated.

`evals/external/togethercomputer-RedPajama-INCITE-7B-Base2.json` names `Base2` in
its filename while `config.model_args` says
`pretrained=togethercomputer/RedPajama-INCITE-7B-Base`. The identity comes from
`model_args`, which is what the harness was actually given.

Two entries are community mirrors rather than first-party repos —
`huggyllama/llama-7b` for LLaMA-1 and `kittn/mistral-7B-v0.1-hf` for Mistral-7B.
They are published as the source ran them; aliasing a mirror onto its upstream is
the eval-card-registry's job, not this adapter's.
