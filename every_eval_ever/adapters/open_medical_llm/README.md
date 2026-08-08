# Open Medical-LLM Leaderboard adapter

Converts the [Open Medical-LLM Leaderboard](https://huggingface.co/spaces/openlifescienceai/open_medical_llm_leaderboard)
results into Every Eval Ever aggregate logs.

- **Source (data):** HF dataset [`openlifescienceai/results`](https://huggingface.co/datasets/openlifescienceai/results),
  laid out as `<developer>/<model>/results_*.json` in lm-evaluation-harness output format.
- **Grain:** one `EvaluationLog` per model (`developer/model`), with one
  `EvaluationResult` per medical benchmark. Aggregates only — no per-item data.
- **Benchmarks (9):** MedMCQA, MedQA (USMLE 4-options), PubMedQA, and six MMLU
  medical subjects (anatomy, clinical knowledge, college biology, college
  medicine, medical genetics, professional medicine). Each result's
  `source_data` points at that benchmark's own HF dataset repo.
- **Metric:** `acc,none` → `metric_id: accuracy` (the registry's global metric —
  the benchmark is kept apart by `evaluation_name`), `continuous` in `[0, 1]`,
  higher is better; `acc_stderr,none` → `uncertainty.standard_error` when present.
- **Model identity:** the dataset path and the run config normally agree, and the
  path is used. When they disagree either can be the typo, so both are resolved
  through HuggingFace's alias redirect: two spellings of one repo give one answer
  (`aaditya/OpenBioLLM-Llama3-70B` and the config's `aaditya/OpenBioLLMLlama-70B`
  both resolve to `aaditya/Llama3-OpenBioLLM-70B`), and genuinely different repos
  mean the evaluated model is not recoverable — that row is reported as
  unconvertible rather than attributed by preference.
- **Model metadata:** `model_args.pretrained` is the harness loading a checkpoint
  locally, so `deployment_type=self_deployed` and
  `model_availability=open_weights` are set from it. Rows that record no
  checkpoint keep the `unknown` placeholders.
- **Provenance:** the leaderboard re-hosts numbers it did not itself produce, so
  `source_type="documentation"` and `evaluator_relationship=third_party`; the
  harness is unmistakably `lm-evaluation-harness` (from the `acc,none` /
  `acc_stderr,none` / `bootstrap_iters` keys), so `eval_library` names it.
- **evaluation_id** is keyed on the result file's own path and timestamp (parsed
  from the filename), so re-ingesting the same run is idempotent. It deliberately
  does *not* use the resolved model repo or the registry id: an alias redirect or a
  registry re-map moves those, and would hand one source file a second identity.
  Root-level 2-segment baselines (hand-curated closed-model paper numbers, e.g.
  `GPT-4/results_*.json`) have different provenance and are skipped; hidden dirs and
  Jupyter `*-checkpoint.json` files are filtered out.

## Run

```bash
uv run python -m every_eval_ever.adapters.open_medical_llm.adapter --output-dir /tmp/eee-omll [--limit N]
uv run python -m every_eval_ever validate /tmp/eee-omll
```

Options: `--output-dir` (default `data/open-medical-llm`), `--limit N` (first N
models; a negative value is rejected, and a selection that ends up empty stops the
run rather than reporting a refresh that wrote nothing), `--workers` (concurrent
fetches, default 8), `--no-registry-resolve` (skip the registry lookup and the HF
alias check), `--replace-existing`.

Every selected result file is accounted for: a file that yields no record is a
failure, not a skip, so it lands in `adapter_reports/` and the command exits
non-zero. The 5 hand-curated baselines are recorded as exclusions and do not fail
the run. The report is rewritten on every run, a clean one included — an earlier
run's copy left in place would read as this run's — and it is swapped in atomically,
so an interrupted write cannot replace a complete report with a truncated one.

Record filenames are fresh uuid4s, so a re-run over a populated output directory is
an error until `--replace-existing` is passed — otherwise it would add a second copy
of every `evaluation_id` rather than replace it. With that flag the prior records are
removed only once this run's records are on disk, so a validation or write failure
leaves the previous publication whole instead of a gap where it used to be.
