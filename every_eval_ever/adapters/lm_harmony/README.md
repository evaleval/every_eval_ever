# LM-Harmony

Converts the results matrix behind *Train-before-Test Harmonizes Language Model
Rankings* ([arXiv:2507.05195](https://arxiv.org/abs/2507.05195)) into
`data/lm-harmony/`.

The source is one file — `notebooks/all_results.json` in
[socialfoundations/lm-harmony](https://github.com/socialfoundations/lm-harmony) —
shaped `{block: {task: {hf_model_id: score}}}` over four blocks, 27 tasks and 61
models, with a published standard error for every cell.

```bash
uv run python -m every_eval_ever.adapters.lm_harmony.adapter \
  --output-dir /tmp/lm-harmony-smoke/data/lm-harmony \
  --save-raw-json /tmp/lm-harmony-raw.json

uv run python -m every_eval_ever validate \
  '/tmp/lm-harmony-smoke/data/lm-harmony/*/*/*.json'
```

Replay the saved payload with no network. The revision is required, because the
records cite the commit their bytes came from.

```bash
uv run python -m every_eval_ever.adapters.lm_harmony.adapter \
  --input-json /tmp/lm-harmony-raw.json \
  --revision 88bc5c352f6491bc9e6c19a361096a31c4df6e16 \
  --output-dir /tmp/lm-harmony-replay/data/lm-harmony
```

## Two protocols, one benchmark

`direct_eval` is the ordinary zero-shot lm-evaluation-harness number
(`simple_evaluate(num_fewshot=0)`). Under `train_before_test` the model is
fine-tuned on the task's own training split first and then evaluated, which is the
paper's contribution and a different quantity.

Both are published, so both are converted. They share an `evaluation_name`
(`lm_harmony.<task>`), because they are the same benchmark, and they are kept
apart by `metric_id`: only `direct_eval` takes the canonical global id, while
`train_before_test` is namespaced `lm_harmony.train_before_test.<metric>`. The
protocol also appears in `evaluation_result_id` and in `metric_config`. A consumer
joining on `accuracy` therefore cannot pool a zero-shot score with a task-trained
one.

## Metrics

Per-task metric selection follows the repository's own
`notebooks/analyze_results.ipynb`.

| Metric | Tasks | `metric_id` | Bounds |
|---|---|---|---|
| `acc` | 12 GLUE/SuperGLUE-style and commonsense tasks | `accuracy` | `[0, 1]` |
| `acc_norm` | 9 multiple-choice tasks, incl. `mathqa` and `sciq` | `normalized-accuracy` | `[0, 1]` |
| `exact_match` | `gsm8k`, `nq_open` | `exact-match` | `[0, 1]` |
| `mcc` | `cola` | `matthews-correlation` | `[-1, 1]` |

`cola`'s bounds are the one that bites. Matthews correlation is on `[-1, 1]` and
nine of the 61 models score below zero, so `[0, 1]` bounds would be a hard
validator error rather than a rounding quibble.

`acc_norm` is length-normalized accuracy, a different computation from `acc` on
the same items, and the registry keeps the two apart — it is `normalized-accuracy`
there, with `acc_norm` already an alias. The hosted resolver returns no match for
it because the live Space lags the seed data, so resolve metric ids against the
seed rather than trusting a `no_match` from the API.

## Dataset provenance

Every task's dataset, config and scored split come from the `task:` field of the
vendored `lm_eval/tasks/**.yaml` at the pinned commit, never from a filename —
`qnli.yaml` also exists under `basqueglue/` and `social_iqa.yaml` under
`bigbench/`, so matching by path would silently convert the wrong dataset.

Eleven of the sixteen dataset paths are legacy Hub names that redirect today
(`sciq` → `allenai/sciq`, `glue` → `nyu-mll/glue`, `math_qa` → `allenai/math_qa`).
`hf_repo` carries the resolved repo, and the string the harness actually asked for
stays in `additional_details.lm_eval_dataset_path`.

Two caveats travel with every result rather than being tidied away. lm-eval scores
what it calls a task's test docs, which for a benchmark that withholds test labels
is its validation split. And the published runs cap the scored split with
`--dataset_param.max_num_test` under a seeded permutation, so a split larger than
the cap was scored on a random subsample of it. The source does not state the
resulting n per task, so `num_samples` is left unset rather than derived.

## What is not converted

The matrix also carries `wiki_2025`, `arxiv_2025` and `stackexchange_2025` — the
paper's post-cutoff perplexity corpora, scored as `bits_per_byte` for 53 of the 61
models. The repository commits no task definition for them, so the dataset the
scores cover cannot be named, and `bits_per_byte` resolves to no registry metric.
They are recorded as exclusions in the conversion report rather than published
under an invented dataset.

A task that appears upstream but is absent from this adapter's table is a
**failure**, not a skip, and the run exits non-zero. Its dataset, split and metric
would otherwise be guesses.

## Coverage

24 tasks × 61 models × 2 protocols. The 2026-09-02 conversion of commit
`88bc5c352f` read 3,087 source cells and wrote 61 records carrying 2,928 results,
0 dropped, 3 tasks excluded. All 61 pass the CLI validator with semantic checks on
and no warnings.
