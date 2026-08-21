# Maintaining metric bounds for the Papers with Code adapter

*Scope: the one piece of this adapter that needs periodic human judgment —
giving every metric a canonical **min / max / direction** so the emitted
`EvaluationLog`s are valid and comparable. The adapter itself is documented in
the [adapters README](../README.md#papers-with-code).*

The short version: **run the health check, register the `RECURRING` bucket, and
leave the `BESPOKE` bucket to `--allow-unresolved` until you have time to read
papers.** That keeps data flowing and never ships a wrong bound.

---

## 1. Why this is manual at all

The PwC dump gives us, per metric: the **name**, a `direction` hint
(higher/lower better), a free-text unit/scale hint, an `evaluation_description`,
and a **paper URL**. It does **not** give bounds. But the EEE schema requires a
finite `min_score`/`max_score` for every `continuous` metric (the validator
raises otherwise), and comparability requires a *canonical* scale, not whatever
scale a given leaderboard happened to report in.

So bounds have to come from somewhere deterministic. That "somewhere" is the
**eval-card-registry** (`seed/metrics.yaml`). Populate it once from cited
definitions and every future run resolves names → bounds by static YAML lookup —
**no LLM at resolution time.** The registry is the durable, non-LLM source of
truth; an LLM/agent is only ever used *once*, at registration time, to read a
paper — and its answer is then frozen as a citation in the entry.

```
PwC dump ──► adapter ──► resolve(name) ──► registry_snapshot.json (snapshot of
                                            eval-card-registry seed/metrics.yaml)
                                          └► bounds + direction, deterministically
```

`registry_snapshot.json` is a **vendored snapshot**; regenerate it with
`refresh_registry_snapshot.py` after every registry change (see §5).

---

## 2. The health check (start here every time)

```bash
uv run python -m every_eval_ever.adapters.paperswithcode.adapter \
    --dump <dump.dump> --all --best-effort --output-dir /tmp/eee-pwc-check
```

`--best-effort` emits everything it can represent instead of aborting, and
**still prints the full imperfection report to stderr**. The unresolved section
is **triaged into three buckets** for you:

- **RECURRING** — the name matches a known standard family. A ~60-second registry
  add with the family's bound/direction (§3). Do these.
- **BESPOKE** — no family match; needs a read of the defining paper (§4). Do these
  when you have time; `--allow-unresolved` covers them meanwhile.
- **AMBIGUOUS** — the name matches *two* canonical ids. This is a duplicate
  alias/display_name in the registry, **not** a missing metric. Fix the collision
  in `seed/metrics.yaml` and refresh.

The triage uses `classify_metric_family()` in `adapter.py` — the same family
taxonomy that seeded the recurring metrics. It is a **hint, not an auto-rule**
(see the warning in §4).

---

## 3. RECURRING bucket — register a standard-family metric

These follow from the family definition, so you can register them without
reading a paper — but **confirm the bound matches the family** (a name can lie;
see §4). Family → canonical bound/direction:

| family       | min | max  | lower_is_better | typical names |
|--------------|-----|------|-----------------|---------------|
| `rate`       | 0.0 | 1.0  | false           | accuracy, F1, AP/mAP, AUROC/AUPRC, recall@k, IoU, R@k IoU=x, success rate, OLS |
| `pose-error` | 0.0 | null | true            | MPJPE, PVE/MVE, N-MPJPE, PA-PVE, MRPE, end-point error (mm) |
| `dist-error` | 0.0 | null | true            | MSE, RMSE, MAE, L1/L2 distance (m/mm) |
| `spec-loss`  | 0.0 | null | true            | Mel Loss, STFT distance, F0-RMSE |
| `psnr`       | 0.0 | null | false           | PSNR (dB) |
| `pesq`       | -0.5| 4.5  | false           | PESQ, PESQ-NB, PESQ-WB |
| `mos`        | 1.0 | 5.0  | false           | MOS, DNSMOS, PLCMOS, UTMOS, ViSQOL |
| `stoi`       | 0.0 | 1.0  | false           | STOI, ESTOI |
| `mcd`        | 0.0 | null | true            | MCD, mel-cepstral distortion |
| `gen-dist`   | 0.0 | null | true            | FID, FVD, rFVD, KID |
| `bitrate`    | 0.0 | null | true            | bpp, bpsp, bits-per-* |
| `bd`         | null| null | true            | BD-Rate, Bjontegaard-delta (signed %) |

`null` max = genuinely unbounded above (the adapter emits `inf`, which the schema
serializes as the JSON string `"Infinity"`). `null`/`null` = signed & unbounded.

**`rate` and the percent question.** Register `rate` metrics as **[0, 1]**. Many
leaderboards report them as percent (0–100); the adapter reconciles that *per
`(metric, dataset)` group* and rescales percent → proportion automatically
(`analyze_group` + `reconcile_scale`, §6). You do **not** register a second
[0,100] version. The only time you register a `> 1` max (e.g. `[0, 100]`) is a
metric that is *intrinsically* on that scale and never a proportion — i.e. a
metric whose real values legitimately exceed 1 (a bad-pixel percentage that runs
0–3%, an intrinsic-percent score). If you register such a metric as `[0, 1]` by
mistake, the adapter will not silently squash it: it flags the whole board
`group_scale_mismatch` (§6, outcome *anomaly*), which is your signal to
re-register it on its natural scale.

### Entry template

Append to the appropriate `# --- family: ... ---` group in
`eval-card-registry/seed/metrics.yaml`:

```yaml
- id: n-mpjpe                       # kebab-case, unique; the adapter also matches
  display_name: 'N-MPJPE'           # by normalized alias (case/-/space-insensitive)
  aliases:
  - 'N-MPJPE'                       # every surface form the dump uses
  score_type: continuous
  lower_is_better: true
  min_score: 0.0
  max_score: null                   # unbounded above
  metadata: '{"kind": "real", "confidence": "high", "family": "pose-error", "source": "3D joint position error (mm); [0,inf), lower better", "provenance": "paperswithcode-adapter"}'
  review_status: draft              # flip to reviewed once a human has verified
```

Then **refresh + re-run** (§5). Done.

---

## 4. BESPOKE bucket — read the paper

A bespoke metric is a paper-specific composite (e.g. `WorldScore-Dynamic`,
`Driving Score`, `EPDMS`, `PIE-Bench Background LPIPS`). There is no family
default; you must read its defining paper. This is the part that needs judgment.

> ### ⚠️ Why you cannot infer bounds from the name
> Name inference is unsafe **for bounds**, even when it is fine for triage:
> - `PIE-Bench Background LPIPS` — LPIPS is definitionally [0,1], but PIE-Bench
>   reports it **×10³**, so observed values are 62–304. The honest registry bound
>   is **[0, 1000]** (the scale the dump uses), not [0,1]. Only the paper tells
>   you the ×10³.
> - `BD-Rate (PSNR RGB)` — the name contains "PSNR", but it is a Bjontegaard
>   **rate**: signed, unbounded, lower-better — not a [0,∞) higher-better PSNR.
> - `S-BERT`, `F0-CORR`, cosine similarities — bounded **[-1, 1]**, not [0,1],
>   even though observed values sit in the positive range.
>
> This is exactly why the family table in §3 is a *hint* and every bound is
> **cross-checked against observed data** (§4.3) before it is trusted.

### 4.1 Find the paper

Each metric's paper URL travels with the data. In the emitted logs it is under
`score_details.details.paper_arxiv_url` / `source_url` /
`external_source_url`, and `source_data.additional_details.paper_url`. In the raw
dump it is on the evaluation/result row. A PwC `/paper/<id>` slug maps to
`arxiv.org/abs/<id>`; numeric PwC ids may need a PwC-page or title search.

### 4.2 Deduce the bound shape

Read for the metric's **definition** and decide the shape:

| the paper says… | register as |
|---|---|
| a probability / rate / fraction / normalized-to-[0,1] score | `[0, 1]`, higher better (unless it's an error) |
| a percentage the paper reports on 0–100 and never as a fraction | `[0, 100]` |
| a cosine / correlation | `[-1, 1]` |
| an error / distance / divergence, ≥ 0, no ceiling | `[0, null]`, lower better |
| a score in std-dev units, or a signed delta (can be negative, no ceiling) | `[null, null]` |
| a rubric score 1–N (MOS, GPT-judge 1–10) | `[1, N]` (note the floor is the rubric min, not 0) |
| bounded on a stated interval | that interval |

**Direction:** trust the paper's arrow, not the PwC `direction` hint (they
disagreed in real cases — e.g. RealIR's `LPS` is reported lower-better despite a
"similarity" name).

**If the paper is not enough → `null`.** Do **not** guess a bound for a shared
registry others depend on. Register **name-only with null bounds** (keeps the
name/direction/alias resolvable; the adapter emits `Infinity` bounds for it), or
leave it unresolved. Either is honest; a wrong bound is not. Record the
uncertainty in `metadata.confidence` (`low`) and `review_status: draft`.

### 4.3 The observed-range cross-check (do this for EVERY bespoke bound)

Before trusting a paper-claimed finite bound, confirm the dump's observed value
range fits it — this is a deterministic self-audit that catches scale mistakes:

```
observed [omin, omax] fits [lo, hi]                     → OK, use it
does not fit, but [omin/100, omax/100] fits AND hi<=1    → OK (adapter rescales percent→proportion)
does not fit either way                                  → SCALE MISMATCH: re-read the paper for
                                                            a ×10ⁿ reporting convention (LPIPS ×10³),
                                                            or a signed/percent scale you missed.
                                                            Register on the scale the dump uses.
```

This is what caught the 4 mis-bounded `BD-Rate` variants (observed negatives vs a
claimed `[0, ∞)`) and confirmed the PIE-Bench ×10³ scales. `consolidate_bounds`
implements it; the rule above is all you need to redo it by hand.

### 4.4 Using a model for the read

A model can do the paper-read; it must not do the bound. Give it the metric name,
its paper URL, its observed range, and the PwC direction, and require a **direct
paper quote** as evidence for `{min, max, lower_is_better, scale_note, evidence,
source_url, confidence}`. Then apply the §4.3 cross-check yourself and demote
anything that fails it. That cross-check is what makes a model's answer safe to
commit — never skip it. For a large first pass, cluster metrics by paper and
batch the reads; the results are frozen as citations, so it is a one-time cost.

---

## 5. Refresh + re-run (after ANY registry change)

```bash
uv run python -m every_eval_ever.adapters.paperswithcode.refresh_registry_snapshot \
    --seed <path-to>/eval-card-registry/seed/metrics.yaml
# writes registry_snapshot.json + records the registry git revision in _meta
```
Commit the registry change first so the snapshot pins a clean (non-`-dirty`)
revision. Then re-run the health check (§2); the metrics you registered should
move out of the unresolved report. A strict run (no `--best-effort`) should now
exit 0 for them.

---

## 6. Registry scale conventions (cheat-sheet)

Confirmed from the existing registry + adapter behavior:

- proportions / accuracy / recall / AP / AUROC / F1 / IoU → **[0, 1]** (percent
  boards auto-rescaled by group median; do not add a [0,100] twin)
- intrinsic percentages never reported as fractions → **[0, 100]**
- unbounded errors/distances (PSNR, MPJPE, MSE, FID, MCD, bitrate) →
  **max = null**, `lower_is_better` per whether up or down is good
- signed / unbounded both ways (BD-Rate, NSS, score deltas) → **min = max = null**
- MOS-type → **[1, 5]**; GPT-judge rubric → **[1, 10]** (or 0–100 if scaled)
- correlations / cosines → **[-1, 1]**
- aesthetic predictors → **[0, 10]**; angular error → **[0, 180]**

### How the adapter maps reported values onto your canonical scale

The canonical **target** scale is whatever you register here (min/max). The
adapter never guesses it. What it *does* infer, once per `(metric, dataset)`
leaderboard, is how that board's numbers map onto your scale. It works in
**log10** space with mass-aware gates (`analyze_group`), because a raw
high/low ratio stops being informative on a big board — a 3 000-row leaderboard
*will* contain both small and large values. Three outcomes:

- **uniform** — one reporting scale for the whole board. If it already matches
  canonical, values pass through untouched. If the board's robust centre is out
  of range but a single `×100` / `÷100` brings (nearly) all of it in, the WHOLE
  group is rescaled (`rescale_basis=group_uniform`) — a technical rescale, e.g.
  an all-percent board for a `[0,1]` metric.
- **mixed** — two scales genuinely coexist: two log-separated clusters, **each**
  clearing the mass floor `max(3, 5% of N)`, split by an empty ~0.75-decade
  valley. Each cluster gets its own factor (`rescale_basis=group_mixed`).
- **anomaly** — a *substantial minority* (≥ the mass floor) is out of range with
  no single consistent rescale and no clean valley. That is **not** a per-row
  typo; it means the metric's **registered scale is probably wrong**. The whole
  group is flagged `group_scale_mismatch`, values kept **raw**. This is your
  cue to re-register the metric on its natural scale (§4.3) — the adapter
  refuses to silently squash a genuine 0–3% metric into 0–0.03.

A **lone** out-of-range value *below* the mass floor, uniquely fixable by
`×100`/`÷100` (impossible under the declared range, one factor lands it in), is
fixed **per-row** (`rescale_basis=per_row`) — the value is corrected, the raw is
always kept in `score_details.raw_value`, and it is flagged. Everything else
outside range with no unique factor stays `scale_anomaly` (kept raw, flagged).

**Fatal vs informational.** `scale_anomaly` (incl. `group_scale_mismatch`) is a
strict-mode imperfection. Successful `group_uniform`/`group_mixed`/`per_row`
reconciliations are **informational only** (reported, never fatal): they are the
data being scaled the right way.

**Publication is separate from that gate, and stricter.** A record's declared
`[min_score, max_score]` has to contain its score — the datastore's semantic
check rejects it otherwise — so the last thing the adapter does before grouping
results into logs is drop any cell whose emitted score is outside its canonical
bounds, in **either** mode, and list it in the failure report with the metric it
came from. Two cases reach that gate: a `scale_anomaly` (kept raw on purpose)
and a boundary overrun inside the classification tolerance. So an
unreconcilable number is never published, and a run that dropped one exits
non-zero even under `--best-effort`. If a whole board turns up there, that is
the same signal as `group_scale_mismatch`: re-register the metric.

So: pick the max that matches the scale the metric is *intrinsically* on. If you
get it wrong, you won't corrupt data — you'll get a `group_scale_mismatch` flag
pointing you back here.

---

## 7. What "good" looks like

- Every new entry cites its basis in `metadata.source` (+ `metadata.paper` for
  bespoke) and starts at `review_status: draft`.
- No entry asserts a bound the observed data contradicts (run §4.3).
- Unknowns are `null`, not guessed.
- The snapshot is refreshed and its `_meta.registry_revision` is clean.
- A strict `--all` run's remaining unresolved set is only genuinely-new or
  genuinely-unknowable metrics — everything else resolved to a cited bound.
