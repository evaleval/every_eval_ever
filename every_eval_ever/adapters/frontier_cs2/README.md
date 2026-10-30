# FrontierCS 2 Adapter

Convert formal evaluation run outcomes from the [FrontierCS 2 Benchmark](https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-runs) into Every Eval Ever (EEE) format.

## Overview
- **Source**: Hugging Face dataset [`FrontierCS2/FrontierCS-2-runs`](https://huggingface.co/datasets/FrontierCS2/FrontierCS-2-runs)
- **Paper**: [FrontierCS 2: A Benchmark for Computer Science R&D](https://huggingface.co/papers/2610.29932)
- **Artifacts**:
  - `metadata/formal-results.json`: Canonical outcome records for 48 tasks and 19 model candidates (912 runs).
  - `metadata/trace-manifest.tsv`: Content-addressed trace archive references and digests.
- **Grain**: One aggregate `EvaluationLog` per model (19 logs), each containing 48 `EvaluationResult` entries corresponding to each evaluated task.

## Running the Adapter

Direct execution:
```bash
python -m every_eval_ever.adapters.frontier_cs2.adapter --output-dir data/frontier-cs-2
```

Source freshness token probe:
```bash
python -m every_eval_ever.adapters.frontier_cs2.adapter --emit-source-version
```

Scheduled cron execution (dry-run):
```bash
python -m every_eval_ever.cron run --adapter frontier_cs2 --dry-run
```
