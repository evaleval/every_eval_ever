"""sayf-eval adapter for every_eval_ever.

Converts sayf-eval's canonical *results record* (a pipeline-config-embedded
scores artifact) into the Every Eval Ever schema. Aggregate-only by design:
sayf-eval is a cybersecurity benchmark whose per-sample item text is dual-use and
kept private, so this converter never emits instance-level ``_samples.jsonl``.
"""
