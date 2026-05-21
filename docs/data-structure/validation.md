---
layout: default
title: Validation
parent: Data Structure
nav_order: 2
---

# Validation

Validate aggregate `.json` files and instance-level `.jsonl` files:

```bash
uv run python -m every_eval_ever validate data/
```

Output formats:

```bash
uv run python -m every_eval_ever validate --format rich data/
uv run python -m every_eval_ever validate --format json data/
uv run python -m every_eval_ever validate --format github data/
```

Exit code is `0` when all files pass and `1` when any file fails.
