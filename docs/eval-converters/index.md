---
layout: default
title: Eval Converters
nav_order: 4
---

# Eval Converters

Supported conversion targets:

- Inspect AI
- HELM
- lm-evaluation-harness

These are the three main general-purpose converters expected to be supported in the core package.

Example commands:

```bash
uv run python -m every_eval_ever convert inspect --log_path <path>
uv run python -m every_eval_ever convert helm --log_path <path>
uv run python -m every_eval_ever convert lm_eval --log_path <path>
```

Adapter source code lives under [every_eval_ever/converters](../../every_eval_ever/converters/).

One-off adapters also exist under [utils](../../utils/) for source-specific parsing and business logic.
