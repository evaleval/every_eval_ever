"""Scheduled ingestion: run one adapter, then submit what it produced.

The pipeline is deliberately linear and each stage refuses to guess:

``catalog`` (which adapters, and how to invoke them)
    -> ``runner``     stage into a private tree, validate, fingerprint
    -> ``provenance`` mark the surviving records as cron-produced
    -> ``store``      snapshot raw payloads and the per-adapter ledger
    -> ``submit``     upload to that adapter's own datastore pull request

Run it with ``uv run python -m every_eval_ever.cron``.
"""
