# Provenance

Produced by `scripts/upstream_smoke/lighteval_smoke.py --refresh`,
not hand-copied from a real evaluation.

- lighteval version: `0.13.0`
- tasks: `anli:r1|0,squad_v2|0`
- samples per task: `2`
- model: `eee-smoke/dummy-model`

The model is lighteval's own `DummyModelConfig` (seed 42): random
logprobs and fixed text, no weights and no inference. The scores here
are therefore meaningless as measurements — this tree exists to pin
the *shape* of lighteval output that the converter reads.

To update after an upstream change, re-run the command above and read
the diff before committing it.
