<!-- Thanks for contributing! Delete sections that don't apply. -->

## What / source
<!-- What does this change do? For an adapter: what source does it convert, and to what grain? -->

## Review lane
<!-- See CONTRIBUTING.md § "How your PR gets reviewed". Pick one; a maintainer may disagree. -->
- [ ] **Fast** — no conflicts, scoped (tests / one adapter / one file), verified by me, under ~1000 hand-written lines
- [ ] **Needs a human** — design change, cross-package, large, refactor, material change in outcome, or large agent-authored change

<!-- If this is a large or structural change: link the issue or discussion where the design was
     agreed. Structural PRs opened without prior agreement will sit — we can't review them cold. -->
Design agreed in:

## Checklist
- [ ] `uv run python -m every_eval_ever validate <files>` is clean — **no warnings either** —
      run at the final `data/<collection>/<dev>/<model>/` path, where the semantic
      checks run
- [ ] every unconvertible source row is in `adapter_reports/`, and the command exits
      non-zero (not a silent skip)
- [ ] offline unit test added + full `uv run pytest tests` green
- [ ] `uv run ruff check` clean
- [ ] model/benchmark ids resolve in the registry (or an alias PR is prepared)
- [ ] content spot-checked (no answer leak, not double-counted, stable `evaluation_id`)

## Decisions & coverage
<!-- Skip this section if this isn't a data/conversion or skill change.
     Otherwise: this PR should be ready to merge, and this section makes the non-obvious
     calls visible so a maintainer can comment and the skill/schema can improve. Log every
     non-obvious CHOICE (not just where it was hard — a confident wrong choice has no
     "friction"). "None" is a valid answer. Mark anything that would recur on other
     datasets as General, so it can become a follow-up rather than being re-solved by
     the next contributor. -->

- Decision / where: 
  Chose / instead of: 
  Confidence (high/med/low): 
  General? (yes/no): 

**Coverage:** N source rows → N records, M dropped (reason) — <!-- no silent caps -->

**Operator asked about policy calls?** <!-- new canonical id / big data drop /
ambiguous or unbounded metric / re-hosting large data — which, and what was decided -->
