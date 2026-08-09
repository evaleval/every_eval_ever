# EEE deployment metadata proposal

- PR: `<discussion URL>`
- PR head: `<40-character commit SHA>`
- Schema version: `<version>`
- Scope: `<model count> models / <file count> files`
- Status: **PENDING HUMAN APPROVAL**

| model_id | proposed deployment_type | proposed model_availability | confidence | source(s) |
|---|---|---|---|---|
| `<exact model_info.id>` | `<self_deployed / externally_managed / unknown>` | `<open_weights / closed_weights / unknown>` | `<D: high/medium/low; A: high/medium/low>` | `<D:S1; A:S2,S3>` |

## Sources

- **S1** — [`<source title>`](<URL>), `<revision or access date>` — `<what it establishes>`

## Unknown rationale

Complete one row for every `unknown` cell. Do not finalize this proposal while any
axis is `research_incomplete`.

Coverage codes: `R` record/run artifacts, `A` adapter/config, `E` eval card/README/
methodology/leaderboard, `M` exact model card/release, `P` official provider/API,
`W` paper/repository/archive.

| model_id | axis | status/reason | checked | confusion or evidence gap | evidence needed to resolve |
|---|---|---|---|---|---|
| `<exact model_info.id>` | `<deployment_type / model_availability>` | `<conflicting_sources / unavailable_after_search>: <reason code>` | `<R:S1; A:S2; E:none; M:S3; P:n/a; W:S4>` | `<concise explanation>` | `<specific artifact or statement>` |

## Other unresolved questions

- None.

## Approval

Approve only the completed proposal's reported SHA-256 at the PR head above. Any
change to the head, table, or evidence invalidates approval.
