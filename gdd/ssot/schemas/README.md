# GDD SSoT schemas

Every YAML document under `gdd/ssot/` (outside this directory) must match exactly
one schema file here. `scripts/check_gdd_ssot_schema.py` checks this as part of
the static GDD preflight.

## What `schema_version` means

The `schema_version` declared by an SSoT document is the version of the schema
file that the document must satisfy; the checker rejects a mismatch.

Every document and schema is at version 1. A breaking structural change adds
`<family>.v<N+1>.yaml` together with a matching document version bump.

## Schema file format (`schema_format: 1`)

- `schema_id`: unique stable identifier of this schema file.
- `match`: `path` (exact) or `path_glob` plus an optional `lifecycle`
  discriminator. An exact path match always wins over a glob.
- `document_schema_version`: the `schema_version` the matching document must
  declare.
- `required_top_level` / `optional_top_level`: exact top-level vocabulary; a
  top-level key outside both lists is an error.
- `document_block`: required fields of the `document:` header, or `null` for
  contracts without one.
- `placeholder_invariants`: when true, the document must be a placeholder ledger
  (placeholder lifecycle and symbol status together, empty guarantee list).
- `enforcement_map_required`: when true, every other top-level section must be
  covered, either by `global_enforcement` or by the document's top-level
  `enforcement_map`. Each `enforcement_map` value must be an existing checker
  script path, a `gdd.engine.*` symbol, or the literal `not-yet-enforced` (see
  the reading guide in `gdd/ssot/README.md`), and must not repeat the global
  enforcer of its section. The map is required unless the schema lists it as
  optional.
- `global_enforcement`: section -> checker for checks that run over every
  document of the schema. Each checker must be an existing
  `scripts/<checker>.py` listed in `RETAINED_CHECKERS` of
  `scripts/run_gdd_preflight.py`.
- `prose_keys_extra`: file-specific additions to the global prose-key list.

## Repository-wide authoring rules (enforced globally)

Authored evidence-node fields are closed by `gdd.engine.ledger.AUTHORED_NODE_FIELDS`.
The optional `condition_check_iterations` maps a source assertion identity to
exactly `{comparison: <executable-check-name>}`. That comparison supplies the
required iteration labels and existing precision-specific tolerance policy;
the condition may not introduce a second tolerance allowance. The loader and
verdict-semantics gate require a mandatory conditioning assertion and comparator
in the same invocation. Runtime capture and completion require every label's
actual, normally completed condition check and replay its numeric inputs using
the linked tolerance policy. Unknown fields, links, labels or input types fail.

- A lowercase word-style mapping key uses snake_case or kebab-case, never both
  inside one key, is at most 48 characters, and has at most 6 segments. Keys
  that read as sentences are rejected; state the requirement as a short token
  and put the explanation in a prose field.
- Scalar string values containing whitespace are prose and are only allowed
  under designated prose keys (`purpose`, `statement`, `plain_language`,
  `notes`, `rule`-suffixed keys, and the other keys listed in the checker).
  Everything else must remain whitespace-free tokens or references, so the
  machine-readable subset of a contract stays mechanically separable from its
  documentation.
- Every `lifecycle` value is one of `placeholder`, `pre-cutover`, `cutover`.
- Every declared rule carries its guaranteeing implementation: a list item
  containing a `rule` key must also carry `enforced_by` (checker script path,
  `gdd.engine.*` symbol, or `not-yet-enforced`); a section made of such items
  uses `per-rule` in the document's `enforcement_map`.
