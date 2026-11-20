# GDD Single Source of Truth

This directory holds the authored jQMC Guarantee-Driven Development contracts:

- stable guarantee definitions and their test registrations;
- supported-universe axes and concrete obligation generators;
- applicability, exclusion, and equivalence rules;
- required proof, fixture-witness, liveness, precision and backend policy;
- the contract-weakening policy;
- the mapping rules used to construct the completion certificate.

Authored guarantees are tracked; derived bindings are a rebuildable, ignored
cache. A test may be registered under several guarantees. The commands that
check these contracts are described in the [GDD README](../README.md).

## How to read these contracts

Every fixture/policy document contains only these kinds of content, and the
top-level `enforcement_map` in each document says which is which, per section.
For a governed guarantee ledger, a section that every ledger shares is named
once in the schema's `global_enforcement` table instead; the ledger's own
`enforcement_map` names only a check specific to it:

1. **Engine-enforced** (`gdd.engine.<module>.<symbol>`): the named executable
   type or function is the single authority for that section's shapes and
   cross-field rules. The YAML deliberately does not duplicate its field lists;
   read the engine dataclass to see the exact shape. Static checks verify the
   YAML vocabularies against the engine source, so the two cannot drift.
2. **Statically checked** (`scripts/check_gdd_*.py`): the named checker
   mechanically checks that section. Editing the section and running the
   script shows exactly what the contract enforces.
3. **`not-yet-enforced`**: reviewed intent with no mechanical check yet -
   visible debt.
4. **`per-rule`**: the section is a list of rule items and every item names its
   own enforcer in an `enforced_by` field. Declaring a rule without its
   enforcing implementation is a schema error; `not-yet-enforced` rules are
   per-row visible debt.

To check a section: find it in `enforcement_map` (or the schema's
`global_enforcement`), open the named enforcer, and
check that the YAML statement and the enforcement agree. A section whose
enforcer you cannot find is a contract bug.

## Authority boundaries

Each concrete guarantee case is judged along independent axes, each owned
by exactly one document. No other document may restate an owned concern.

| concern | owning document | binding/receipt authority |
|---|---|---|
| Which physical wavefunction/Hamiltonian case is required | `fixtures/wavefunctions.yaml` | static check |
| Where the realized input comes from (parameter set or committed artifact) | `fixtures/input_bindings.yaml` | `gdd.engine.evidence.InputBinding` |
| Which committed artifacts exist and their stable IDs | `fixtures/sources.yaml` | static check |
| Which electron positions are tested and any cond_2(G) screen | `fixtures/electron_configurations.yaml` | `gdd.engine.evidence.ElectronConfigurationBinding` |
| How the pass criterion is evaluated and recorded per precision mode | `verdicts.yaml` | `gdd.engine.evidence.ComparisonBinding` / `ObservationBinding` |
| What must be true, over which obligations | `guarantees/<module>.yaml` + `guarantees/<module>/<ID>.yaml` (one file per guarantee) | static check + runtime completion |
| Which production symbol belongs to which ledger | `symbol_ownership.yaml` | static check |

Fixture evidence is relational and lives at the evidence-node (or
parameter-case) grain:

`guarantee -> pytest node -> parameter-case selector -> input/source/wavefunction/electron configuration -> verdict`.

The generated ledgers render only that direct relation as fixture evidence. A
node without a direct binding fails the direct fixture check; no aggregate
guarantee profile substitutes for a concrete node binding.

## Authored and generated files

`guarantees/*.yaml` contains only reviewed meaning, allowed verdict methods,
and the pytest-node registration. The mechanically derivable case counts,
claims, selectors, fixture paths, and comparison bindings live in
`gdd/derived/*.bindings.yaml`. These generated files are ignored by Git and
rebuilt automatically; they must never be edited by hand.

Readable Markdown under `gdd/ledger/`, the generated
`gdd/ssot/guarantees/README.md` and the complete views under `gdd/derived/`
are also ignored generation outputs. The `make -C doc html` build converts
the complete linked GDD information into the Sphinx manual.

Every consumer loads the authored/derived pair through `gdd.engine.ledger`.
The loader rejects a missing file, source or generator hash drift,
extra/missing guarantees, and extra/missing/malformed nodes before exposing a
case binding. The preflight snapshot covers authored `gdd/ssot`, not local
cache metadata.

`python scripts/gdd_sync.py` derives counts and YAML bindings, then runs the
static checks. It never runs the scientific tests and never rewrites guarantee
universes, declared gaps, tolerances, exemptions, or ownership inventories.
`python scripts/check_gdd_contract_changes.py --base-ref <base>` compares the
authored contract with a trusted base revision: evidence additions pass, while
deletions, same-count replacements, scientific fixture/policy changes, scope
narrowing, and specialized-manifest changes are reported for separate review.
It resolves base and candidate registries separately, preserves existing input
kinds and concrete sources, and checks retained per-node artifact IDs, paths
and hashes. Existing node requirements, including precision and comparison
policies, remain protected.

## Guarantee classes and verdict methods

One root contract, `verdicts.yaml` (rendered as `gdd/ledger/verdicts.md`),
defines the core of GDD:

- the five-class correctness hierarchy (PHY / MATH / NUM / IMPL / OPS) - what
  kind of correctness a passing test proves and what it deliberately does not
  prove (parity is never physical proof); and
- the finite verdict-method catalog - by exactly which oracle/mechanism and
  pass criteria a test may claim correctness. Every guarantee assigns its
  verdict methods by ID; the static check rejects an unknown method, a method
  illegal for the guarantee's class, or a method whose pass criteria reference
  unknown policies.

## Glossary

- **guarantee**: one reviewed property with a stable ID (e.g.
  `MATH-DERIV-LOG-WAVEFUNCTION-001`) that must hold over a declared scope; its
  prefix names the guarantee class above.
- **obligation / case / cell**: one concrete required instance of a guarantee
  (one wavefunction x axis x quantity combination), identified by an obligation
  ID such as `ln_Psi|matrix|<cell>|axis=c`.
- **binding**: the declaration a test case makes BEFORE running - which input,
  which electron configuration, which comparison policy it will use. Shapes are
  the `*Binding` dataclasses in `gdd/engine/evidence.py`.
- **receipt**: the validated record a test case emits AFTER running - what was
  actually realized, observed, and compared. Shapes are the `*Receipt`
  dataclasses in `gdd/engine/evidence.py`; checksum-protected summaries feed the
  cross-worker completion certificate.
- **witness**: a receipt element proving non-vacuity (the route actually ran,
  the leaf is live, the precondition was observed), as opposed to a green
  pytest node name.
- **lane**: one evidence environment (`xla-full`, `mixed`, `rust`).
- **completion certificate**: the record issued only when every required
  obligation has a matching, untampered receipt in every required lane.

## Current state

Every document in this directory carries `lifecycle: cutover`, and all
guarantee ledgers are governed with zero placeholders. `guarantees/derivatives.yaml` is the deepest
contract: it binds the reviewed derivative obligation ledger with runtime
receipts, while the other governed ledgers complete through the static
preflight checks plus module receipts.

`fixtures/wavefunctions.yaml` defines the shared fixture universe. It requires
constrained one-way coverage of independent wavefunction and Hamiltonian case
values through the reviewed pinned canonical cases and forbids a full Cartesian
product. Derived classifications are not duplicated as axes. Committed artifact
locations live in `fixtures/sources.yaml`, while `fixtures/input_bindings.yaml`
selects the actual input for each guarantee obligation.

`fixtures/input_bindings.yaml` requires every concrete guarantee obligation to bind
exactly one input origin: an explicitly specified `parameter-set` or a
`committed-artifact`. The binding and its deterministic transformations must be
present in a runtime receipt; a pytest fixture name cannot serve as that receipt.

`fixtures/electron_configurations.yaml` independently requires every concrete
obligation to declare whether electron positions are needed. Required positions are
either reviewed explicit arrays or a deterministic seeded sequence with a declared
configuration count. The geminal-matrix policy is explicitly one of not applicable, not limited,
upper-bounded `cond_2(G)`, or a target `cond_2(G)` range. Constrained cases record
one observation per realized electron configuration; an optional `sigma_min(G)`
guard remains separate because a condition number is scale-invariant.

`verdicts.yaml` defines raw-bit, exact-value, and centrally governed
allclose policies. Every required precision mode must select an explicit policy or
reviewed non-comparison disposition. Full and mixed precision never inherit an
implicit common default, and the runtime receipt records the resolved tolerance and
observed error.

The derivative contract is enforced at three levels:

- `scripts/check_gdd_derivatives_contract.py`, one of the static checks of the
  GDD preflight and of `gdd_sync.py --check`;
- `test_gdd_derivatives_contract.py`, which binds the YAML counts and digests
  to the live XLA and Rust required universe, with no Rust-side N/A
  obligations; and
- the actual collection, which must close the exact reviewed XLA, FD, mixed,
  Rust-fragment and Rust-obligation ID universes
  (`scripts/check_gdd_required_cases.py --collection-dir`), while every
  registered case checks its own checksum-protected receipts at teardown.

Production-entry discovery independently scans the source boundary and requires
exact agreement with the SSoT. The generated derivative fixture matrix contains
explicit MO-representation AGP cells.

The wavefunction fixture contract has a static check of its source-discovered
production class vocabulary, physical rules, committed source registry, and
canonical-suite policy. The input-binding and comparison policies have a static
check of their schema, registered `_setting.py` symbols and `_precision.py`
resolvers, and policy references. The derivative guarantee consumes their runtime
receipts; other guarantees connect their own consumers rather than inheriting
derivative completion.

`symbol_ownership.yaml` is the production-symbol ownership contract. Python AST
discovery partitions every in-scope top-level class and function into public and
private symbols and requires an exact match with one primary guarantee ledger.
Methods, nested definitions, imports, aliases, assignments, and generated version
metadata remain outside this inventory. A governed ledger resolves every owned
symbol to an existing guarantee through a default or per-symbol `guarantee_refs`
binding, and every declared guarantee has blocking required evidence. This is
traceability, not a claim that every helper has a direct test. The reviewed
inventory of modules and symbols lives in `symbol_ownership.yaml` and is not
duplicated in this prose. Rust symbol discovery still needs a Rust-aware contract
(a declared gap on `jqmc_rs.yaml`), so `jqmc_rs.yaml` governance covers the
reviewed test surface, not AST-discovered Rust symbols.
`scripts/check_gdd_symbol_ownership.py` checks the inventory against the live
AST; the inventory is authored and reviewed, never regenerated by the checker, so
an intentional public API change edits `symbol_ownership.yaml` in the same change.

An unauthored ledger is never an empty YAML mapping. It declares
`document.lifecycle: placeholder`, `symbol_contract.status: placeholder`, and an
empty guarantee list. A module becomes governed by replacing placeholder lifecycle
and symbol status together, declaring guarantees with blocking evidence, and
closing its symbol bindings. A newly added production module must ship its governed
ledger and completion adapter in the same change, or the preflight fails.

GDD contains no CI machinery: it reads no workflow file and assigns no test case
to a CI job. The GDD preflight (`scripts/run_gdd_preflight.py`) runs the retained
`scripts/check_gdd_*.py` checkers, the actual full/mixed collection (and the MPI
collection of every file with a case that requires an MPI world), the mapping and
required-case checks over that collection, and the static GDD contract tests, and
emits a checksum- and commit-bound certificate. The preflight is static-only:
contract tests that execute jqmc numerics (JAX or `jqmc_rs`) carry the marker
`gdd_runs_science`, are deselected by `scripts/gdd_static_preflight.py`, and run
with the ordinary test suite. Every unmarked contract test runs in the preflight
under a guard that fails it on any JAX or `jqmc_rs` execution, and the executed
count must equal the selected count. The node-level registry
`gdd/engine/derivative_registry.py` derives each collected case's derivative
receipt keys from its definition and real callspec values; the required-case
check closes the registered keys of each lane against its sha256-pinned exact
universe (each key on exactly one collected required case). At teardown every
registered case checks that it emitted exactly its registered receipts. A
required test that did not run fails by construction: a missing case, a
missing shard or an unexpected skip is not a success. A successful run
certificate has scope `currently-governed-contracts` and seals
`repository_cutover_certified: true`; the certificate tests pin zero
`pre-cutover` ledgers and zero placeholders, so a single ledger regressing turns
the preflight red.

## Document schemas and authoring conventions

Every document in this directory has a versioned schema under `schemas/`, and
`scripts/check_gdd_ssot_schema.py` checks documents against them. A document's
`schema_version` names the schema file it must satisfy, so a structural change
requires a new schema version rather than silent drift. The static GDD preflight
runs the schema checker.

The authoring conventions: word-style mapping keys never mix snake_case and
kebab-case, never grow into sentences, and prose (any string value containing
whitespace) lives only under designated prose keys such as `purpose`,
`statement`, `notes`, and `rule`-suffixed keys. Machine-checked fields and human
documentation therefore remain mechanically separable. Lifecycle values use the
single vocabulary `placeholder`, `pre-cutover`, and `cutover`.

## Rules

1. Store current normative truth here; do not store historical rationale here.
2. Generate human-readable ledgers from the executable authority; do not maintain a
   second hand-edited ledger.
3. A scoped guarantee must expand to concrete required cells.
4. A pytest node or fixture label alone is not evidence. Required evidence must bind
   the cell to construction provenance, an observed case signature, an activation or
   route witness, and an execution result.
5. Contract reductions, exclusions, tolerance loosening, weaker proof kinds, and
   skip/fallback changes require an explicit weakening review.
6. Unknown or missing evidence remains visible debt. Do not convert it to `N/A` only
   to close the certificate.

Representation: declarative contracts and policies are YAML, while a constrained
Python engine resolves registered operation IDs, expands obligations, executes
fixture witnesses, and produces certificates. Generated JSON and Markdown are
derived views, not parallel authorities.
