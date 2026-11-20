# jQMC Guarantee-Driven Development (GDD)

GDD records what jQMC guarantees and which tests support each guarantee.
The essential relation is two-way: every in-scope test belongs to at least
one guarantee, and every guarantee has test evidence. Both directions allow
multiple links; a shared test does not need a forwarding copy.

## Scope

GDD guarantees cover jQMC's own behavior: calculations are correct, wrong
input is rejected, post-processing is correct, and published interfaces and
examples are correct. Checks of the GDD machinery itself are admitted only
when they prevent a false green (a required case that did not run or failed,
or a selection that collected nothing, counted as a pass) or a change that
weakens the policy judging it (dropping a required case, making a verifier
conditional, editing a protected contract, unregistering evidence). Those
checks live in the `gdd_integrity` module
(`gdd/ssot/guarantees/gdd_integrity.yaml`), one test per property.
Performance, presentation, merge ergonomics, optional coverage and checks
that fail loudly anyway are not guarantees.

## Read the GDD manual

The Sphinx manual includes GDD pages generated from the authored YAML and
the test descriptions. Run `make -C doc html`, then open
`docs_build/gdd_audit/index.html`. Each guarantee keeps its statement,
tests, test summaries, fixtures, electron configurations and comparison
criteria together, with links to the underlying requirements and inputs.
Native Rust evidence is listed there as well.

`python scripts/render_gdd_ssot_ledgers.py` creates the same information as
readable Markdown views under `gdd/ledger/` and `gdd/derived/human-ledger/`.
These views are generated and not kept in Git.

A ledger describes requirements; it is not a claim that a particular
revision passed. Whether a revision passes is decided by running its tests.

## Layout

| Files | Role | Git |
| --- | --- | --- |
| `gdd/ssot/guarantees/<module>.yaml` and `gdd/ssot/guarantees/<module>/<ID>.yaml` | Authored guarantees and their test registrations | Tracked |
| `gdd/ssot/fixtures/`, `gdd/ssot/verdicts.yaml`, `gdd/ssot/evidence_roles.yaml`, `gdd/ssot/symbol_ownership.yaml`, `gdd/ssot/test_registry.yaml`, `gdd/ssot/inventories/` | Authored fixture contracts, verdict methods and tolerance policies, evidence roles, symbol ownership, test registry and reviewed inventories | Tracked |
| `gdd/ssot/schemas/` | Versioned schemas of the documents above | Tracked |
| `gdd/engine/` | Python engine that loads, validates and expands the authored contracts | Tracked |
| `gdd/derived/*.bindings.yaml` | Rebuildable descriptions extracted from the tests | Ignored cache |
| `gdd/ledger/`, `gdd/ssot/guarantees/README.md`, `gdd/derived/human-ledger/`, `gdd/derived/derivatives-requirements.md` | Generated readable views | Ignored |
| `doc/gdd_audit/` | Generated GDD pages included by the Sphinx manual build | Ignored |

Every readable GDD page is regenerated from the current source. Do not
hand-edit generated Markdown or bindings. Missing or stale bindings are
regenerated before their consumers run; they never require a commit.

Each guarantee lives in its own file, `gdd/ssot/guarantees/<module>/<ID>.yaml`
(the derivative ledger's `module_evidence` rows in
`gdd/ssot/guarantees/derivatives/module_evidence/<ID>.yaml`); the file name is
the guarantee id. The module file `gdd/ssot/guarantees/<module>.yaml` keeps the
module-level sections and a split marker. Guarantees are assembled in id order,
so adding one never edits a shared order list. Rust tests follow the same rule:
no collection-count snapshot is kept, because a total cannot identify missing or
substituted tests.

## Adding or changing a test

1. Write the deterministic test using central tolerances from
   `jqmc/_setting.py`.
2. Register its `pytest_node` under the appropriate guarantee's
   `required_evidence`. A test can support more than one guarantee.
   Native unit tests use `native_evidence` with their crate-qualified names.
3. When scientifically necessary, declare the required parameter cases,
   precision modes, execution profiles or selected fixture interactions.
   These are real requirements, not a full Cartesian product.
4. Run `python scripts/gdd_sync.py` to refresh the derived bindings and
   check the authored contracts.
5. Commit the authored changes; leave generated Markdown and bindings out
   of Git.

## Running the checks locally

The repository's `.pre-commit-config.yaml` runs the GDD checks as commit
hooks once `pre-commit install` has been run:

- `gdd-contracts` runs `python scripts/gdd_sync.py --check` whenever a file
  under `gdd/`, `tests/`, `jqmc/` or `scripts/` changes. It rebuilds missing
  or stale bindings and runs the static GDD checkers on the working tree:
  test mapping, required cases, fixture and derivative contracts, module
  contracts, module-ledger classification, Rust inventory, orphan tests,
  medium-tolerance sites, symbol ownership and bound comparisons.
- `gdd-deleted-path-hooks` runs the same checks when a commit only deletes a
  GDD input.
- `no-inline-tolerances-in-tests` rejects numerical tolerance literals in
  tests and production comparisons.

The same commands can be run directly:

```bash
python scripts/gdd_sync.py --check
python scripts/run_gdd_preflight.py --commit-sha "$(git rev-parse HEAD)" --certificate "${TMPDIR:-/tmp}/gdd-certificate.json"
python scripts/check_gdd_contract_changes.py --base-ref origin/main
```

`run_gdd_preflight.py` runs the static checkers, collects the test suite in
full and mixed precision (and the MPI collection of every file with a case
that requires an MPI world), checks mapping and required cases over that
collection, runs the GDD contract tests under `tests/test_gdd_simplification`,
and writes a checksum-protected certificate. `--gates-only` skips the
contract tests; `--pr-collection` runs only the collection checks.
Missing, stale or edited Markdown never affects these checks.

`check_gdd_contract_changes.py` compares the authored contracts with a
trusted base revision and runs the base revision's own copy of the checker.
Additive changes pass. Removing a guarantee or evidence node, narrowing
required cases or precision, changing a scientific claim or other protected
text, adding a skip, or loosening an effective tolerance is reported and the
check stays red; such a change needs its own review before it is merged.
Adding evidence, adding a requirement, tightening a tolerance, or renaming
an ordinary helper needs no count, digest, order or symbol-list update.

## Enforced properties

- **Mapping:** authored registrations against discovered tests, with the
  actual pytest and Cargo collections. Pure GDD framework self-tests are
  checked separately; the `jqmc_workflow` package is outside the scope.
- **Required conditions:** declared parameter, precision, execution and
  fixture interactions must be covered. Separate cases cannot be combined
  to pretend that one required interaction was exercised.
- **Classification:** every guarantee's ID prefix matches its class, and
  every verdict method that registers a test is legal for that class
  (`gdd/ssot/verdicts.yaml`). Whether a test actually verifies its claim is
  judged by the test itself and by review, not inferred from its syntax.
- **Ledger sections:** two checks run over every governed guarantee ledger
  and are not repeated in any ledger's `enforcement_map`:

  | section | checker |
  | --- | --- |
  | `guarantees` | `scripts/check_gdd_module_ledgers.py` |
  | `symbol_contract` | `scripts/check_gdd_symbol_ownership.py` |

  The table is `global_enforcement` in
  `gdd/ssot/schemas/guarantee-ledger-governed.v1.yaml`. A ledger's
  `enforcement_map` names only a check specific to that ledger.
- **Central tolerances:** numerical tolerance literals are forbidden in
  tests and production comparisons; values come from `jqmc/_setting.py`.
- **Execution:** every collected required case must pass setup, call and
  teardown in its required precision, profile and MPI ranks. Missing cases
  and unexpected skips are not successes.
- **Derivative coverage:** the analytical, autodiff and finite-difference
  chain keeps its exact required cases and per-case scientific comparisons.
- **Authored contract:** GDD validation and contract identities use the
  authored YAML, source and test definitions and verified execution
  evidence. Markdown is display output only.

Runtime completion records native pytest outcomes; it does not trace every
Python frame or reinterpret each numerical assertion.

## Scientific test rules

Use deterministic assertions, not statistical sigma windows. Evaluate full
and mixed precision within their respective definitions. A NaN match is not
a valid numerical result. Preserve meaningful fixture, route, finite-value
and physical-invariant assertions in the tests themselves.
