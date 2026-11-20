# GDD Fixture Contracts

This directory owns reusable fixture universes and their construction and witness
requirements. A fixture contract is independent of any one feature guarantee:
features subscribe to a canonical fixture suite and add only feature-specific
interaction coverage and evidence requirements.

`wavefunctions.yaml` defines the physical cases a shared wavefunction fixture suite
must cover. Its 16 canonical cases cover the independent determinant, Jastrow,
electron-count, periodicity, Bloch-k, cell-metric, and orbital-signature values without
claiming their full Cartesian product. Values derivable from those facts, such as the
overall scalar field and Jastrow composition, are deliberately not separate axes.

`sources.yaml` is the stable registry of committed TREXIO and QE artifacts. Keeping
artifact location out of `wavefunctions.yaml` prevents a physical case from being
confused with one particular way of constructing it.

`input_bindings.yaml` defines how each concrete guarantee case selects its realized
input. The two allowed origins are an explicitly specified `parameter-set` and a
registered `committed-artifact`; neither may be selected only inside a pytest fixture.
The semantic case and the concrete input origin are independent declarations and must
agree in the runtime receipt.

`electron_configurations.yaml` requires a separate declaration for the electron
positions used to probe that input. A case says either that positions are not required,
or identifies explicit coordinates or a seeded random sequence and its count. Any
constraint on the realized `G = compute_geminal_all_elements(...)` must state its
`cond_2(G)` bounds and comparators and emit one observed value per configuration.
AO/MO overlap and optimizer condition numbers are outside this contract.

A fixture label, filename, or pytest parameter ID is never sufficient evidence.
The realized object must emit a semantic signature and activation or absence
witnesses that agree with the required pattern.
