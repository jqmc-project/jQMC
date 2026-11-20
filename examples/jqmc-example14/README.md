# jqmc-example14: Frozen orbitals in a tiled supercell

The converter uses common primitive blocks for B-spline, PW, pGTO and PW+pGTO, with
single-sector or twist-average selection. J3 uses an independent ordinary
supercell basis. The [Technical Note](../../doc/notes/orbital_tiling.md)
explains the design, supported inputs and limitations. Total process memory
does not necessarily decrease for every backend or system size.

The first recipe uses a Gamma-point supercell and keeps the B-spline tables
on the primitive cell. Later sections cover the other representations, a
general supertwist and twist averaging.  For a tiling matrix `S`, jQMC reads exactly `|det(S)|` compatible,
unreduced primitive-cell k points and evaluates their Bloch orbitals directly
in the supercell.  It does not build the same-density supercell spline grid.

The representation is intended for a frozen nodal surface.  Jastrow parameters
may still be optimized, but keep `opt_geminal`, `opt_lambda_basis_exp` and
`opt_lambda_basis_coeff` false. These are Slater settings; the independent
J3 basis has its own `opt_J3_basis_exp` and `opt_J3_basis_coeff` flags.

## 1. Generate the compatible QE k mesh

For a `2 x 1 x 1` supercell, use the primitive structure in `pw.x` and include:

```text
&SYSTEM
  ...
  nosym = .true.
  noinv = .true.
/
K_POINTS automatic
2 1 1 0 0 0
```

The saved k points must be a complete unreduced set satisfying
`S @ k = integer`; jQMC checks this rather than assuming the order or mesh is
correct. Symmetry-reduced/incomplete cosets are rejected. Compatible shifted
meshes are accepted and determine a general supercell twist.

## 2. Convert without constructing supercell spline tables

```bash
jqmc-tool qe out/primitive.save \
  --kind bspline \
  --tiling "2 1 1" \
  --mesh-factor 0.75 \
  --jastrow-dtn u_chi_f \
  --output hamiltonian_data.h5
```

One quoted `--tiling` argument contains three diagonal factors, or nine
row-major entries (for example `--tiling "2 1 1 1 1 1 3 2 1"`). The resulting structure, Coulomb
terms and electron counts belong to the supercell, while each orbital row
retains a primitive-grid periodic part and its own primitive-cell Bloch phase.
The common block model retains each local AO-to-MO map; the B-spline identity
case needs no coefficient array or dense global identity matrix.

To use AO J3 instead of DTN, replace `--jastrow-dtn u_chi_f` with
`-j3 ao-small` (or ao-medium/large/full). UPF exponent seeds define an
independent ordinary periodic GTO basis on every supercell atom. J3 shares
the physical supercell but does not use Slater k blocks. Its initially zero
coupling matrix can be optimized with `opt_J3_param = true`; it includes
cross-cell couplings and retains the usual quadratic matrix storage.
`-j3 mo` is unavailable for tiled conversion.

For `N = |det(S)|` primitive cells, the tiled spline coefficient storage grows
as `O(N)`: `N` k blocks on one primitive grid.  An explicit same-density
supercell representation has `N` times as many bands on a grid with `N` times
as many points, hence `O(N^2)` spline storage.

## 3. Keep the nodes frozen

In a VMC input, optimize only the desired Jastrow channels:

```toml
[vmc]
opt_geminal = false
opt_lambda_basis_exp = false
opt_lambda_basis_coeff = false
opt_JDTN_u_param = true
opt_JDTN_chi_param = true
opt_JDTN_f_param = true
```

For an AO-J3 run, use `opt_J3_param = true` to optimize its coupling matrix.
Choose the J1/J2/J3 or DTN family at conversion; do not combine the two families.
The snippet above supplies optimization settings for a complete VMC input.

## 4. Select a general supertwist or an ensemble

The same tiling option works with `--kind pw`, `--kind pgto` or
`--kind pgto+pw`. The composite keeps both primitive basis components inside
each k block and preserves their joint projection coefficients. A single
complete source coset automatically determines the fractional supercell twist.
To select one sector from a larger complete mesh, add
`--supertwist "0.2 -0.1 0"` (the requested coset must exist in the input).
To keep all complete sectors, add `--twist-average` instead. Each sector uses
the same physical supercell and independent Jastrow basis, and observables
are combined with normalized sector weights. Spins remain `(+k_s,-k_s)`.

For PySCF-produced multi-k TREXIO GTO input, the corresponding command is:

```bash
jqmc-tool trexio convert-to primitive.h5 \
  --tiling "2 1 1" \
  --twist-average \
  -j3 ao-small \
  --output hamiltonian_twists.h5
```

The TREXIO source must store the primitive fractional vectors in
`pbc_k_point`, their weights in `pbc_k_point_weight`, and one `mo_k_point`
index for every MO row. Multi-k input with a missing or inconsistent MO-to-k
map is rejected before orbital conversion.

The ordinary GTO J3 basis covers every physical-supercell atom and has no
Bloch phase. Its M matrix is independent of the frozen Slater blocks.
Neither a Gamma supercell nor twist averaging removes the requirement for
complete primitive cosets; one primitive Gamma orbital set cannot fill a
nontrivial supercell determinant.

## 5. Source and supercell sector combinations

Both converters use the same matrix and fractional-supertwist options:

| Source | Gamma supercell | One general supertwist | Weighted ensemble |
|---|---|---|---|
| QE PW / B-spline / pGTO / PW+pGTO | `--supertwist "0 0 0"` | `--supertwist "0.22 0.13 0.09"` | `--twist-average` |
| PySCF/TREXIO Cartesian or spherical GTO | same | same | same |

A source containing exactly one complete sector can omit `--supertwist`.
An input with several sectors must select one or request `--twist-average`.
The converter rejects incomplete/duplicate cosets, nonfinite k points or
weights, unequal weights inside one coset and incompatible spin populations.
`QE_Workflow` and `WF_Workflow` accept the corresponding `tiling=[2,1,1]`,
`supertwist=[...]` and `twist_average=True` arguments.

The small Li and O sources of the test suite contain Gamma and
(0.22,0.13,0.09) supercell sectors with weights 0.2/0.8
for `--tiling "2 1 1"`. Li is restricted and O has independent spin orbitals.
These are small deterministic verification inputs, not converged production
energy or force examples. Force/stress comparisons use identical electron
positions, finite Jastrow parameters and the same ECP quadrature in tiled and
ordinary explicit-supercell representations.

## 6. Reproduce conversion with the small Li sources

This example directory ships no source files of its own. The Li sources are
the test-suite inputs of the source tree, so the commands below need a source
checkout or source archive (not only an installed package), but no new DFT
calculation:

- QE: `tests/qe_example_files/Li_tiling_sectors/out/li_tiling.save`, written
  by `li_scf.in` and `li_nscf.in` in `tests/qe_example_files/Li_tiling_sectors/`
  (QE logs `qe-scf.log`, `qe-nscf.log` in the same directory).
- PySCF/TREXIO: `tests/trexio_example_files/lithium_tiling_sectors.h5`,
  written by `tests/trexio_example_files/run_pyscf_Li_tiling_sectors.py`.

The ccECP pseudopotential `Li.ccECP.upf` is not shipped with the QE source,
and the converter reads it from the save directory. Run the commands from
`examples/jqmc-example14`: they copy the save directory into `work/`,
download <https://pseudopotentiallibrary.org/recipes/Li/ccECP/Li.ccECP.upf>
into the copy and write the converted files to `work/`:

```bash
mkdir -p work
cp -r ../../tests/qe_example_files/Li_tiling_sectors/out/li_tiling.save work/li_tiling.save
curl -L -o work/li_tiling.save/Li.ccECP.upf https://pseudopotentiallibrary.org/recipes/Li/ccECP/Li.ccECP.upf

jqmc-tool qe work/li_tiling.save \
  --kind bspline --tiling "2 1 1" --twist-average \
  -j1 0.5 -j2 0.7 -j3 ao-small --output work/li_qe_tiled_twists.h5

jqmc-tool qe work/li_tiling.save \
  --kind pgto+pw --tiling "2 1 1" --twist-average \
  -j1 0.5 -j2 0.7 -j3 ao-small --output work/li_qe_composite_tiled_twists.h5

jqmc-tool trexio convert-to ../../tests/trexio_example_files/lithium_tiling_sectors.h5 \
  --tiling "2 1 1" --twist-average \
  -j1 0.5 -j2 0.7 -j3 ao-small --output work/li_trexio_tiled_twists.h5
```

Replace `--twist-average` by `--supertwist "0 0 0"` or
`--supertwist "0.22 0.13 0.09"` to select one of the two complete sectors.
The output contains four physical Li atoms and four valence electrons.
The converter initializes J3 M to zero; the tests that use these sources
set nonzero cross-cell M entries before comparing energies and derivatives.
These inputs demonstrate conversion and consistency, not converged bulk Li
predictions. Keep Slater optimization disabled when running their QMC inputs.


## 7. Use the converted Hamiltonian in QMC

Tiling is selected at conversion. In the ordinary VMC, MCMC or LRDMC TOML,
point `control.hamiltonian_h5` to the converted file; there is no driver-side
`tiling` or `supertwist` key. For example, add the following control settings
to a complete input when using the QE ensemble above:

```toml
[control]
hamiltonian_h5 = "work/li_qe_tiled_twists.h5"
restart_chk = "restart_li_tiled_twists"
```

The HDF5 content selects a single-sector or twist-averaged driver. An ensemble
uses a checkpoint directory, including when it contains only one sector;
choose a new path that is not an existing single-sector checkpoint file.
VMC dumps preserve the complete ensemble and its primitive blocks. Use the
usual [optimized-Hamiltonian handoff](../../doc/examples.md#running-an-example) before
starting a fresh MCMC or LRDMC run. Rust is the CPU production evaluator;
XLA is also supported. Both use the same input and physical supercell.
