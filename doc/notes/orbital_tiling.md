# Representation-independent orbital tiling

Orbital tiling supports B-spline, plane-wave, Cartesian or spherical periodic GTO, and PW+pGTO orbitals in a physical supercell. Converters for QE and PySCF/TREXIO accept Gamma, general-supertwist, and weighted twist-ensemble inputs. Determinant orbitals and geminals are frozen; Jastrow parameters stay optimizable. The physical supercell keeps its ions, electrons, walkers, interactions, and observables; tiling changes only how the orbital representation is stored, in primitive-cell blocks, and does not fold the cell or average separate primitive calculations.


## Preparing inputs

Supply the supercell matrix as three diagonal integers or nine row-major integers in one quoted argument:

```bash
jqmc-tool qe silicon.save --tiling "3 3 3"
jqmc-tool trexio convert-to multi_k_trexio.h5 --tiling "2 1 1 1 1 1 3 2 1"
```

Workflow APIs take integer lists (`tiling=[3, 3, 3]`). Driver TOMLs select the converted HDF5 through `control.hamiltonian_h5` and do not take a `tiling` key.

| Option | Effect |
|---|---|
| `--tiling` | Supercell factors (3 integers) or row-major matrix (9 integers). |
| `--supertwist "0.2 -0.1 0"` | Select one complete twist sector from a larger mesh. |
| `--twist-average` | Convert every complete twist sector. |
| `-j3 ao-small` / `ao-medium` / `ao-large` / `ao-full` / `ao` | Uncontracted J3 GTO basis on every physical-supercell atom. |

`--supertwist` and `--twist-average` are exclusive and apply to both `jqmc-tool qe` and `jqmc-tool trexio convert-to`; they do not change the (+k_s, -k_s) spin rule. A Gamma supercell still needs a complete primitive k coset, so primitive Gamma alone cannot supply nontrivial tiling. Missing, duplicate, or symmetry-reduced representatives are rejected rather than interpolated. Each twist sector is a separate wave function, and observables combine with normalized sector weights componentwise.

### Independent supercell J3

`-j3 mo` is rejected for tiled conversion; the Python Hamiltonian API accepts ordinary supercell MOs supplied by the caller. With `opt_geminal=false`, use `opt_J3_param=true` for J3 matrix optimization, and `opt_J3_basis_exp` / `opt_J3_basis_coeff` separately for its basis exponents / coefficients.

## What is stored

The converted Hamiltonian file preserves the selected representation on save/load and restart. Orbital tiling reduces orbital storage; the physical supercell still determines the size of J3, Slater matrices, walkers and estimator state. A smaller orbital representation therefore does not by itself guarantee smaller total memory.

## Restrictions

- Fractional occupations are unsupported. Source adapters validate integer occupations and require `N_up >= N_down >= 0`.
- pGTO+B-spline tiled blocks, dense AGP/AGPn conversion, and native independently displaced geometry continuation are outside the current interface.
- Orbital and geminal optimization flags are rejected before metric allocation; Jastrow optimization and physical-coordinate force and stress responses remain enabled.
- B-spline and fixed PW data carry no nuclear-position response, but freezing Slater parameters does not set physical-coordinate responses to zero.

## Convergence and checks

- Store one serialized supercell normalization factor and apply it exactly once; stored tables and coefficients remain primitive-normalized.
- pGTO lattice sums are truncated, so choose a cutoff that converges every quantity you need. A cutoff adequate for values need not bound strain or higher derivatives, and agreement between analytical and automatic derivatives of one truncated sum does not prove convergence to the infinite-sum derivative. Converge the pGTO image tolerance for the values and derivatives required by the calculation. The tolerance is preserved on save and restart; primitive and explicit-supercell calculations need independent convergence checks.
- An explicit-supercell representation of folded PW orbitals is an equality of represented wave functions, not of optimized supercell SCF solutions; reprojecting onto a supercell-periodic spline needs its own grid-convergence study, especially for Laplacians.
- LRDMC-n and LRDMC-t consume the same provider without changing its normalization or spin convention; checkpoints must preserve primitive blocks, twist sector, and logical spin sharing.

Related reading: the [occupied-orbital SD note](sd_occupied_orbitals.md), the [derivative note](derivatives.md), and the CLI reference and tiled-supercell tutorial for supported commands and runtime restrictions. Derivations of periodization, supercell correspondence, and periodic-derivative constructions are WIP and not required to run a calculation.
