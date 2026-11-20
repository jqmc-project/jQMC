(trexio_tags)=

# Interfaces with TREXIO

The public `read_trexio_file` loader supports molecular files, periodic
single-k files, and selecting one row from a periodic multi-k file. Complex
Bloch coefficients and non-$\Gamma$ twists are supported. When a multi-k file
is read without `twist_ik`, the loader warns and selects row zero. To convert
all rows of a multi-k file into one twist-averaged Hamiltonian, use
`jqmc-tool trexio convert-to --twist-average` (see "Multi-twist input" below).

## Command-line conversion

Most users convert a TREXIO file with the `jqmc-tool` command rather than the
Python API:

```bash
jqmc-tool trexio show-info molecule.h5                 # print the system, basis, MOs and ECPs
jqmc-tool trexio convert-to molecule.h5 -o hamiltonian_data.h5 -j2 1.0
```

`convert-to` reads the file with `read_trexio_file`, adds the Jastrow factors
requested on the command line (`-j1`, `-j2`, `-j3`, `--jastrow-dtn`, ...; see
`jqmc-tool trexio convert-to --help`) and writes `hamiltonian_data.h5` (or the
name given with `-o`). The Python workflow class `WF_Workflow` runs the same
command (see {doc}`workflows`).

Without `--twist-average`, a multi-k TREXIO file is converted from row zero
with the warning described above. To use another single k-point, call
`read_trexio_file(..., twist_ik=N)` from Python and assemble the Hamiltonian
there.

## Conventions employed in jQMC and TREXIO

jQMC utilizes the [TREXIO](https://github.com/TREX-CoE/trexio) library for handling input and output of quantum chemical data. This interface ensures standardization and portability of wavefunction parameters and system coordinate data. jQMC only reads TREXIO files, and only files written with the HDF5 back end; files written with the TEXT back end must be converted to HDF5 first; opening them fails with an error from the `trexio` library. The `trexio` Python package (version 2.6.0 or later) is installed as a dependency of jQMC.

### The `read_trexio_file` Function

The public entry point is imported from the package root:

```python
from jqmc import read_trexio_file
```

It parses a TREXIO HDF5 file and populates jQMC data structures.

**Signature:**

```python
def read_trexio_file(
    trexio_file: str,
    store_tuple: bool = False,
    twist_ik: int | None = None,
) -> tuple[
    Structure_data,
    AOs_sphe_gto_data | AOs_cart_gto_data
    | AOs_sphe_pgto_data | AOs_cart_pgto_data
    | AOs_sphe_pgto_complex_data | AOs_cart_pgto_complex_data,
    MOs_data,
    MOs_data,
    SD_data,
    Coulomb_potential_data,
]:
    ...
```

**Arguments:**

*   `trexio_file` (str): The file path to the TREXIO file (typically `.h5`).
*   `store_tuple` (bool, default=`False`): If `True`, stores list-like fields as tuples, which makes the returned objects immutable. `jqmc-tool trexio convert-to` uses `True`. Loading is slower with `True`; the choice does not change the physical content.
*   `twist_ik` (int or `None`, default=`None`): Select a zero-based k-point from a multi-k TREXIO file. `None` selects row zero; when more than one row is present, the loader emits a warning.

**Returns:**

Ordering of the returned tuple:

1.  `Structure_data`: Molecular geometry and atomic information.
2.  An `AOs_*_gto_data` molecular basis, an `AOs_*_pgto_data` periodic basis (real, $\Gamma$ point) or an `AOs_*_pgto_complex_data` complex Bloch periodic basis (non-$\Gamma$ twist, or any row of a multi-k file), in Cartesian or spherical form. How the down-spin channel is paired with it is described in section 5.
3.  `MOs_data` (Up): Molecular Orbitals for spin-up electrons.
4.  `MOs_data` (Down): Molecular Orbitals for spin-down electrons.
5.  `SD_data`: The occupation-selected Slater determinant assembled from the MO block.
6.  `Coulomb_potential_data`: Information on pseudopotentials (ECPs) if present.

---

### Implementation Details

#### 1. System Structure

*   **Boundary Conditions**: The wrapper checks `trexio.read_pbc_periodic`.
    *   **Open Boundary Conditions (Molecules)**: Fully supported.
    *   **Periodic Boundary Conditions (Crystals)**: Supported.
        *   **Single-k $\Gamma$**: real wave function; the determinant AO basis has `k_point = None`.
        *   **Non-$\Gamma$ twist**: complex wave function; the determinant AO basis has `k_point = (k_a, k_b, k_c)` in fractional reciprocal-lattice units (for example, `(0.5, 0, 0)` is the X point of an orthorhombic cell). A $\Gamma$ row inside a multi-k file remains the complex uniform representation with `(0, 0, 0)`. The slot read is the TREXIO standard `pbc/k_point`.
*   **Atomic Information**: Reads nuclear labels (`H`, `C`, etc.) and coordinates. Labels are converted to atomic numbers (Z) using an internal mapping supporting elements up to Radon (Z=86).

#### 2. Electron Configuration

*   Reads `electron_up_num` and `electron_dn_num`.
*   Determines if the system is **spin-polarized**:
    *   `spin_polarized = True` if $N_{\uparrow} \neq N_{\downarrow}$.
    *   `spin_polarized = False` if $N_{\uparrow} = N_{\downarrow}$.

#### 3. Atomic Orbitals (AOs)

The wrapper distinguishes between Cartesian and Spherical Harmonic basis sets via `trexio.read_ao_cartesian`.

**Cartesian Gaussians:**
*   Polynomial orders ($n_x, n_y, n_z$) for each orbital are reconstructed based on the angular momentum $L$.
*   In the case of polynomials, the canonical (or alphabetical) ordering is used (e.g., $xx, xy, xz, yy, yz, zz$ for $d$-orbitals).
*   Basis-function normalization and coefficients are stored in an
    `AOs_cart_gto_data` or `AOs_cart_pgto_data` object, depending on the boundary
    condition.

**Spherical Harmonics:**
*   If `ao_cartesian` is false, an `AOs_sphe_gto_data` or
    `AOs_sphe_pgto_data` object is initialized.
*   The Magnetic quantum numbers ($m$) follow the TREXIO canonical order: $0, 1, -1, 2, -2, \dots$ for a given $L$.

#### 4. Molecular Orbitals (MOs)

*   **Real vs. Complex**: The loader checks `trexio.has_mo_coefficient_im`.
    *   A molecular or single-k $\Gamma$ file without an imaginary-coefficient dataset uses fp64 MO storage.
    *   In a Bloch context, the imaginary dataset is combined with the real dataset as `mo_coefficient + 1j * mo_coefficient_im`, producing complex128 coefficients. A complex MO in a single-k $\Gamma$ file is rejected because its physical interpretation is ambiguous.
*   **Spin Handling**:
    *   **Restricted (RHF/ROHF)**: Used when `spin_dependent` is false. The down channel is derived from the same spatial-orbital block.
    *   **Unrestricted (UHF)**: Used whenever `spin_dependent` is true, including broken-symmetry cases with equal electron counts. Alpha and beta coefficients are read separately.
*   **Occupation selection**: `SD_data` stores exactly
    $N_\uparrow$/$N_\downarrow$ occupied rows selected from the TREXIO
    occupations. The single- and multi-twist loaders apply this same rule;
    virtual TREXIO bands never enter canonical SD storage.

#### 5. Time-reversal pairing of the spin channels

For a restricted complex Bloch determinant the up-spin orbitals satisfy the
Bloch condition at $+\mathbf{k}$ and the down-spin orbitals at $-\mathbf{k}$
(see {doc}`wavefunction`, "Periodic determinants", and
{doc}`sd_occupied_orbitals`), so the two spin channels use separate `+k` and
`-k` AO bases.

```{math}
C_{\mathrm{dn}}^{-k} = \bigl(C_{\mathrm{up}}^{+k}\bigr)^{*}
\quad\text{(restricted case)}.
```

For a restricted TREXIO MO block, the loader constructs the down channel by
conjugating the up coefficients and attaching them to the `-k` basis. It sets
`SD_data.time_reversal = True`. For an unrestricted block, stored alpha and
beta coefficients remain variationally independent, and the down channel
uses the same opposite-twist rule with its own coefficients: the conjugated
stored beta coefficients on the `-k` basis, $C_{\mathrm{dn}}^{-k} =
(C_{\beta}^{+k})^{*}$, never the conjugated alpha coefficients, and
`time_reversal = False`.

The loader does not compare independently stored `+k` and `-k` rows against a
numerical time-reversal tolerance. Such a source-data validation must not be
inferred from the construction above.

#### 6. MO phase gauge normalization at load

The complex MO coefficients carry an arbitrary U(1) phase per (band, k) pair from the upstream DFT (no physical content; it is a gauge choice). To avoid this gauge becoming a slow mode in SR / LM, the loader applies a **deterministic gauge fix** at TREXIO load time:

```{math}
\phi_{\mathrm{band}}\ \mapsto\ e^{-i \arg c_{\mathrm{first}}}\,\phi_{\mathrm{band}},
```

where `c_first` is the first coefficient of the band whose magnitude exceeds
`1e-8`. This rotates each complex band so the chosen anchor coefficient is
real and positive. The operation costs `O(M_ao N_band)` for each selected
twist.

#### 7. Multi-twist input

`jqmc-tool trexio convert-to --twist-average` converts every k-point of a
multi-k file (for example from PySCF `KRHF`) into one multi-twist HDF5 file,
using the k-points and weights stored in the file:

```bash
jqmc-tool trexio convert-to kpts.h5 -o hamiltonian_twists.h5 -j2 1.0 --twist-average
```

The file needs at least two k-points. An `ao*` J3 basis (`-j3 ao`, `ao-small`,
`ao-medium`, `ao-large` or `ao-full`) is built once from the file's AO basis,
which is common to all k-points, in its spherical or Cartesian form. The Bloch
phase is removed, so the J3 basis is an ordinary periodic GTO basis, and every
twist shares the same J3 basis and matrix. `-j3 mo` is rejected: MO orbitals
differ from one twist to another, so they cannot form one shared J3 basis.
`--ao-conv-to` is not available. Every twist, including $\Gamma$, uses
the complex Bloch determinant representation. The output selects the twist-averaged
calculation in `jqmc` in the same way as a `jqmc-tool qe --twist-average` file.

For the plane-wave based representations (`pgto+pw`, `pgto+bspline`, `pw`,
`bspline`), convert a Quantum ESPRESSO save directory with `jqmc-tool qe`
instead; see {doc}`ao` and {doc}`workflows`.

#### 8. Pseudopotentials (ECPs)

*   The function checks for the existence of ECP data using `trexio.has_ecp_num`.
*   If present, `Coulomb_potential_data` is populated with:
    *   Core charges ($Z_{\text{core}}$)
    *   Angular momentum limits
    *   Exponents, coefficients, and powers ($n$) for the Gaussian expansion of the potential.
