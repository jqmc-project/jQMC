(sec:sd_occupied_orbitals)=
# Canonical occupied-orbital Slater determinants

`SD_data` stores one determinant through occupied spin-orbital providers. Ordinary MO-backed providers hold one occupied coefficient matrix per spin with shape `(num_mo, num_ao)`; rows are occupied orbitals and columns span the full AO basis. The occupied counts satisfy `orb_data_up_spin.num_mo == num_electron_up` and `orb_data_dn_spin.num_mo == num_electron_dn`. For example, QE with `nbnd = 72` and 64 occupied bands per spin stores 64 rows per spin, not 72 rows plus zero occupations. `SD_data` has no `lambda_matrix`, `lambda_diag`, or `lambda_unpaired` field.

Occupied orbitals can still relax toward virtual directions: each occupied coefficient row spans the full AO basis. Explicit virtual-orbital rows are not stored. A finite-occupation geminal eigenvalue is different physics and belongs to `AGPn_data`. To go beyond a single determinant with ordinary MO-backed orbitals, convert the SD Hamiltonian file with `jqmc-tool hamiltonian conv-wf -c jagpn` (AGPn) or `-c jagp` (AGP) before optimization; see {doc}`../api_reference_cli_tool`.

## Options and conversion

- Ordinary MO-backed SD optimization is supported. With `time_reversal=True`, only the up block is independent; the down rows must be the conjugate of the first `N_down` up rows, and a violating stored down block is rejected. With `time_reversal=False`, both occupied coefficient matrices are independent and both are optimized.
- The QE and TREXIO converters (`jqmc-tool qe`, `jqmc-tool trexio convert-to`; see {doc}`../api_reference_cli_tool`) keep only occupied rows, independently for each spin and twist.
- An ordinary MO-backed SD stores explicitly typed `MOs_data` providers and contains no lambda dataset. Loading rejects SD files with a lambda matrix or extra MO rows; regenerate the Hamiltonian file with the converter. AGP and AGPn files are unaffected.

## Tiled occupied-orbital providers

- A tiled provider supplies occupied values per primitive k sector through separate local coefficient blocks; readers and backends do not materialize a global block-diagonal matrix. A local identity marker is accepted when a backend already supplies the occupied bands.
- The serialized representation records the normalization already applied to stored tables; do not apply an extra factor after loading.
- Tiled spin sectors use the paired sector convention; equal spin populations alone do not establish a conjugation relation. Stored phase, normalization, coset, and physical-coordinate conventions are summarized in [orbital tiling](orbital_tiling.md).
- Tiled SD stores explicitly typed `TiledOrbitalSet_data` providers and their primitive blocks, for a single sector or for each sector of an ensemble. Its normalization convention is `primitive-cell-to-supercell`.
- The current tiled interface freezes the Slater parameters, rejects dense AGP/AGPn conversion, and retains derivatives with respect to physical supercell atom positions and strain. An independent ordinary supercell J3 basis and its full coupling matrix remain separate variational Jastrow data.

Detailed formulations of occupied-orbital optimization and tiled occupied-orbital periodization are work in progress.
