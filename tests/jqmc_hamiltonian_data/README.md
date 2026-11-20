# Ordinary Li supercell input for periodic J3 basis validation

Repository path: `tests/jqmc_hamiltonian_data/periodic_j3_basis_audit_initial.h5`.

SHA256: `738c99afececf6d53437358d991edc0afa58f16f7ee26452fdb8b0bd83ab15c3`.
Size: approximately 304 KiB. This is a jQMC Hamiltonian HDF5 input, not a TREXIO file and not an expected optimizer output.

Source: an independent orbital-tiling validation.

The validation converted the committed Li QE sector data through the converter CLI, selected sector1, used the independent ordinary physical-supercell `ao-small` J3 basis with supercell matrix `[[2,1,0],[0,1,0],[0,0,-1]]`, and expanded the tiled PW determinant explicitly into an ordinary supercell PW SD. Thus loading this input requires no determinant-tiling feature.

The J3 matrix is exactly `M = np.random.default_rng(41).normal(scale=0.02, size=j3.j_matrix.shape)` followed by `M[:, :-1] = (M[:, :-1] + M[:, :-1].T) / 2`; the final column is not symmetrized. Coordinates are `ru = [[.31,.43,.57],[1.12,-.62,.91]]` and `rd = ru + [.13,.21,-.17]`. The test uses the actual MCMC seed718 with two walkers, four steps, one SR update, two bins, delta=epsilon=1e-3, and only J3 exponent optimization enabled.

The fixture supplies physical input data only. Acceptance oracles are positive finite exponents; exact physical shell/image ties; unchanged matrix when only basis exponents are selected; and exact within-mode HDF5/real-checkpoint continuation. No output from the optimizer is stored as the numerical reference.
