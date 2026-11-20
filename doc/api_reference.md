(api_reference_link)=

# Public Python API (jqmc)

This page documents the **public API** of the `jqmc` package: exactly the names
re-exported from `jqmc/__init__.py` (its `__all__`). These are the objects a
Python script needs to run jQMC without the command-line interface
(see `jqmc-example11`, `jqmc-example12` and `jqmc-example13` in {doc}`examples`):

- the sampling drivers `MCMC` (VMC sampling and optimization), `GFMC_t` and
  `GFMC_n` (the two LRDMC drivers);
- the user-assembled data containers: `Structure_data`,
  `Coulomb_potential_data`, `Hamiltonian_data`, `Wavefunction_data`, the
  Jastrow containers (`Jastrow_data`, `Jastrow_one_body_data`,
  `Jastrow_two_body_data`, `Jastrow_three_body_data`, `Jastrow_DTN_data`) and
  the determinant containers (`SD_data`, `AGP_data`, `AGPn_data`, `to_AGPn`);
- the one-particle basis containers (`MOs_data` and the `AOs_*_data` classes for
  Gaussian, periodic Gaussian, plane-wave, B-spline and composite representations)
  and their evaluation kernels `compute_AOs`, `compute_AOs_grad`,
  `compute_AOs_laplacian`, `compute_MOs`, `compute_MOs_grad`, `compute_MOs_laplacian`;
- the Hamiltonian terms at a given electron configuration: the bare Coulomb
  channels (`compute_bare_coulomb_potential` and its `_el_el`, `_el_ion`,
  `_ion_ion` parts) and the ECP local and non-local parts
  (`compute_ecp_local_parts_all_pairs`, `compute_ecp_non_local_parts_all_pairs`,
  `compute_ecp_non_local_parts_nearest_neighbors`), see `jqmc-example11`;
- the `read_trexio_file` loader and the `cli_parameters` input schema.

These names are kept stable within a release series. Everything else in the
`jqmc.*` sub-modules is internal machinery (numeric kernels, builders,
file-format parsers) that requires knowledge of jqmc internals to use and may
change without notice. The complete module-level reference generated from the
docstrings, including these internal modules, is in
{ref}`api_ref_modules`.

```{eval-rst}
.. automodule:: jqmc
   :members:
```

## `GFMC_n` projections per measurement

`num_mcmc_per_measurement` of `GFMC_n`
is required and has no default, like `[lrdmc-bra] num_mcmc_per_measurement` in
the CLI: constructing the driver without it, or with `None`, raises
`ValueError` before any sampling. `run_twist_average_gfmc(gfmc_kind="n", ...)`
likewise requires it in `gfmc_kwargs`. The value is the number of projections
between two branchings; the imaginary time between them is about
`num_mcmc_per_measurement * alat**2 / (3 N_e)`, so choose it for the system and
the lattice spacing. An integer >= 1 is used unchanged, and
`GFMC_n.load_from_hdf5` restores the checkpoint's value. `GFMC_t` has no such
parameter (its time per measurement is `tau`).

## Inherited LRDMC result and restart methods

Both LRDMC drivers expose the following public methods inherited from their
shared driver base. All ranks in the driver's communicator must participate
in collective estimators; do not call them on rank zero alone. Force/stress
results require the corresponding observable to have been enabled at sampling;
the MPC estimators require a driver constructed with `mpc=True`. Each driver
writes its checkpoint with its own `save_to_hdf5` method.

```{eval-rst}
.. automethod:: jqmc.GFMC_n.get_E
.. automethod:: jqmc.GFMC_n.get_aF
.. automethod:: jqmc.GFMC_n.get_aT
.. automethod:: jqmc.GFMC_n.get_aP
.. automethod:: jqmc.GFMC_n.get_P_hydrostatic
.. automethod:: jqmc.GFMC_n.get_E_mpc
.. automethod:: jqmc.GFMC_n.get_dE_mpc
.. automethod:: jqmc.GFMC_n.load_from_hdf5
.. automethod:: jqmc.GFMC_t.get_E
.. automethod:: jqmc.GFMC_t.get_aF
.. automethod:: jqmc.GFMC_t.get_aT
.. automethod:: jqmc.GFMC_t.get_aP
.. automethod:: jqmc.GFMC_t.get_P_hydrostatic
.. automethod:: jqmc.GFMC_t.get_E_mpc
.. automethod:: jqmc.GFMC_t.get_dE_mpc
.. automethod:: jqmc.GFMC_t.load_from_hdf5
```
