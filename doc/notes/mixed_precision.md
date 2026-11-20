# Mixed Precision

jQMC supports mixed precision computation, allowing selected parts of the
calculation to run in float32 while keeping numerically sensitive operations
in float64. This can reduce device memory and improve GPU throughput, but the
gain depends on the AO representation, system size, and active observables.
Benchmark the representative production workload and verify the reported
observables against full precision before adopting it for a study (see
{ref}`sec:mixed_precision_validation`).

CPU Rust and CPU/GPU XLA all support this policy. Precision does not select
the sampling backend; see {doc}`../execution` for backend and thread controls.

## Quick start

Add a `[precision]` section to your TOML input file:

```toml
[precision]
mode = "mixed"
```

Or keep the default (all float64, the behavior of earlier jQMC versions):

```toml
# [precision] section omitted -> mode="full"
```

## Precision modes

| Mode    | Description |
|---------|-------------|
| `full`  | All zones float64 (default) |
| `mixed` | Recommended mixed precision (see zone table below) |

For a **complex Bloch wave function** (non-$\Gamma$ twist), every zone that
consumes a complex AO basis uses the complex dtype with the same component
precision: fp64 -> complex128 and fp32 -> complex64. Model and checkpoint
storage may remain complex128; the zone-owning function performs the local
arithmetic cast. The user-facing TOML key (`mode = "mixed"` or
`mode = "full"`) is the same for real and complex wave functions.

## Precision zones

jQMC divides the computation into 23 **Precision Zones**.  Each zone is
owned by exactly one module and is named for its *purpose* (not its
dtype).  The mapping from zone to dtype is determined entirely by the
chosen mode and, for complex wave functions, by the AO basis type at runtime.

> **Component-width rule for complex data.** No `*_complex` zone strings exist
> in `_precision.py`. `get_complex_dtype_jnp(zone)` and
> `get_complex_dtype_np(zone)` derive the corresponding complex dtype from
> the zone's real component width. `cast_to_zone` preserves whether a path is
> complex but does not allow complex128 storage to override an f32 zone.

| Zone               | Owning module          | `full` | `mixed`  | Complex dtype in `mixed` | risk     | E_L path   |
|--------------------|------------------------|--------|----------|--------------------------|----------|------------|
| `ao_eval`          | `atomic_orbital`       | f64    | **f32**  | complex64                | low      | core       |
| `ao_bloch_eval`    | `atomic_orbital`       | f64    | f64      | complex128               | high     | core       |
| `ao_grad_lap`      | `atomic_orbital`       | f64    | f64      | complex128               | high     | core       |
| `mo_eval`          | `molecular_orbital`    | f64    | f64      | complex128               | high[^mo] | core       |
| `mo_grad`          | `molecular_orbital`    | f64    | f64      | complex128               | high     | core       |
| `mo_lap`           | `molecular_orbital`    | f64    | f64      | complex128               | high     | core       |
| `tiling_eval`      | `orbital_tiling`      | f64    | f64      | complex128               | high     | core       |
| `tiling_grad`      | `orbital_tiling`      | f64    | f64      | complex128               | high     | core       |
| `tiling_lap`       | `orbital_tiling`      | f64    | f64      | complex128               | high     | core       |
| `jastrow_eval`     | `jastrow_factor`       | f64    | **f32**  | f32 (real)               | low      | core[^eval] |
| `jastrow_grad_lap` | `jastrow_factor`       | f64    | **f32**  | f32 (real)               | low      | core       |
| `jastrow_grad_lap_carry` | `jastrow_factor` | f64    | f64      | f64 (real)               | high[^carry] | core   |
| `jastrow_ratio`    | `jastrow_factor`       | f64    | f64      | f64 (real)               | high     | indirect[^ratio] |
| `det_eval`         | `determinant`          | f64    | f64      | complex128               | high     | core       |
| `det_grad_lap`     | `determinant`          | f64    | f64      | complex128               | high     | core       |
| `det_ratio`        | `determinant`          | f64    | f64      | complex128               | high     | indirect[^ratio] |
| `coulomb`          | `coulomb_potential`    | f64    | f64      | f64 (real)               | low-med  | core       |
| `ewald`            | `coulomb_potential`    | f64    | f64      | f64 (real)               | low      | core       |
| `wf_eval`          | `wavefunction`         | f64    | f64      | complex128               | high[^mo] | core[^eval] |
| `wf_kinetic`       | `wavefunction`         | f64    | f64      | complex128               | high     | core       |
| `wf_ratio`         | `wavefunction`         | f64    | f64      | complex128               | high     | no         |
| `local_energy`     | `hamiltonians`         | f64    | f64      | complex128               | high     | core       |
| `swct`             | `swct`                 | f64    | f64      | f64 (real)               | high     | no         |

> **Why `ao_grad_lap` is one zone.** The AO gradient and Laplacian share a
> single zone (both take the same dtype) because the production fused kernel
> `compute_AOs_value_grad_lap` writes value, gradient and Laplacian in one
> pass and cannot dispatch them to different dtypes. fp32 is not accurate
> enough for the analytic Laplacian, so the unified zone stays fp64 in mixed
> mode.

[^mo]: `mo_eval` (and likewise `wf_eval`) is high-risk even though the consumed AO values are fp32:
the small `mo_coefficients @ aos` matmul runs in this zone, and its
output feeds the determinant matrix, so fp32 round-off is amplified
by `log|det|`.

[^eval]: `jastrow_eval` and `wf_eval` are on the E_L core path but their
forward values (J and ln|Psi|) do not enter the E_L formula directly
(E_L depends on *derivatives* of ln|Psi|), so fp32 in these zones alone
does not bias E_L.

[^carry]: `jastrow_grad_lap_carry` is the carried DTN gradient/Laplacian of the
GFMC/LRDMC projection (seed, per-hop moved-electron delta and accumulator),
on both XLA and Rust. A fresh `jastrow_grad_lap` evaluation re-draws its fp32
rounding at every call, but a carried seed would keep the same rounding for the
whole branching step or projection interval, so the carry is fp64 in both
modes; the per-hop delta is small, so fp64 costs little. Its DTN coefficients
are held at fp64 in the carry zone on both backends.

[^ratio]: `det_ratio` and `jastrow_ratio` affect E_L **indirectly** through the
ECP non-local potential, which evaluates `Psi(R')/Psi(R)` on a
quadrature grid via rank-1 ratio updates.  In non-ECP systems these
zones have no E_L impact.

## Complex component precision

Complex dtype follows the selected zone, not the storage dtype. Complex
periodic GTO values use `ao_bloch_eval`, which stays float64/complex128 in
both modes. This guards cancellation in Bloch image sums and subsequent
local MO/inverse contractions. Rounding the completed Bloch AO before these
contractions also discards precision that higher-precision consumers cannot
recover. The policy is identical in
ordinary and tiled cells. Real GTO/J3, PW and B-spline values retain `ao_eval`
and its mixed float32/complex64 policy.

The fused AO VGL path computes its shared heavy block in `ao_grad_lap` and
returns the value in its leaf's value zone. `mo_eval` and the tiling contraction
zones consume complex128; widening a previously rounded value does not recover
its lost precision. Determinant and local-energy paths remain complex128.

The common tiled provider uses `tiling_eval`, `tiling_grad` and `tiling_lap`
for local coefficient contractions and supercell normalization. Primitive
backend evaluation retains its own AO zones. See [orbital tiling](orbital_tiling.md) for the common block model and
its physical-coordinate derivatives.

If numerical evidence requires a kernel to use complex128, its owning zone must
select f64. Widening an already-computed fp32 result to complex128 is not a
precision fallback because it cannot recover discarded bits.

## Workflow integration

When using `jqmc_workflow`, pass `precision_mode` to `VMC_Workflow`,
`MCMC_Workflow`, `LRDMC_Workflow` or `LRDMC_Ext_Workflow` (the conversion
workflows `WF_Workflow` and `QE_Workflow` have no such argument):

```python
from jqmc_workflow import VMC_Workflow

wf = VMC_Workflow(
    server_machine_name="cluster",
    num_opt_steps=20,
    precision_mode="mixed",
)
```

Per-zone assignments are defined in `_FULL_PRECISION` / `_MIXED_PRECISION`
inside `jqmc/_precision.py` and are not configurable from TOML or workflow
parameters. Changing an assignment requires updating the corresponding mode
map and calling `configure()` again.

## What always stays in float64

Independently of the mode, the following are always float64 (complex128 for
complex data):

* the walker coordinates and the MCMC / GFMC driver state (weights,
  accumulated observables);
* the stored model data: AO exponents and coefficients, nuclear coordinates,
  MO and geminal coefficients, Jastrow parameters, as loaded from TREXIO /
  HDF5 or produced by the optimizer (see the HDF5 section below for the
  in-memory copy used in mixed mode);
* the analytic derivative kernels that build the integrands of atomic forces,
  stress and parameter derivatives. In mixed mode these kernels still receive
  some float32 inputs from the zones above (for example AO values), so the
  sampled distribution changes by a relative amount of order $10^{-7}$, far
  below the statistical error; the derivative arithmetic itself is never done
  in float32.

The zone assignment follows design rules for contributors (each zone is owned
by one module, and the function doing the arithmetic performs the cast); they
are documented in the module docstring of `jqmc/_precision.py`.

(sec:mixed_precision_validation)=

## Validating mixed precision for your system

Before using `mode = "mixed"` in production, compare it with `mode = "full"`
on a small but representative system (same basis, same type of wave function
and ECP, a smaller cell or molecule):

1. Run the same `mcmc` job (same `hamiltonian_data.h5`, same `mcmc_seed`,
   same number of steps) once with each mode.
2. Compare the energy, the energy variance and, if you need them, the atomic
   forces or the stress. The differences should be compatible with zero
   within the combined error bars.
3. If you optimize wave functions in mixed mode, also compare the final VMC
   energy of a short optimization in both modes.
4. Measure the time per step and the device memory in both runs; the gain
   depends on the system, so decide on the basis of this measurement.

All job types (`mcmc`, `vmc`, `lrdmc-bra`, `lrdmc-tau`) accept the
`[precision]` section. The run log prints the active mode and the per-zone
assignment at start-up.

## API reference

The programmatic API is in the module `jqmc._precision`:

* `configure(mode)` -- activate `"full"` or `"mixed"`.
* `is_mixed_precision_enabled()` -- `True` when at least one zone is float32.
* `mode_label()` -- a short label of the active mode, e.g.
  `"Mixed Precision (FP32 + FP64)"`.
* `zone_detail()` -- a multi-line string listing the dtype of every zone.
* `get_dtype_jnp(zone)` / `get_dtype_np(zone)` -- return the JAX / NumPy
  dtype currently assigned to *zone*.
* `get_complex_dtype_jnp(zone)` / `get_complex_dtype_np(zone)` -- return the
  complex dtype with the zone's component width.
* `cast_to_zone(value, zone, *extra_dtypes)` -- cast arithmetic data without
  dropping an upstream complex component.
* `precision_policy_snapshot()` -- copy the complete 23-zone policy for a
  native run context.

The following helpers are test utilities:

* `get_tolerance(zone, level)` -- return `(atol, rtol)` for tests, scaled
  by the zone's current dtype (`level` = `"strict"` or `"medium"`). The
  medium tier is reserved for explicitly reviewed numerically amplified
  comparisons; ordinary same-quantity parity uses strict.
* `get_tolerance_min(zones, level)` -- return the loosest `(atol, rtol)`
  across the given zones.  Use this when a test compares two paths whose
  combined dtype span crosses multiple zones; the achievable agreement
  is bounded by the weakest zone on the path.

`get_eps(name, dtype)` lives in the module `jqmc._setting`, not
`jqmc._precision`; it returns dtype-aware numerical-stability constants such
as `"rcond_svd"` and `"machine_precision"`.

## HDF5 save/load round-trip semantics

`Hamiltonian_data.save_to_hdf5` / `load_from_hdf5` does **not** guarantee
bit-exact dtype preservation across precision modes. On load, six hooks fire
in order:

1. `_apply_image_expansion_post_load` (re-tiles periodic AO bases);
2. `_apply_ewald_rebuild_post_load` (recomputes the heavy Ewald arrays from
   the persisted `eta` / `r_real_cut` / `g_max`);
3. `_apply_jastrow_structure_backfill_post_load` (back-fills missing
   structure-data references in legacy checkpoints);
4. `_apply_ao_basis_dtype_alignment_post_load`,
   `_apply_geminal_dtype_alignment_post_load`, and
   `_apply_dtn_dtype_alignment_post_load` align Gaussian model storage,
   geminal storage and DTN parameter dtypes respectively.

Gaussian radial exponents and coefficients remain float64 in both modes,
including after HDF5 load and basis updates. The value kernels own their
`ao_eval` or `ao_bloch_eval` arithmetic casts; loading does not quantize
Gaussian model data. Geminal alignment targets `det_eval`, which remains
float64/complex128. Jastrow parameters (J1 / J2 parameters, the J3 `j_matrix`
and the DTN independent coefficients) are float64 in both modes at
construction, optimizer update and load; DTN alignment lifts a
float32 checkpoint written by an older mixed run to float64.
Older checkpoints that already contain float32 Gaussian parameters are lifted
to float64 on load, but the precision lost when those old files were written
cannot be recovered. New mixed-mode files preserve float64 Gaussian storage.

The round-trip identity is therefore:

| Mode | Equality |
|---|---|
| `full` | bit-exact (all zones `float64`) |
| `mixed` | equal up to the loosest zone tolerance among `(ao_eval, jastrow_eval, det_eval)` |

**Implication for callers.** If you call `Hamiltonian_data.load_from_hdf5`
and compare the loaded values to the in-memory originals, use the tolerance
helper `get_tol_hdf5_roundtrip(level)` from `jqmc._setting` rather than an
exact comparison. Bit-exact equality is only available in full mode.
