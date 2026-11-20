# TurboRVB reference revalidation

The seven AE/ECP tests' historical literals were independently re-evaluated on
2026-09-25. This records **a new revalidation**, not the unknown build/date of
the original 2025 calculation. No jQMC production function supplies the oracle.

The [machine report](turborvb_reference_provenance.json) contains every fixed
old/new electron configuration, actual/reference values and differences, input
hashes, build commands, and hashes of the actual external libraries, executable,
and generated harness (``<work>/`` is the run's temporary directory). The file
hashes are those of the inputs of this run. Absolute system paths identify this
run, not dependencies required at those paths on another host.

## What was evaluated

| Tests | Committed TurboRVB input | Enabled Jastrow |
| --- | --- | --- |
| AE, without Jastrow | `turborvb_WF_w_1b_2b_1b3b_ae.txt` | none |
| AE, full Jastrow | same | original J1/J2/J3 |
| ECP, without Jastrow | `turborvb_WF_w_2b_1b3b_w_ecp.txt` | none |
| ECP, J2 | same | J2 only |
| ECP, J2/J3 | `turborvb_WF_w_2b_3b_w_ecp.txt` | original J2/J3 |
| ECP, J2/J1J3, including the full acceptance test | `turborvb_WF_w_2b_1b3b_w_ecp.txt` | original J2/J1J3 |

The runner takes the exact coordinates and reference literals from the two
committed test modules. It retains the original external `testad` initializer,
overrides **both** `kel` and `kelind`, evaluates `compute_eloc_logpsi` once, and
stops before its derivative/random-walk tests. `subener` supplies the external
kinetic energy. Rydberg energies are divided by two to get Hartree.
Jastrow removal disables the two-body selector, one-body selector, and J3 matrices;
the J2-only case disables only the J3 matrices (the ECP fixture has no J1 cusp).

The last ECP test's `WF_ratio` is the **AS-regularized ratio**, not the raw squared
wavefunction ratio. Its AS scales and weights are recomputed from TurboRVB's
inverse geminal; the transition factor is evaluated from the fixed coordinates
and TREXIO valence charges. The report also lists the inverse geminals and the
squared Frobenius norms, which the tests do not assert; their reproduction does
not certify them.

## ECP serialization matters

The historical converter writes coefficients and exponents with **six decimal
places** and uses `pseudo.x` at a cutoff-selection threshold of `0.00001`.
Reproducing that recipe from `water_ccecp_ccpvqz.h5` gives radii **1.08, 0.80,
0.80 bohr**. The quadrature is six directions, without rotation. The local
channel uses TREXIO's `max_ang_mom_plus_1`, and radial powers are converted by
adding two. The converter implementation inspected was TurboGenius commit
`db1801c9ce8085f8970fb6e8d33f748ff672f7e6`,
`turbogenius/pyturbo/pseudopotentials.py::write_pseudopotential_turborvb_file`
and `set_cutoffs`. This is a revalidated recipe, not proof that this newer commit
was used in the original run.

Using unrounded coefficients and a 10-bohr cutoff instead changes the ECP
potential by up to `1.43e-5 Ha`. That is an **input mismatch**, not evidence of a
jQMC production defect. The original ECP literals must not be described as
untruncated/full-precision TREXIO-potential references.

## Result and limits

The external checkout was `def81edc3486a74e00e875ecdb908bd5a6d76468`
(2024-06-27), with no tracked changes. The recorded library/binary hashes identify
the actual installed build; checkout identity alone does not prove how those
pre-existing libraries were compiled.

- AE energies and wavefunction ratios: maximum difference `6.89e-15`.
- ECP kinetic energy: maximum difference `2.85e-14 Ha`.
- ECP total potential: maximum difference `6.40e-13 Ha`.
- AS-regularized wavefunction/transition/final ratios: maximum difference `1.36e-14`.
- Geminals: `1.78e-15`; inverse geminals: `1.24e-12`.
- Historical squared Frobenius norms near 54000: absolute difference `8.52e-10`
  (relative about `1.6e-14`), consistent with the printed reference precision.

This is fixed-input external regression provenance, not an arbitrary-wavefunction
accuracy proof, a full/mixed cross-comparison, or a repository CI certificate.
The ordinary tests still evaluate each supported jQMC precision separately.

## Reproduce

Use a serial TurboRVB build containing the pinned `testad` source, its serial
module files and static libraries, and `bin/pseudo.x`. Then run in the jQMC
environment (NumPy and h5py are needed, JAX is not):

```sh
python scripts/revalidate_turborvb_references.py \
  --turborvb-root /path/to/turborvb \
  --compiler /usr/bin/gfortran \
  --blas /path/to/libopenblas.so
```

The command creates a fresh temporary directory, builds only a short external
harness against those libraries, and writes raw logs and `report.json`. It never
edits the TurboRVB checkout, test literals, tolerance settings, or GDD contracts.
Unsupported source layout/hash and external execution errors fail explicitly.
It reports differences; it does **not** silently replace references or declare
new tolerances. The generated harness retains TurboRVB's GPL notice and remains
an external build artifact, not vendored jQMC production code.
