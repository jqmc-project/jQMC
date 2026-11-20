"""Stored MCMC stress buffers against AD of an independently written strained e_L.

The production MCMC driver stores, per walker and per measurement,
``stress_HF = d e_L / d eps``, ``stress_PP = d ln|Psi| / d eps`` and
``E_L_stress_PP = e_L * stress_PP``, where the strain ``F = I + eps`` maps the
cell rows, every nucleus and every electron as ``x -> F x`` (the fixed static
ansatz: basis exponents, MO coefficients and Jastrow parameters are held
fixed).  The reference in this module is a separate, closed-form program for
exactly that functional: it never calls a production evaluator, strain helper
(``_apply_strain``, ``_strain_structure_data``, ``accumulate_strain_grad``),
stress kernel or ``_auto`` twin.  It is written from the documented formulas:

* Cartesian periodic Gaussian orbitals
  ``N(a, n) (x-X)^nx (y-Y)^ny (z-Z)^nz exp(-a |r - R - T|^2)`` summed over a
  fixed 7x7x7 lattice-image block around the home cell, with
  ``N = sqrt((2a/pi)^(3/2) (8a)^l nx! ny! nz! / ((2nx)! (2ny)! (2nz)!))``;
* one up and one down electron, so each spin determinant is the orbital value;
* the exponential J2 ``(1 - exp(-b d)) / (2 b)`` on the positive
  full-Gram periodic distance (MATH-JAS-PERIODIC-DISTANCE-LAW-001): with
  ``f = d inv(C)``, ``h = sin(pi f) cos(pi f)`` and ``z = sin^2(pi f)``,
  ``d^2 = (|h C|^2 + sum_k |a_k|^2 z_k^2) / pi^2``.  The
  diagonal-Gram sine law ``d^2 = sum_k (|a_k| / pi)^2 sin^2(pi f_k)``, whose
  cusp is direction dependent in this triclinic cell, is kept only as a
  negative control that the stored buffers must reject;
* the kinetic energy ``-1/2 (lap ln Psi + |grad ln Psi|^2)`` by ``jax.hessian``;
* the total Ewald energy of the neutral point-charge cell with this file's own
  splitting parameter, real-space images and reciprocal set (the total is
  independent of the split, so it need not match production's per-channel
  parameters).

``jax.jacrev`` of that program with respect to ``eps`` at zero is the per-walker
reference for all nine tensor entries (the six symmetric components plus the
antisymmetric part, which the p-type orbitals make non-zero).

Precision: the program is evaluated in the
active mode's zone dtypes, never as an fp64 answer for a mixed buffer.  AO
values are computed at ``ao_eval`` and every AO derivative at ``ao_grad_lap``;
J2 values at ``jastrow_eval`` and every J2 derivative at ``jastrow_grad_lap``
(see ``_zoned``).  Geometry, the MO contraction, ``ln|det|``, the kinetic
assembly and the Ewald sum run in the caller's fp64, which is the dtype of
their production zones (``mo_eval``, ``det_eval``, ``wf_kinetic``, ``ewald``)
in both modes; the test asserts that policy before using the reference.  In
full mode every zone is fp64 and the program is unchanged.
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in
#   the documentation and/or other materials provided with the
#   distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import itertools
import math
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import erfc

from jqmc._precision import get_dtype_jnp, get_tolerance_cancelling_mean, get_tolerance_min, is_mixed_precision_enabled
from jqmc.atomic_orbital import AOs_cart_pgto_data
from jqmc.coulomb_potential import Coulomb_potential_data, build_ewald_data
from jqmc.determinant import SD_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.wavefunction import Wavefunction_data
from tests.test_rs_kernel.conftest import assert_native_ran, assert_no_native_ran, native_ctx_spy
from tests.test_rs_kernel.test_gfmc_projection import _backend_env

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]

#: Triclinic cell (rows are lattice vectors, bohr): no two rows orthogonal.
_CELL = 5.2 * (np.eye(3) + np.array([[0.05, 0.12, -0.04], [0.03, -0.06, 0.09], [-0.08, 0.04, 0.10]]))
#: Two all-electron hydrogen nuclei, fractional coordinates.
_NUCLEI_FRAC = np.array([[0.21, 0.27, 0.19], [0.62, 0.58, 0.66]])
#: One primitive per AO: (nucleus, exponent, Cartesian powers). s and p shells.
_SHELLS = (
    (0, 0.55, (0, 0, 0)),
    (0, 1.40, (0, 0, 0)),
    (0, 0.80, (1, 0, 0)),
    (0, 0.80, (0, 1, 0)),
    (0, 0.80, (0, 0, 1)),
    (1, 0.60, (0, 0, 0)),
    (1, 1.20, (0, 0, 0)),
    (1, 0.90, (1, 0, 0)),
    (1, 0.90, (0, 1, 0)),
    (1, 0.90, (0, 0, 1)),
)
_C_UP = np.array([[0.9, 0.3, 0.25, -0.15, 0.1, 0.5, 0.2, -0.2, 0.3, 0.12]])
_C_DN = np.array([[0.4, 0.2, -0.1, 0.3, -0.25, 0.8, 0.35, 0.15, -0.1, 0.2]])
_J2_PARAM = 1.1
#: Production AO image truncation for this fixture: far below the comparison
#: tolerance so the reference's fixed image block and production's truncated
#: image set describe the same periodic orbital.
_AO_IMAGE_TOL = 1.0e-14

_ZONES = ("ao_eval", "ao_grad_lap", "det_eval", "jastrow_eval", "jastrow_grad_lap", "wf_eval", "wf_kinetic", "coulomb", "ewald")

# --- independent reference tables (integer sets, fixed for the derivative) ---
_AO_IMAGES = np.array(list(itertools.product(range(-3, 4), repeat=3)), dtype=np.float64)
_EWALD_IMAGES = np.array(list(itertools.product(range(-2, 3), repeat=3)), dtype=np.float64)
_EWALD_G = np.array([m for m in itertools.product(range(-9, 10), repeat=3) if m != (0, 0, 0)], dtype=np.float64)
_EWALD_ETA = 0.75
_CHARGES = np.array([1.0, 1.0, -1.0, -1.0])


def build_hamiltonian():
    """Production Hamiltonian of the fixture, from raw inputs through the public builders."""
    positions = _NUCLEI_FRAC @ _CELL
    structure = Structure_data.build(
        positions=positions,
        pbc_flag=True,
        vec_a=_CELL[0],
        vec_b=_CELL[1],
        vec_c=_CELL[2],
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H0", "H1"),
    )
    n = len(_SHELLS)
    aos = AOs_cart_pgto_data.build(
        structure_data=structure,
        nucleus_index=[shell[0] for shell in _SHELLS],
        num_ao=n,
        num_ao_prim=n,
        orbital_indices=list(range(n)),
        exponents=np.array([shell[1] for shell in _SHELLS]),
        coefficients=np.ones(n),
        angular_momentums=[sum(shell[2]) for shell in _SHELLS],
        polynominal_order_x=[shell[2][0] for shell in _SHELLS],
        polynominal_order_y=[shell[2][1] for shell in _SHELLS],
        polynominal_order_z=[shell[2][2] for shell in _SHELLS],
        tol=_AO_IMAGE_TOL,
    )
    determinant = SD_data.build(
        num_electron_up=1,
        num_electron_dn=1,
        mos_up=MOs_data.build(aos_data=aos, mo_coefficients=_C_UP),
        mos_dn=MOs_data.build(aos_data=aos, mo_coefficients=_C_DN),
        time_reversal=False,
    )
    jastrow = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=_J2_PARAM, jastrow_2b_type="exp", structure_data=structure
        ),
        jastrow_three_body_data=None,
        jastrow_dtn_data=None,
    )
    potential = Coulomb_potential_data(structure_data=structure)
    potential = potential.replace(ewald_data=build_ewald_data(structure, potential, n_electron=2))
    wavefunction = Wavefunction_data(geminal_data=determinant, jastrow_data=jastrow)
    return Hamiltonian_data.build(structure_data=structure, coulomb_potential_data=potential, wavefunction_data=wavefunction)


def _zoned(value_zone, derivative_zone):
    """Evaluate a reference piece where production evaluates it: primal and derivatives by zone.

    ``program(dtype, *args)`` reconstructs geometry in the caller's precision
    (fp64 walker state), does its own arithmetic in ``dtype`` and returns the
    result in the caller's dtype, as production's owning module hands its zone
    output to the fp64 consumer.  The primal is evaluated at the active mode's
    ``value_zone`` dtype and its JVP -- hence every higher derivative taken by
    the Hessian kinetic energy and the strain Jacobian -- at ``derivative_zone``.
    In full mode both are float64 and this is the plain fp64 program.  In mixed
    mode the reference rounds where production rounds, so a mixed buffer is
    never judged against an fp64 answer.
    """

    def decorate(program):
        @jax.custom_jvp
        def zoned(*args):
            return program(get_dtype_jnp(value_zone), *args)

        @zoned.defjvp
        def zoned_jvp(primals, tangents):
            _, tangent = jax.jvp(partial(program, get_dtype_jnp(derivative_zone)), primals, tangents)
            return zoned(*primals), tangent

        return zoned

    return decorate


@_zoned("ao_eval", "ao_grad_lap")
def _ao_values(dtype, cell, nuclei, r):
    """Every normalized Cartesian periodic Gaussian AO at ``r``, summed over the fixed image block."""
    frac = r @ jnp.linalg.inv(cell)
    r_home = (frac - jnp.floor(frac)) @ cell
    shifts = _AO_IMAGES @ cell
    values = []
    for atom, a, powers in _SHELLS:
        ratio = math.prod(math.factorial(p) for p in powers) / math.prod(math.factorial(2 * p) for p in powers)
        norm = jnp.asarray(math.sqrt((2.0 * a / math.pi) ** 1.5 * (8.0 * a) ** sum(powers) * ratio), dtype=dtype)
        d = (r_home[None, :] - nuclei[atom][None, :] - shifts).astype(dtype)
        poly = d[:, 0] ** powers[0] * d[:, 1] ** powers[1] * d[:, 2] ** powers[2]
        values.append(norm * jnp.sum(poly * jnp.exp(-jnp.asarray(a, dtype=dtype) * jnp.sum(d * d, axis=1))))
    return jnp.stack(values).astype(r.dtype)


def _orbital(coefficients, cell, nuclei, r):
    """MO value: the fp64 coefficient contraction (``mo_eval``) of the zone-evaluated AO values."""
    return jnp.asarray(coefficients) @ _ao_values(cell, nuclei, r)


def _full_gram_distance(f, cell):
    """Adopted law on fractional ``f``: ``sqrt(|h C|^2 + sum_k |a_k|^2 z_k^2) / pi`` (Euclidean cusp)."""
    phase = jnp.pi * f
    h = jnp.sin(phase) * jnp.cos(phase)
    z = jnp.sin(phase) ** 2
    v = h @ cell
    return jnp.sqrt(jnp.sum(v * v) + jnp.sum(jnp.sum(cell * cell, axis=1) * z * z)) / jnp.pi


def _superseded_diagonal_distance(f, cell):
    """Diagonal-Gram sine law; negative control only (direction-dependent cusp when skew)."""
    lengths = jnp.sqrt(jnp.sum(cell * cell, axis=1))
    v = lengths / jnp.pi * jnp.sin(jnp.pi * f)
    return jnp.sqrt(jnp.sum(v * v))


def _j2_program(law):
    def program(dtype, cell, d):
        f = (d @ jnp.linalg.inv(cell)).astype(dtype)
        distance = law(f, cell.astype(dtype))
        return ((1.0 - jnp.exp(-_J2_PARAM * distance)) / (2.0 * _J2_PARAM)).astype(d.dtype)

    return _zoned("jastrow_eval", "jastrow_grad_lap")(program)


_J2 = {law: _j2_program(law) for law in (_full_gram_distance, _superseded_diagonal_distance)}


def _ln_psi(cell, nuclei, r_up, r_dn, law=_full_gram_distance):
    """``ln|Psi|``: the zone-evaluated J2 plus the log of each one-electron determinant (fp64 zones)."""
    j2 = _J2[law](cell, r_up[0] - r_dn[0])
    up = _orbital(_C_UP[0], cell, nuclei, r_up[0])
    dn = _orbital(_C_DN[0], cell, nuclei, r_dn[0])
    return jnp.log(jnp.abs(up)) + jnp.log(jnp.abs(dn)) + j2


def _ewald_total(cell, points):
    inv = jnp.linalg.inv(cell)
    shifts = _EWALD_IMAGES @ cell
    not_origin = np.any(_EWALD_IMAGES != 0, axis=1)
    e_real = 0.0
    for i, j in itertools.product(range(len(_CHARGES)), repeat=2):
        f = (points[i] - points[j]) @ inv
        v = ((f - jnp.round(f)) @ cell)[None, :] + shifts
        dist2 = jnp.sum(v * v, axis=1)
        if i == j:
            dist = jnp.sqrt(jnp.where(not_origin, dist2, 1.0))
            term = jnp.where(not_origin, erfc(_EWALD_ETA * dist) / dist, 0.0)
        else:
            dist = jnp.sqrt(dist2)
            term = erfc(_EWALD_ETA * dist) / dist
        e_real = e_real + 0.5 * _CHARGES[i] * _CHARGES[j] * jnp.sum(term)
    g = _EWALD_G @ (2.0 * jnp.pi * inv.T)
    g2 = jnp.sum(g * g, axis=1)
    phase = g @ points.T
    s2 = (jnp.cos(phase) @ _CHARGES) ** 2 + (jnp.sin(phase) @ _CHARGES) ** 2
    volume = jnp.abs(jnp.linalg.det(cell))
    e_recip = (2.0 * jnp.pi / volume) * jnp.sum(jnp.exp(-g2 / (4.0 * _EWALD_ETA**2)) / g2 * s2)
    e_self = -_EWALD_ETA / jnp.sqrt(jnp.pi) * np.sum(_CHARGES**2)
    return e_real + e_recip + e_self


def _local_energy(cell, nuclei, r_up, r_dn, law=_full_gram_distance):
    def ln_psi_flat(x):
        return _ln_psi(cell, nuclei, x[:3].reshape(1, 3), x[3:].reshape(1, 3), law)

    x = jnp.concatenate([r_up.reshape(-1), r_dn.reshape(-1)])
    grad = jax.grad(ln_psi_flat)(x)
    kinetic = -0.5 * (jnp.trace(jax.hessian(ln_psi_flat)(x)) + jnp.sum(grad * grad))
    return kinetic + _ewald_total(cell, jnp.concatenate([nuclei, r_up, r_dn]))


def _strained(eps, r_up, r_dn, law):
    F = jnp.eye(3) + eps
    cell = jnp.asarray(_CELL) @ F.T
    nuclei = jnp.asarray(_NUCLEI_FRAC @ _CELL) @ F.T
    r_up, r_dn = r_up @ F.T, r_dn @ F.T
    return _local_energy(cell, nuclei, r_up, r_dn, law), _ln_psi(cell, nuclei, r_up, r_dn, law)


@partial(jax.jit, static_argnames="law")
def independent_strain_reference(r_up, r_dn, law=_full_gram_distance):
    """``(e_L, d e_L / d eps, d ln|Psi| / d eps)`` of the independent program at eps = 0."""
    zero = jnp.zeros((3, 3), dtype=jnp.float64)
    energy = _strained(zero, r_up, r_dn, law)[0]
    hf = jax.jacrev(lambda e: _strained(e, r_up, r_dn, law)[0])(zero)
    pp = jax.jacrev(lambda e: _strained(e, r_up, r_dn, law)[1])(zero)
    return energy, hf, pp


@lru_cache(maxsize=1)
def _hamiltonian():
    return build_hamiltonian()


def _tolerance():
    level = "medium" if is_mixed_precision_enabled() else "strict"
    return get_tolerance_min(_ZONES, level)


def _jackknife_stress_reference(weights, energy, hf, pp, volume):
    """Delete-one jackknife of the covariance-corrected stress from reference tensors.

    With one measurement per bin block every (measurement, walker) sample is
    its own jackknife bin, so replicate ``k`` is the weighted average over all
    other samples; each replicate is symmetrised and divided by the volume.
    """
    weights, energy = weights.reshape(-1), energy.reshape(-1)
    hf, pp = hf.reshape(-1, 3, 3), pp.reshape(-1, 3, 3)
    replicates = []
    for deleted in range(len(weights)):
        w = np.where(np.arange(len(weights)) == deleted, 0.0, weights)
        norm = np.sum(w)
        mean_hf = np.einsum("s,sij->ij", w, hf) / norm
        mean_pp = np.einsum("s,sij->ij", w, pp) / norm
        mean_epp = np.einsum("s,s,sij->ij", w, energy, pp) / norm
        mean_e = np.sum(w * energy) / norm
        sigma = mean_hf + 2.0 * (mean_epp - mean_e * mean_pp)
        replicates.append(0.5 * (sigma + sigma.T) / volume)
    replicates = np.asarray(replicates)
    mean = np.mean(replicates, axis=0)
    std = np.sqrt((len(replicates) - 1) * np.mean((replicates - mean) ** 2, axis=0))
    return mean, std


@pytest.mark.parametrize("backend", ["xla", "rust"])
def test_mcmc_stress_buffers_match_independent_strain_ad(backend):
    """Every stored per-walker stress tensor equals AD of the independent strained program.

    Fixture (module constants): triclinic cell, two all-electron H nuclei, one
    up and one down electron in a Cartesian periodic s+p Gaussian SD with the
    exponential J2 (b = 1.1), built through the public constructors.  Two
    walkers from seed 34 take two moves per measurement at Dt = 0.7, AS = 0,
    identity quadrature rotation; two successive one-measurement production
    runs keep their actual sampled positions.  At each stored position the
    stored e_L, stress_HF, stress_PP and E_L_stress_PP are compared entry-wise
    (all nine entries: six symmetric components and the antisymmetric part)
    with the independent program of the module docstring.  The antisymmetric
    part of the reference must be non-vacuous, so a transposed buffer fails.
    At every step the same program on the diagonal-Gram J2 distance must
    disagree with the stored buffers beyond the same tolerance, so the fixture
    discriminates the skew-cell geometry from the diagonal-Gram one.
    Finally ``get_aT`` on the same run is compared with a delete-one jackknife
    assembled from the reference tensors, so dropping or transposing the
    per-block symmetrisation fails.  XLA and Rust are both required.  The
    reference program runs at the active mode's zone dtypes (module docstring),
    so full and mixed are each judged against a reference of their own
    precision, at the zone policy of the force sampler reference (strict in
    full, medium in mixed); no statistical window and no fp64 answer for mixed.
    """
    caller_precision_zones = ("mo_eval", "det_eval", "wf_eval", "wf_kinetic", "coulomb", "ewald")
    assert all(get_dtype_jnp(zone) == jnp.float64 for zone in caller_precision_zones), (
        "the reference evaluates these zones in the caller's fp64; mirror any policy change in the program"
    )
    ham = _hamiltonian()
    atol, rtol = _tolerance()
    num_walkers, num_steps = 2, 2
    reference = {"e_L": [], "stress_HF": [], "stress_PP": []}
    with _backend_env("fast" if backend == "rust" else "jax"):
        driver = MCMC(
            hamiltonian_data=ham,
            num_walkers=num_walkers,
            mcmc_seed=34,
            num_mcmc_per_measurement=2,
            Dt=0.7,
            epsilon_AS=0.0,
            random_discretized_mesh=False,
            comput_position_deriv=False,
            comput_stress=True,
        )
        with native_ctx_spy() as counts:
            for step in range(num_steps):
                driver.run(num_mcmc_steps=1)
                r_up_carts = np.asarray(driver._MCMC__latest_r_up_carts)
                r_dn_carts = np.asarray(driver._MCMC__latest_r_dn_carts)
                assert r_up_carts.shape == r_dn_carts.shape == (num_walkers, 1, 3)
                rows = [independent_strain_reference(r_up_carts[w], r_dn_carts[w]) for w in range(num_walkers)]
                energy = np.array([float(row[0]) for row in rows])
                hf = np.array([np.asarray(row[1]) for row in rows])
                pp = np.array([np.asarray(row[2]) for row in rows])
                for key, value in (("e_L", energy), ("stress_HF", hf), ("stress_PP", pp)):
                    reference[key].append(value)
                antisymmetric = np.abs(hf - hf.transpose(0, 2, 1))
                assert np.all(np.max(antisymmetric, axis=(1, 2)) > atol + rtol * np.max(np.abs(hf))), (
                    "p shells must break symmetry"
                )
                stored = {
                    "e_L": np.real(np.asarray(driver._MCMC__stored_e_L)[step]),
                    "stress_HF": np.real(np.asarray(driver._MCMC__stored_stress_HF)[step]),
                    "stress_PP": np.real(np.asarray(driver._MCMC__stored_stress_PP)[step]),
                    "E_L_stress_PP": np.real(np.asarray(driver._MCMC__stored_E_L_stress_PP)[step]),
                }
                for label, expected in (
                    ("e_L", energy),
                    ("stress_HF", hf),
                    ("stress_PP", pp),
                    ("E_L_stress_PP", energy[:, None, None] * pp),
                ):
                    actual = stored[label]
                    label = f"{step}/{label}"
                    assert actual.shape == expected.shape, label
                    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), label
                    assert np.all(np.abs(expected) > atol + rtol * np.abs(expected)), f"{label}: vacuous entry"
                    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)
                # Negative control: the same program on the diagonal-Gram J2
                # geometry must be rejected by this step's comparison.
                old = [
                    independent_strain_reference(r_up_carts[w], r_dn_carts[w], law=_superseded_diagonal_distance)
                    for w in range(num_walkers)
                ]
                old_energy = np.array([float(row[0]) for row in old])
                old_pp = np.array([np.asarray(row[2]) for row in old])
                superseded = {
                    "e_L": old_energy,
                    "stress_HF": np.array([np.asarray(row[1]) for row in old]),
                    "stress_PP": old_pp,
                    "E_L_stress_PP": old_energy[:, None, None] * old_pp,
                }
                assert all(np.all(np.isfinite(value)) for value in superseded.values()), f"{step}: superseded reference"
                assert any(
                    np.any(np.abs(stored[key] - value) > atol + rtol * np.abs(value)) for key, value in superseded.items()
                ), f"{step}: the diagonal-Gram J2 geometry must be rejected"
    if backend == "rust":
        assert_native_ran(counts, "MCMC stress AD reference")
    else:
        assert_no_native_ran(counts, "MCMC stress AD reference")

    weights = np.asarray(driver.w_L)
    assert weights.shape == (num_steps, num_walkers) and np.all(np.isfinite(weights))
    energy, hf, pp = (np.asarray(reference[key]) for key in ("e_L", "stress_HF", "stress_PP"))
    volume = abs(float(np.linalg.det(_CELL)))
    expected_mean, expected_std = _jackknife_stress_reference(weights, energy, hf, pp, volume)
    actual_mean, actual_std = driver.get_aT(num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=num_steps)
    scale = max(np.max(np.abs(hf)), 2.0 * np.max(np.abs(energy[:, :, None, None] * pp))) / volume
    tensor_atol, tensor_rtol = get_tolerance_cancelling_mean(atol, rtol, scale)
    for label, actual, expected in (("get_aT/mean", actual_mean, expected_mean), ("get_aT/std", actual_std, expected_std)):
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), label
        np.testing.assert_allclose(actual, expected, atol=tensor_atol, rtol=tensor_rtol, equal_nan=False, err_msg=label)
