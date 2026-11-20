"""Native SWCT geometry and actual driver forces against deterministic references."""

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

from itertools import product

import numpy as np
import pytest
import jax.numpy as jnp

import jqmc_rs

from jqmc import jqmc_gfmc, jqmc_mcmc
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc.structure import Structure_data
from tests.test_jqmc_kernel.test_gfmc_force_analytic import _build_ham
from tests.test_rs_kernel.conftest import assert_native_ran, native_ctx_spy

pytestmark = [pytest.mark.with_chdir]


def _geometry(boundary, empty=False, complex_field=False):
    cell = np.array([[4.0, 0.0, 0.0], [1.6, 3.1, 0.0], [0.7, 0.9, 3.7]])
    structure = Structure_data.build(
        positions=np.array([[0.2, 0.4, 0.1], [1.8, 1.3, 2.1], [2.6, 2.2, 1.1]]),
        pbc_flag=boundary == "skew-pbc",
        vec_a=cell[0] if boundary == "skew-pbc" else (),
        vec_b=cell[1] if boundary == "skew-pbc" else (),
        vec_c=cell[2] if boundary == "skew-pbc" else (),
        atomic_numbers=[1, 1, 1],
        element_symbols=["H", "H", "H"],
        atomic_labels=["H1", "H2", "H3"],
    )
    r_carts = np.array(
        [
            [[0.8, 0.6, 0.3], [2.1, 1.7, 2.5], [3.4, 2.4, 0.7]],
            [[1.2, 0.9, 1.4], [2.8, 1.1, 3.6], [0.5, 2.7, 2.8]],
        ]
    )
    fields = np.arange(36, dtype=np.float64).reshape(2, 3, 6) / 10 - 1.2
    if complex_field:
        fields = fields + 1j * fields[..., ::-1]
    if empty:
        r_carts, fields = r_carts[:, :0], fields[:, :0]
    return structure, r_carts, fields


def _reference(structure, r_carts, fields):
    """Explicit nearest-image inverse-fourth weights and their normalized derivatives."""
    positions = np.asarray(structure.positions)
    batch, _, channels = fields.shape
    contracted = np.zeros((batch, len(positions), channels), dtype=fields.dtype)
    domega = np.zeros((batch, len(positions), 3), dtype=np.float64)
    if structure.pbc_flag:
        cell = np.asarray(structure.cell)
        inverse = np.linalg.inv(cell)
        translations = np.array(list(product(range(-structure._n_max, structure._n_max + 1), repeat=3))) @ cell
    for walker in range(batch):
        for electron, point in enumerate(r_carts[walker]):
            differences = point - positions
            if structure.pbc_flag:
                fractional = differences @ inverse
                central = (fractional - np.rint(fractional)) @ cell
                candidates = central[:, None, :] + translations[None, :, :]
                best = np.argmin(np.sum(candidates**2, axis=-1), axis=1)
                differences = candidates[np.arange(len(positions)), best]
            d2 = np.sum(differences**2, axis=1)
            kappa = 1 / d2**2
            weight = kappa / np.sum(kappa)
            log_gradient = -4 * differences / d2[:, None]
            mean_gradient = np.sum(weight[:, None] * log_gradient, axis=0)
            contracted[walker] += weight[:, None] * fields[walker, electron]
            domega[walker] += weight[:, None] * (log_gradient - mean_gradient)
    return contracted, domega


@pytest.mark.light
@pytest.mark.parametrize("boundary", ["molecular", "skew-pbc"])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("complex_field", [False, True])
@pytest.mark.parametrize("structure_storage", ["numpy", "jax"])
def test_native_swct_geometry_reference(boundary, empty, complex_field, structure_storage):
    """Fused native SWCT equals explicit inverse-fourth geometry in both precision modes.

    Three synthetic nuclei, two explicit three-electron configurations and six
    real or complex field channels exercise molecular and triclinic minimum
    images. Empty-electron rows model an empty spin channel and must contribute
    exact zero. The reference uses NumPy nearest-image enumeration and normalized
    log-weight derivatives; no production SWCT implementation supplies it.
    Valid concrete NumPy and JAX-array nuclear positions represent supported
    host boundary inputs; this does not invoke a JAX SWCT evaluator.
    """
    structure, r_carts, fields = _geometry(boundary, empty, complex_field)
    if structure_storage == "jax":
        structure = structure.replace(positions=jnp.asarray(structure.positions))
        structure.sanity_check()
    actual = jqmc_rs._swct_corrections_batched(structure, r_carts, fields)
    reference = _reference(structure, r_carts, fields)
    atol, rtol = get_tolerance_min(("swct",), "strict")
    for quantity, observed, expected in zip(("contracted", "domega"), actual, reference, strict=True):
        assert np.asarray(observed).shape == np.asarray(expected).shape
        assert np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))
        np.testing.assert_allclose(observed, expected, atol=atol, rtol=rtol, equal_nan=False)
        if empty:
            np.testing.assert_array_equal(observed, np.zeros_like(expected))
        else:
            assert np.any(np.abs(expected) > atol + rtol * np.abs(expected))


@pytest.mark.light
@pytest.mark.parametrize("complex_field", [False, True])
def test_native_swct_lattice_translation(complex_field):
    """Independent lattice translations leave triclinic SWCT fields and gradients unchanged.

    The same explicit two-configuration fixture moves each electron by a
    different integer cell vector. Both the weighted real/complex field and
    summed weight derivative must equal the untranslated NumPy reference.
    """
    structure, r_carts, fields = _geometry("skew-pbc", complex_field=complex_field)
    shifts = np.array([[3, -2, 1], [-1, 2, 3], [2, 1, -3]]) @ np.asarray(structure.cell)
    reference = _reference(structure, r_carts, fields)
    actual = jqmc_rs._swct_corrections_batched(structure, r_carts + shifts[None, :, :], fields)
    atol, rtol = get_tolerance_min(("swct",), "strict")
    for quantity, observed, expected in zip(("contracted", "domega"), actual, reference, strict=True):
        assert np.asarray(observed).shape == np.asarray(expected).shape
        assert np.all(np.isfinite(observed)) and np.all(np.isfinite(expected))
        np.testing.assert_allclose(observed, expected, atol=atol, rtol=rtol, equal_nan=False)


_DRIVER_CASES = [
    ("mcmc", "H2_ae_ccpvdz_cart.h5"),
    ("mcmc", "lithium_k025.h5"),
    ("gfmc-t", "H2_ae_ccpvdz_cart.h5"),
    ("gfmc-t", "lithium.h5"),
    ("gfmc-n", "H2_ae_ccpvdz_cart.h5"),
    ("gfmc-n", "lithium.h5"),
]


@pytest.mark.heavy
@pytest.mark.parametrize("driver_name,trexio_file", _DRIVER_CASES)
def test_native_swct_driver_force_reference(monkeypatch, driver_name, trexio_file):
    """Actual native MCMC and both GFMC force buffers equal explicit SWCT reassembly.

    Committed H2 AE molecular and Li ccECP Gamma/twist fixtures, SD with one
    up/down electron and no Jastrow, use seed 34 and one walker/measurement.
    Molecule and twist MCMC exercise chunk and per-step force consumers;
    conventional and load-balanced GFMC cover molecular and periodic runs.
    Passively captured raw derivative inputs and explicit NumPy SWCT geometry
    at the actual one-walker input reconstruct stored HF/Pulay and
    energy-Pulay buffers. Raw force derivatives and local energy are shared
    inputs, not independent evidence for their physics. This is a numerical
    correctness test, not a forbidden-call or backend-routing contract.
    No statistical acceptance or cross-precision reference is used.
    """
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    raw_inputs = []
    original = jqmc_mcmc._native_swct_forces

    def capture_inputs(structure, r_up, r_dn, hf_R, hf_up, hf_dn, pp_R, pp_up, pp_dn):
        # Passive observation only: no forbidden-call or call-count policy.
        raw_inputs.append(tuple(np.array(value, copy=True) for value in (r_up, r_dn, hf_R, hf_up, hf_dn, pp_R, pp_up, pp_dn)))
        return original(structure, r_up, r_dn, hf_R, hf_up, hf_dn, pp_R, pp_up, pp_dn)

    monkeypatch.setattr(jqmc_mcmc, "_native_swct_forces", capture_inputs)
    monkeypatch.setattr(jqmc_gfmc, "_native_swct_forces", capture_inputs)
    ham = _build_ham(trexio_file)
    shared = dict(
        hamiltonian_data=ham,
        num_walkers=1,
        mcmc_seed=34,
        random_discretized_mesh=False,
        comput_position_deriv=True,
        use_swct=True,
    )
    if driver_name == "mcmc":
        driver = jqmc_mcmc.MCMC(**shared, num_mcmc_per_measurement=1, Dt=0.7, epsilon_AS=0.0)
        prefix = "_MCMC__stored_"
    elif driver_name == "gfmc-t":
        driver = jqmc_gfmc.GFMC_t(**shared, num_gfmc_collect_steps=1, alat=0.3, tau=0.05, epsilon_PW=0.0)
        prefix = "_stored_"
    else:
        driver = jqmc_gfmc.GFMC_n(
            **shared, num_gfmc_collect_steps=1, num_mcmc_per_measurement=1, alat=0.3, E_scf=-1.0, epsilon_PW=0.0
        )
        prefix = "_stored_"
    with native_ctx_spy() as counts:
        driver.run(num_mcmc_steps=1)
    assert_native_ran(counts, "native fixture for force numerical comparison")
    r_up_carts, r_dn_carts, hf_R, hf_up, hf_dn, pp_R, pp_up, pp_dn = raw_inputs[-1]
    assert r_up_carts.shape == r_dn_carts.shape == (1, 1, 3)
    up_field = np.concatenate((hf_up, pp_up), axis=-1)
    dn_field = np.concatenate((hf_dn, pp_dn), axis=-1)
    up, dup = _reference(ham.structure_data, r_up_carts, up_field)
    dn, ddn = _reference(ham.structure_data, r_dn_carts, dn_field)
    bare_hf, bare_pp = hf_R[0], pp_R[0]
    expected_hf = bare_hf + up[0, :, :3] + dn[0, :, :3]
    expected_pp = bare_pp + up[0, :, 3:] + dn[0, :, 3:] + 0.5 * (dup[0] + ddn[0])
    energy = np.asarray(getattr(driver, prefix + "e_L"))[0, 0]
    level = "medium" if is_mixed_precision_enabled() else "strict"
    zones = ("ao_eval", "ao_grad_lap", "det_eval", "wf_eval", "wf_kinetic", "coulomb", "swct")
    atol, rtol = get_tolerance_min(zones, level)
    assert np.any(np.abs(expected_hf - bare_hf) > atol + rtol * np.abs(expected_hf))
    assert np.any(np.abs(expected_pp - bare_pp) > atol + rtol * np.abs(expected_pp))
    for field, reference in (("force_HF", expected_hf), ("force_PP", expected_pp), ("E_L_force_PP", energy * expected_pp)):
        actual = np.asarray(getattr(driver, prefix + field))[0, 0]
        assert actual.shape == reference.shape
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False)
