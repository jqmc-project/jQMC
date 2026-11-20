"""Actual native GFMC_t / GFMC_n stress buffers against strain derivatives of the physical quantities.

One real native projection interval runs for two walkers of
``pbc_gamma_sphepgto_j12`` (committed ``lithium.h5``: periodic Li ccECP, spherical pGTO, Gamma SD,
exponential J1/J2) with seed 34356, a fixed identity mesh and ``alat`` 0.30; the conventional driver
uses ``tau`` 0.05 and the load-balanced driver two substeps with ``E_scf`` -1. A forwarding spy around
``jqmc_rs.build_obs_ctx`` records the native interval outputs before the driver consumes them; it
substitutes nothing. The returned electron positions are held fixed while the strain moves the
Hamiltonian and the electrons (``r @ F.T``) while the identity quadrature rotation stays fixed, so
every strained ECP quadrature point lies on the sphere of the strained electron-nucleus distance.

At each returned walker position:

1. native ``stress_HF`` is the strain derivative of the **production t/n measurement energy**
   (``jqmc_gfmc._compute_V_elements_t`` and the verbatim ``GFMC_n.run`` closure, the same entry
   points as the established forward-production binding test), not of a substituted Hamiltonian;
2. native ``stress_PP`` is the strain derivative of the primal ``ln|Psi|``;
3. the stored ``stress_HF``, ``stress_PP`` and ``E_L_stress_PP`` of that measured step equal the
   weighted reductions of those per-walker derivatives (weights ``w_L`` for GFMC_t and
   ``w_L / (V_diag - E_scf)`` for GFMC_n, normalized within the step).

Precision (#163): full correctness retains the original AD formulas at strict.
The full-only finite-difference node and its two stencil steps remain unchanged.
Mixed compares the actual recorded native per-walker and stored moment blocks
with independent full native measurement kernels at the same recorded coordinates.
The full shadow keeps the actual step's t/n weights, normalization and measured
energy fixed for the three stored reductions. This certifies deterministic
measurement-kernel-to-buffer fidelity, not agreement of two stochastic trajectories.
The interval's actual mixed-native buffers also retain their same-policy direct
kernel/reduction checks. Original all-entry medium liveness is retained in both
callbacks using the tuple captured before changing precision policies.

Every compared reference entry must be finite and distinguishable from zero at its tolerance. This is
deterministic one-interval evidence for the physical-kernel-to-buffer connection; it certifies no
force/stress estimator identity, statistical agreement, high-cond(G) range or Monte Carlo convergence.
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

from __future__ import annotations

from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc import jqmc_gfmc
from jqmc._precision import configure, get_tolerance_min, is_mixed_precision_enabled, precision_policy_snapshot
from jqmc._setting import ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT, get_tolerance_mixed_fidelity
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jqmc_mcmc import _NV_BASE_MESH, Nv_default
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block
from jqmc.wavefunction import evaluate_ln_wavefunction
from tests.test_jqmc_kernel.test_gfmc_forward_production_binding import _production_n
from tests.test_rs_kernel._rs_fixtures import rs_ham
from tests.test_rs_kernel.test_gfmc_chain_modes import _run_gfmc

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

#: Strain steps of the fourth-order stencil (the second is the step-doubling check).
_STEPS = (5.0e-4, 1.0e-3)
#: ``E_scf`` hard-wired for GFMC_n by ``_run_gfmc``.
_E_SCF = -1.0
#: ``alat`` hard-wired by ``_run_gfmc``.
_ALAT = 0.30
#: Zones crossed by the measurement energy and ln|Psi| of this periodic ECP J1/J2 fixture.
_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "mo_grad",
    "mo_lap",
    "jastrow_eval",
    "jastrow_grad_lap",
    "jastrow_ratio",
    "det_eval",
    "det_grad_lap",
    "det_ratio",
    "coulomb",
    "ewald",
    "wf_eval",
    "wf_kinetic",
    "wf_ratio",
    "local_energy",
)
_COMPONENTS = tuple((i, j) for i in range(3) for j in range(3))


class _ForwardingContext:
    """Forward every native call; record (never replace) projection-interval outputs."""

    def __init__(self, inner, captured):
        self._inner = inner
        self._captured = captured

    def __getattr__(self, name):
        attribute = getattr(self._inner, name)
        if not name.startswith("run_gfmc_projection_"):
            return attribute

        def invoke(*args, **kwargs):
            result = attribute(*args, **kwargs)
            assert kwargs["random_mesh"] is False
            self._captured.append({key: np.array(value, copy=True) for key, value in result.items()})
            return result

        return invoke


def _run_native_interval(driver, monkeypatch, physical_source=None):
    """One actual native interval; returns the driver and the recorded native outputs."""
    captured = []
    original = jqmc_rs.build_obs_ctx
    monkeypatch.setattr(jqmc_rs, "build_obs_ctx", lambda *a, **k: _ForwardingContext(original(*a, **k), captured))
    ham = rs_ham("pbc_gamma_sphepgto_j12") if physical_source is None else physical_source()
    run = _run_gfmc(
        driver,
        ham,
        seed=34356,
        mode="rust",
        num_steps=1,
        nsub=2,
        tau=0.05,
        comput_stress=True,
        random_mesh=False,
        num_walkers=2,
    )
    assert len(captured) == 1, f"expected one native projection interval, got {len(captured)}"
    return ham, run, captured[0]


def _forward(ham, driver):
    """Measurement energy and ln|Psi| at strain ``eps``: electrons carried along, quadrature fixed."""

    def evaluate(eps, r_up, r_dn):
        deformation = jnp.eye(3) + eps
        strained = ham._apply_strain(deformation)
        # The identity quadrature rotation is held fixed under the strain.
        up, dn, rotation = r_up @ deformation.T, r_dn @ deformation.T, jnp.eye(3)
        if driver == "t":
            energy = jqmc_gfmc._compute_V_elements_t(strained, up, dn, rotation, "tmove", _ALAT)[2]
        else:
            energy = _production_n()(strained, up, dn, rotation, rotation, "tmove", _ALAT, False)[2]
        ln_psi = evaluate_ln_wavefunction(strained.wavefunction_data, up, dn)
        return jnp.stack((jnp.real(energy), jnp.real(ln_psi)))

    return evaluate


def _strain_fd(evaluate, r_up, r_dn, step):
    """Fourth-order central difference of ``evaluate`` over all nine strain entries; shape (2, 3, 3)."""
    result = np.zeros((2, 3, 3))
    for i, j in _COMPONENTS:
        values = {n: np.asarray(evaluate(jnp.zeros((3, 3)).at[i, j].set(n * step), r_up, r_dn)) for n in (-2, -1, 1, 2)}
        result[:, i, j] = (values[-2] - 8.0 * values[-1] + 8.0 * values[1] - values[2]) / (12.0 * step)
    return result


def _weights(native, driver):
    """Normalized per-walker weights of the measured step."""
    weights = native["w_L"] if driver == "t" else native["w_L"] / (native["V_diag"] - _E_SCF)
    weights = np.asarray(weights, dtype=np.float64)
    assert np.all(np.isfinite(weights)) and np.all(weights > 0.0), weights
    return weights / np.sum(weights)


def _reductions(run, native, driver, per_walker):
    """(stored, expected) pairs for HF, PP and E_L*PP from per-walker (2, 3, 3) derivative stacks."""
    weights = _weights(native, driver)
    energy = np.asarray(native["e_L"], dtype=np.float64)
    hf, pp = per_walker[:, 0], per_walker[:, 1]
    return {
        "stress_HF": (np.asarray(run._stored_stress_HF)[0, 0], np.einsum("w,wij->ij", weights, hf)),
        "stress_PP": (np.asarray(run._stored_stress_PP)[0, 0], np.einsum("w,wij->ij", weights, pp)),
        "E_L_stress_PP": (np.asarray(run._stored_E_L_stress_PP)[0, 0], np.einsum("w,w,wij->ij", weights, energy, pp)),
    }


def _assert_nonvacuous(reference, atol, rtol, label):
    """Every reference entry is finite and distinguishable from zero at the comparison tolerance."""
    reference = np.asarray(reference)
    assert np.all(np.isfinite(reference)), f"{label}: non-finite reference {reference}"
    assert np.all(np.abs(reference) > atol + rtol * np.abs(reference)), f"{label}: vacuous component {reference}"


def _assert_fd_verdict(evaluate, run, native, driver):
    """Full precision: native and stored buffers against the strain finite difference."""
    atol, rtol = ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT
    per_walker = np.zeros((2, 2, 3, 3))
    for walker in range(2):
        r_up, r_dn = jnp.asarray(native["r_up"][walker]), jnp.asarray(native["r_dn"][walker])
        fd, fd_2h = (_strain_fd(evaluate, r_up, r_dn, step) for step in _STEPS)
        label = f"{driver}/walker{walker}"
        np.testing.assert_allclose(fd, fd_2h, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{label}/stencil")
        per_walker[walker] = fd
        for index, field in enumerate(("stress_HF", "stress_PP")):
            actual = np.asarray(native[field][walker], dtype=np.float64)
            reference = fd[index]
            _assert_nonvacuous(reference, atol, rtol, f"{label}/{field}")
            assert np.all(np.isfinite(actual)), f"{label}/{field}: non-finite native buffer {actual}"
            np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{label}/{field}")
    for field, (stored, expected) in _reductions(run, native, driver, per_walker).items():
        _assert_nonvacuous(expected, atol, rtol, f"{driver}/stored/{field}")
        assert np.all(np.isfinite(stored)), f"{driver}/stored/{field}: non-finite {stored}"
        np.testing.assert_allclose(stored, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{driver}/stored/{field}")


def _assert_ad_verdict(evaluate, run, native, driver):
    """Full strict AD correctness, retaining the original medium all-entry liveness."""
    live_atol, live_rtol = get_tolerance_min(_ZONES, "medium")
    atol, rtol = get_tolerance_min(_ZONES, "strict")
    jacobian = jax.jit(jax.jacrev(evaluate))
    per_walker = np.zeros((2, 2, 3, 3))
    for walker in range(2):
        r_up, r_dn = jnp.asarray(native["r_up"][walker]), jnp.asarray(native["r_dn"][walker])
        ad = np.asarray(jacobian(jnp.zeros((3, 3)), r_up, r_dn), dtype=np.float64)
        per_walker[walker] = ad
        label = f"{driver}/walker{walker}"
        for index, field in enumerate(("stress_HF", "stress_PP")):
            actual = np.asarray(native[field][walker], dtype=np.float64)
            reference = ad[index]
            _assert_nonvacuous(reference, live_atol, live_rtol, f"{label}/{field}/ad")
            assert np.all(np.isfinite(actual)), f"{label}/{field}: non-finite native buffer {actual}"
            np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{label}/{field}/ad")
    for field, (stored, expected) in _reductions(run, native, driver, per_walker).items():
        _assert_nonvacuous(expected, live_atol, live_rtol, f"{driver}/stored/{field}/ad")
        assert np.all(np.isfinite(stored)), f"{driver}/stored/{field}: non-finite {stored}"
        np.testing.assert_allclose(
            stored, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"{driver}/stored/{field}/ad"
        )


def _full_driver_physical_source():
    """Prepare one full physical recipe, preserving Python and native entry policies."""
    directory = tempfile.TemporaryDirectory(prefix="jqmc_gfmc_driver_stress_fidelity_")
    source = Path(directory.name) / "full_physical_source.h5"
    original_mode = "mixed" if is_mixed_precision_enabled() else "full"
    original_native = dict(jqmc_rs.native_precision_support())
    try:
        configure("full")
        jqmc_rs.configure_native_precision(precision_policy_snapshot())
        rs_ham("pbc_gamma_sphepgto_j12").save_to_hdf5(source)
    finally:
        configure(original_mode)
        jqmc_rs.configure_native_precision(original_native)

    def factory():
        # Keep the private directory alive for every independent policy reload.
        assert Path(directory.name).is_dir()
        return Hamiltonian_data.load_from_hdf5(source)

    return factory


def _fresh_native_stresses(ham, native):
    """The interval's native HF/Pulay measurement kernels at its recorded positions.

    lib.rs::gfmc_force_compute_real calls PbcStressHfArgs.compute_gfmc (fresh
    native LU) and PbcStressPpArgs.compute with the projection-carried inverse.
    The independent shadow uses the same HF kernel and a fresh native-owned
    sweep-seed matrix at the identical coordinates for its Pulay inverse.
    This certifies a same-configuration measurement, not trajectory equality.
    """
    wf = ham.wavefunction_data
    gem, jas, cp = wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data
    up, down = (np.ascontiguousarray(native[k], dtype=np.float64) for k in ("r_up", "r_dn"))
    assert up.shape[0] == down.shape[0] == 2
    ctx = jqmc_rs.build_obs_ctx(gem, jas, cp, False, True, True, want_force=False, precision_policy=precision_policy_snapshot())
    seed = np.asarray(
        ctx.sweep_seed_geminal_batch(up, down, np.asarray(ham.structure_data.cell), int(ham.structure_data._n_max))
    )
    inverse = np.linalg.pinv(seed)
    pulay = np.asarray(ctx.grad_ln_psi_eps_pbc_batch(up, down, inverse))
    mesh = _NV_BASE_MESH[Nv_default]
    hf = np.asarray(
        [
            jqmc_rs._gfmc_stress_ecp_pbc_withj(
                gem, jas, cp, up[w], down[w], _ALAT, np.eye(3), np.asarray(mesh.weights), np.asarray(mesh.grid_points), 1, False
            )
            for w in range(2)
        ]
    )
    assert hf.shape == pulay.shape == (2, 3, 3)
    return np.stack((hf, pulay), axis=1)


def _stress_quantity_blocks(run, native, driver, per_walker, *, measured):
    assert per_walker.shape == (2, 2, 3, 3)
    values = {
        f"walker{w}-{field}": np.asarray(per_walker[w, i])
        for w in range(2)
        for i, field in enumerate(("stress_HF", "stress_PP"))
    }
    for field, (stored, expected) in _reductions(run, native, driver, per_walker).items():
        values["stored-" + field] = stored if measured else expected
    return values


def assert_parameter_derivative_mixed_fidelity(mixed, full, atol, rtol):
    """The source-qualified central homogeneous-block comparison."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


@pytest.fixture(autouse=True)
def _release_stress_compilations():
    yield
    jax.clear_caches()


@pytest.mark.parametrize("driver", ["t", "n"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
@pytest.mark.skip_mixed_precision(
    reason="FD / numerical quadrature comparison dominated by float32 round-off",
    alternative_guarantee_ref="NUM-GFMC-DRIVER-STRESS-MIXED-FIDELITY-001",
)
def test_gfmc_driver_stress_buffers_match_strain_fd(driver, monkeypatch):
    """Native per-walker and stored GFMC stress buffers equal the strain finite difference (full only).

    See the module docstring for the fixture, the spy and the three compared quantities. Mixed
    precision is judged by ``test_gfmc_driver_stress_buffers_match_same_precision_ad`` on the
    identical fixture and buffers (NUM-GFMC-DRIVER-STRESS-MIXED-FIDELITY-001).
    """
    ham, run, native = _run_native_interval(driver, monkeypatch)
    _assert_fd_verdict(jax.jit(_forward(ham, driver)), run, native, driver)


@pytest.mark.parametrize("driver", ["t", "n"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_gfmc_driver_stress_buffers_match_same_precision_ad(driver, monkeypatch):
    """Full buffers equal strict strain AD; mixed retains independent full measurement fidelity."""
    if is_mixed_precision_enabled():
        live_atol, live_rtol = get_tolerance_min(_ZONES, "medium")
        factory = _full_driver_physical_source()
        _ham, run, native = _run_native_interval(driver, monkeypatch, physical_source=factory)
        measured = np.stack((np.asarray(native["stress_HF"]), np.asarray(native["stress_PP"])), axis=1)
        assert measured.shape == (2, 2, 3, 3)
        # Retain the native-buffer -> stored-moment connection with the actual
        # t/n weights, their normalization, and the measured step energy.
        for field, (stored, expected) in _reductions(run, native, driver, measured).items():
            _assert_nonvacuous(expected, live_atol, live_rtol, f"{driver}/{field}/reduction")
            assert stored.shape == expected.shape == (3, 3)
            assert np.all(np.isfinite(stored)), (field, stored)
            np.testing.assert_allclose(stored, expected, atol=live_atol, rtol=live_rtol, equal_nan=False)

        def analytical(fresh):
            per_walker = _fresh_native_stresses(fresh, native)
            if is_mixed_precision_enabled():
                # Same-policy measured route remains a separate support check.
                np.testing.assert_allclose(measured, per_walker, atol=live_atol, rtol=live_rtol, equal_nan=False)
                per_walker = measured
            values = _stress_quantity_blocks(run, native, driver, per_walker, measured=is_mixed_precision_enabled())
            expected_fields = {f"walker{w}-{field}" for w in range(2) for field in ("stress_HF", "stress_PP")}
            expected_fields.update("stored-" + field for field in ("stress_HF", "stress_PP", "E_L_stress_PP"))
            assert set(values) == expected_fields, values.keys()
            for field, value in values.items():
                assert value.shape == (3, 3), (field, value.shape)
                assert np.all(np.isfinite(value)), (field, value)
                _assert_nonvacuous(value, live_atol, live_rtol, field)
            return values

        mixed, full = mixed_vs_full(
            factory,
            analytical,
            configure_native=jqmc_rs.configure_native_precision,
            native_policy_snapshot=jqmc_rs.native_precision_support,
        )
        blocks = mixed_fidelity_blocks(mixed, full)
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        for quantity, (mixed_block, full_block) in zip(sorted(mixed), blocks, strict=True):
            assert_parameter_derivative_mixed_fidelity(mixed_block, full_block, atol, rtol)
        mutated = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert_parameter_derivative_mixed_fidelity(mutated, blocks[0][1], atol, rtol)
        return

    ham, run, native = _run_native_interval(driver, monkeypatch)
    _assert_ad_verdict(_forward(ham, driver), run, native, driver)
