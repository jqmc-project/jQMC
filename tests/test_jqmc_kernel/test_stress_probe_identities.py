"""Deterministic identity probes for the periodic stress construction.

Four exact identities, each checked on the production kernels and each paired
with a production mutation that must turn it red:

* Ewald split independence.  Each Ewald channel (ii, ei, ee) carries its
  own uniform-background term, so its energy and its analytic strain tensor
  (``jqmc.derivatives.ewald``) do not depend on the convergence parameter
  ``eta``.  Rebuilding the tables with ``eta_override = 0.8 eta`` and
  ``1.25 eta`` must leave both unchanged.
* Production-route identities of a pure plane-wave determinant.  The
  analytic HF stress has no antisymmetric part (rotation invariance of
  ``e_L``) and equals the ``_auto`` strain derivative; the Pulay stress is zero
  on both routes (a plane-wave value is affine-strain invariant).
* Composite pGTO + plane-wave joint scaling.  Geometry ``F = s I``,
  electrons ``r -> s r``, pGTO exponents ``Z -> Z / s^2`` and contraction
  coefficients ``c -> s^{3/2} c`` leave ``Psi`` pointwise invariant, hence
  ``d ln|Psi|/ds = 0``, ``dT/ds = -2 T`` and ``dV/ds = -V``.  Checked at finite
  ``s`` (stage A), by AD through the joint path (stage B) and reassembled from
  the production analytic strain and basis-parameter blocks (stage C).
* Estimator covariance identity.  For fixed configurations with a
  J2, the production ``MCMC.get_aT`` assembled from production per-walker
  ``e_L``, ``stress_HF``, ``stress_PP`` equals the delete-one jackknife of
  ``jax.jacrev`` over ``eps`` of the reweighted empirical energy
  ``E(eps) = sum_i w_i(eps) e_L,eps(F x_i) / sum_i w_i(eps)``,
  ``w_i = w_L,i |Psi_eps(F x_i)|^2 / |Psi_0(x_i)|^2``.

Every mutation is the production source of one function with one exact edit,
compiled in that function's module namespace and patched in for the duration
of the check.
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

import __future__

import contextlib
import inspect
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mpi4py import MPI

import jqmc.atomic_orbital as atomic_orbital_module
import jqmc.coulomb_potential as coulomb_module
import jqmc.determinant as determinant_module
from jqmc import derivatives
from jqmc._precision import get_tolerance_cancelling_mean, get_tolerance_min
from jqmc.coulomb_potential import build_ewald_data
from jqmc.derivatives import basis, pbc_stress
from jqmc.derivatives import ewald as ewald_stress
from jqmc.hamiltonians import compute_local_energy
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.wavefunction import compute_kinetic_energy, evaluate_ln_wavefunction
from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel.test_stress_homogeneity import _CELL, _KINETIC_ZONES, _VIRIAL_ZONES, build_hamiltonian

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.without_chdir]

_EWALD_ZONES = ("ewald",)
_ETA_FACTORS = (0.8, 1.25)
_CHANNELS = ("ii", "ei", "ee")
#: Extra fixed electron configurations (fractional; one up, one down) of the triclinic fixture.
_CONFIG_FRAC = (
    (np.array([[0.31, 0.72, 0.44]]), np.array([[0.83, 0.15, 0.66]])),
    (np.array([[0.93, 0.07, 0.51]]), np.array([[0.12, 0.88, 0.03]])),
    (np.array([[0.47, 0.36, 0.97]]), np.array([[0.64, 0.52, 0.21]])),
)
_SAVE_DIR = qe_example_files_root() / "H-ae_gamma" / "out" / "h_gamma.save"
#: Fixed asymmetric configuration of the H-ae_gamma fixture (fractional; 4 up then 4 down).
_QE_FRAC = np.array(
    [
        [0.7960744715502008, 0.5879050127212068, 0.27740160528971547],
        [0.8163116219027731, 0.7234488221080789, 0.5142639117232316],
        [0.5607361487761403, 0.7772417387647526, 0.29839660595526374],
        [0.6424449204613618, 0.6112421804501047, 0.3068440599962984],
        [0.6790496959951810, 0.21575010128009267, 0.5384351134176435],
        [0.5335569436581369, 0.48107310556332483, 0.3279662760355768],
        [0.8475973064157509, 0.37086014004899115, 0.7804912127242569],
        [0.30041553538379073, 0.49690527555746056, 0.4217985709480464],
    ]
)
_JOINT_SCALES = (1.02, 0.97)
_BASIS_MASK = {"opt_lambda_basis_exp": True, "opt_lambda_basis_coeff": True}
#: Covariance identity: exponential J2 parameter, 3 steps x 2 walkers of fixed configurations and their weights.
_J2_PARAM = 1.1
_ESTIMATOR_SEED = 1123
_ESTIMATOR_STEPS, _ESTIMATOR_WALKERS = 3, 2
_ESTIMATOR_WEIGHTS = np.array([[1.0, 0.8], [1.3, 0.9], [1.1, 1.2]])


# --------------------------------------------------------------------------- helpers
def _mutated(function, old, new, overrides=None, owner=None):
    """``function`` recompiled from its production source with ``old`` replaced by ``new`` once."""
    source = textwrap.dedent(inspect.getsource(function))
    assert source.count(old) == 1, f"mutation anchor not unique in {function.__qualname__}: {old!r}"
    source = source.replace(old, new)
    if owner is not None:
        source = source.replace("self.__", f"self._{owner}__")
    namespace = dict(function.__globals__)
    namespace.update(overrides or {})
    flags = __future__.annotations.compiler_flag
    exec(compile(source, inspect.getsourcefile(function), "exec", flags=flags, dont_inherit=True), namespace)
    return namespace[function.__name__]


def _assert_close(actual, expected, label, atol, rtol):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), label
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)


def _assert_nonvacuous(value, label, atol, rtol):
    value = np.asarray(value)
    assert np.all(np.isfinite(value)), label
    assert np.max(np.abs(value)) > atol + rtol * np.max(np.abs(value)), f"{label}: vacuous"


def _clear_jit_caches():
    """Drop the cached jitted analytic kernels so a patched module global is retraced."""
    derivatives._jitted_assemble.cache_clear()
    jax.clear_caches()


@contextlib.contextmanager
def _production_mutation(monkeypatch, owner, name, replacement, traced=True):
    """Patch ``owner.name`` with a mutated production function.

    ``traced``: the patched global is read inside cached jitted kernels, so the
    jit caches are dropped on entry and on exit (also when a check fails).
    Eager callees (``traced=False``) need no cache reset.
    """
    if traced:
        _clear_jit_caches()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner, name, replacement)
            yield
    finally:
        if traced:
            _clear_jit_caches()


# --------------------------------------------------------------------------- Ewald eta invariance
def _ewald_channels(potential, ewald, r_up, r_dn):
    """Energy and analytic strain tensor of each Ewald channel, through module attributes (patchable)."""
    ei_up, ei_dn = coulomb_module.compute_ewald_potential_ei(potential, r_up, r_dn, ewald)
    energy = {
        "ii": coulomb_module.compute_ewald_potential_ii(potential, ewald),
        "ei": jnp.sum(ei_up) + jnp.sum(ei_dn),
        "ee": coulomb_module.compute_ewald_potential_ee(r_up, r_dn, ewald),
    }
    strain = {
        "ii": ewald_stress.grad_Eii_wrt_strain(potential, ewald),
        "ei": ewald_stress.grad_Eei_wrt_strain(potential, ewald, r_up, r_dn),
        "ee": ewald_stress.grad_Eee_wrt_strain(ewald, r_up, r_dn),
    }
    return {key: float(value) for key, value in energy.items()}, {key: np.asarray(value) for key, value in strain.items()}


def _eta_sweep(ham, r_up, r_dn):
    """Channel energies/tensors at the production eta and at each rebuilt ``eta_override``."""
    potential = ham.coulomb_potential_data
    base = _ewald_channels(potential, potential.ewald_data, r_up, r_dn)
    rebuilt = {}
    for factor in _ETA_FACTORS:
        ewald = build_ewald_data(
            ham.structure_data,
            potential,
            eta_override=potential.ewald_data.eta * factor,
            n_electron=r_up.shape[0] + r_dn.shape[0],
        )
        rebuilt[factor] = _ewald_channels(potential.replace(ewald_data=ewald), ewald, r_up, r_dn)
    return base, rebuilt


def _check_eta_invariance(base, rebuilt, channels=_CHANNELS, parts=("energy", "strain")):
    atol, rtol = get_tolerance_min(_EWALD_ZONES, "strict")
    for factor, (energy, strain) in rebuilt.items():
        for key in channels:
            if "energy" in parts:
                _assert_close(energy[key], base[0][key], f"eta*{factor}/energy/{key}", atol, rtol)
            if "strain" in parts:
                _assert_close(strain[key], base[1][key], f"eta*{factor}/strain/{key}", atol, rtol)


def _background_is_nonvacuous(ham):
    """Each channel's background term changes by more than the tolerance across the eta sweep."""
    atol, rtol = get_tolerance_min(_EWALD_ZONES, "strict")
    potential = ham.coulomb_potential_data
    volume = abs(float(np.linalg.det(np.asarray(ham.structure_data.cell))))
    charge = float(np.sum(np.asarray(potential._effective_charges)))
    eta = potential.ewald_data.eta
    for factor in _ETA_FACTORS:
        shift = abs(np.pi / (2.0 * volume) * charge**2 * (1.0 / (factor * eta) ** 2 - 1.0 / eta**2))
        assert shift > atol + rtol * shift, f"eta*{factor}: background shift {shift} is vacuous"


@pytest.mark.light
def test_ewald_channels_are_eta_independent():
    """Every Ewald channel energy and analytic strain tensor is independent of eta.

    Fixture: the triclinic two-H, one-up/one-down plane-wave system of
    ``test_stress_homogeneity`` (net ion charge 2 and 2 electrons, so every
    channel's own background term is non-zero and changes with eta).  The
    Ewald tables are rebuilt by ``build_ewald_data(eta_override = f * eta)``
    for f in {0.8, 1.25} (cutoffs re-derived from the default tolerance).  The
    ii, ei, ee energies of the production evaluators and the full 3x3 tensors
    of ``grad_Eii/Eei/Eee_wrt_strain`` must equal the unmodified values at the
    ewald zone strict tolerance (fp64 in both modes).  Full and mixed are
    each judged in their own mode.
    """
    ham, r_up, r_dn = build_hamiltonian()
    _background_is_nonvacuous(ham)
    base, rebuilt = _eta_sweep(ham, r_up, r_dn)
    atol, rtol = get_tolerance_min(_EWALD_ZONES, "strict")
    for key in _CHANNELS:
        _assert_nonvacuous(base[1][key], f"strain/{key}", atol, rtol)
    _check_eta_invariance(base, rebuilt)


@pytest.mark.light
def test_ewald_eta_independence_rejects_dropped_background(monkeypatch):
    """Ewald eta negative controls: dropping one channel's background term makes it eta dependent.

    Same fixture.  Production mutations, one at a time: (a) ``s_bg`` removed
    from the return of ``grad_Eii_wrt_strain`` / ``grad_Eei_wrt_strain`` /
    ``grad_Eee_wrt_strain`` -- that channel's strain check must fail; (b) the
    background term removed from the ee energy (``compute_ewald_potential_ee``)
    and from the ei per-electron energy (``compute_ewald_potential_ei``) --
    that channel's energy check must fail.
    """
    ham, r_up, r_dn = build_hamiltonian()
    strain_mutations = {
        "ii": ("grad_Eii_wrt_strain", "return s_real + s_recip + s_bg", "return s_real + s_recip"),
        "ei": ("grad_Eei_wrt_strain", "return s_real + s_recip + s_bg", "return s_real + s_recip"),
        "ee": ("grad_Eee_wrt_strain", "return s_real + s_recip + s_bg", "return s_real + s_recip"),
    }
    for key, (name, old, new) in strain_mutations.items():
        replacement = _mutated(getattr(ewald_stress, name), old, new)
        with _production_mutation(monkeypatch, ewald_stress, name, replacement, traced=False):
            base, rebuilt = _eta_sweep(ham, r_up, r_dn)
            with pytest.raises(AssertionError, match=rf"(?s)Not equal to tolerance.*strain/{key}"):
                _check_eta_invariance(base, rebuilt, channels=(key,), parts=("strain",))
    energy_mutations = {
        "ee": (
            "compute_ewald_potential_ee",
            "return E_real + E_recip + E_self + E_background",
            "return E_real + E_recip + E_self",
        ),
        "ei": (
            "compute_ewald_potential_ei",
            "return E_real_per_e + E_recip_per_e + ei_background_per_e",
            "return E_real_per_e + E_recip_per_e",
        ),
    }
    for key, (name, old, new) in energy_mutations.items():
        original = getattr(coulomb_module, name)
        replacement = _mutated(getattr(original, "__wrapped__", original), old, new)
        with _production_mutation(monkeypatch, coulomb_module, name, replacement, traced=False):
            base, rebuilt = _eta_sweep(ham, r_up, r_dn)
            with pytest.raises(AssertionError, match=rf"(?s)Not equal to tolerance.*energy/{key}"):
                _check_eta_invariance(base, rebuilt, channels=(key,), parts=("energy",))


# --------------------------------------------------------------------------- production route, pure PW
def _configurations():
    for up, dn in _CONFIG_FRAC:
        yield jnp.asarray(up @ _CELL), jnp.asarray(dn @ _CELL)


def _route_stresses(ham, r_up, r_dn):
    """Production HF/Pulay stresses plus the two cancelling halves of the plane-wave Pulay stress.

    ``virial`` is the electron-coordinate half ``sum_e (d ln|Psi|/d r_e)_c r_e,d``
    from the production analytic drift; ``cell`` is ``jax.jacrev`` over eps of
    ln|Psi| through the production ``Wavefunction_data._apply_strain`` with the
    electrons held fixed.  Their sum is the Pulay stress of the production strain
    map, and plane-wave affine invariance makes them cancel.
    """
    RT = jnp.eye(3)
    flags = {"opt_eps": True}
    drift_up, drift_dn = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, {"opt_r": True}, analytic=True)["r"]
    drift = np.real(np.concatenate([np.asarray(drift_up), np.asarray(drift_dn)]))
    positions = np.concatenate([np.asarray(r_up), np.asarray(r_dn)])

    def ln_psi_cell(eps):
        strained = ham.wavefunction_data._apply_strain(jnp.eye(3) + eps)
        return jnp.real(evaluate_ln_wavefunction(strained, r_up, r_dn))

    return {
        "hf": np.real(np.asarray(derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=True)["eps"])),
        "hf_auto": np.real(np.asarray(derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=False)["eps"])),
        "pp": np.real(np.asarray(derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)["eps"])),
        "virial": np.einsum("ec,ed->cd", drift, positions),
        "cell": np.asarray(jax.jacrev(ln_psi_cell)(jnp.zeros((3, 3)))),
    }


def _check_route_identities(ham, index, stresses, checks=("antisym", "auto", "pulay", "pulay_auto")):
    atol, rtol = get_tolerance_min(_VIRIAL_ZONES, "strict")
    hf, virial = stresses["hf"], stresses["virial"]
    _assert_nonvacuous(hf, f"{index}/hf", atol, rtol)
    _assert_nonvacuous(virial, f"{index}/pulay-virial", atol, rtol)
    p_atol, p_rtol = get_tolerance_cancelling_mean(atol, rtol, np.max(np.abs(virial)))
    if "antisym" in checks:
        _assert_close(hf.T, hf, f"{index}/antisym", atol, rtol)
    if "auto" in checks:
        _assert_close(hf, stresses["hf_auto"], f"{index}/hf-vs-auto", atol, rtol)
    if "pulay" in checks:
        _assert_close(stresses["pp"] - stresses["cell"], virial, f"{index}/pulay-analytic", p_atol, p_rtol)
    if "pulay_auto" in checks:
        _assert_close(-stresses["cell"], virial, f"{index}/pulay-strain-map", p_atol, p_rtol)


@pytest.mark.light
def test_production_stress_route_identities_pure_plane_wave():
    """Pure-PW route identities: rotation invariance, _auto parity and zero Pulay stress of the production routes.

    Fixture: the triclinic Jastrow-free plane-wave SD of
    ``test_stress_homogeneity`` at three fixed electron configurations.  Per
    configuration the production analytic HF stress
    (``derivatives.compute_grad_e_L(..., {"opt_eps": True}, analytic=True)``)
    equals its transpose (no antisymmetric part: e_L is rotation invariant) and
    equals the ``_auto`` jacrev strain derivative through
    ``Hamiltonian_data._apply_strain``.  The Pulay stress vanishes because a
    plane-wave value is invariant under the affine strain: the cell response
    of ln|Psi| through ``Wavefunction_data._apply_strain`` (electrons fixed)
    equals minus the electron virial built from the production analytic drift,
    and the production analytic Pulay stress
    (``compute_grad_ln_Psi(..., {"opt_eps": True}, analytic=True)``) equals the
    sum of the two.  Strict zone tolerance on the local-energy path (the Pulay
    cancellation rebased on the virial scale); full and mixed are each judged
    in their own mode.
    """
    ham, _, _ = build_hamiltonian()
    for index, (r_up, r_dn) in enumerate(_configurations()):
        _check_route_identities(ham, index, _route_stresses(ham, r_up, r_dn))


@pytest.mark.light
def test_production_stress_route_identities_reject_mutations(monkeypatch):
    """Pure-PW route negative controls: production mutations must turn the identities red.

    Same fixture, first configuration.  (a) The electron-ion real-space virial
    in ``grad_Eei_wrt_strain`` contracts its second leg with the min-image
    displacement ``r_iI`` instead of the lattice-imaged ``w = r_iI + T_n``:
    the HF stress acquires an antisymmetric part, so the antisymmetry and the
    ``_auto`` parity checks fail.  (b) ``atomic_orbital._apply_strain_to_aos``
    (as imported by ``determinant``) leaves the plane-wave leaf unstrained, so
    the cell response no longer cancels the electron virial: the strain-map
    Pulay check and the ``_auto`` HF parity check fail.
    """
    ham, _, _ = build_hamiltonian()
    r_up, r_dn = next(_configurations())
    one_leg = _mutated(
        ewald_stress.grad_Eei_wrt_strain,
        's_real = -jnp.einsum("I,iIn,iInc,iInd->cd", Z, gp / dd, w, w)',
        's_real = -jnp.einsum("I,iIn,iInc,iInd->cd", Z, gp / dd, w, jnp.broadcast_to(r_iI[:, :, None, :], w.shape))',
    )
    with _production_mutation(monkeypatch, ewald_stress, "grad_Eei_wrt_strain", one_leg):
        stresses = _route_stresses(ham, r_up, r_dn)
        for check in ("antisym", "auto"):
            with pytest.raises(AssertionError, match=r"(?s)Not equal to tolerance.*ei-leg/"):
                _check_route_identities(ham, "ei-leg", stresses, checks=(check,))
    with _production_mutation(monkeypatch, determinant_module, "_apply_strain_to_aos", _unstrained_leaf("AOs_plane_wave_data")):
        stresses = _route_stresses(ham, r_up, r_dn)
        for check in ("pulay_auto", "auto"):
            with pytest.raises(AssertionError, match=r"(?s)Not equal to tolerance.*pw-unstrained/"):
                _check_route_identities(ham, "pw-unstrained", stresses, checks=(check,))


def _unstrained_leaf(leaf_class):
    """``_apply_strain_to_aos`` mutated to return every ``leaf_class`` leaf unstrained."""
    return _mutated(
        atomic_orbital_module._apply_strain_to_aos,
        "return aos.replace(structure_data=_strain_structure_data(aos.structure_data, F))",
        f"return aos if isinstance(aos, {leaf_class}) else "
        "aos.replace(structure_data=_strain_structure_data(aos.structure_data, F))",
    )


# --------------------------------------------------------------------------- composite pGTO + PW
_JOINT_ZONES = _VIRIAL_ZONES + ("wf_eval",)


def _composite_hamiltonian():
    """The branch-probe composite: H-ae_gamma (8 H, Gamma, AE) with one s pGTO shell per atom plus PW."""
    from jqmc.pw_pgto_projection import project_pw_to_mixed
    from jqmc.qe_to_jqmc import build_hamiltonian_from_projection
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import PrimitiveShell

    reference = load_qe_reference(str(_SAVE_DIR))
    g_max = float(np.sqrt(2.0 * reference.ecutwfc))
    shell = PrimitiveShell(angular_l=0, exponents=(0.8, 2.4), coefficients=(0.7, 0.4), rel_error=0.0, overlap_cond=1.0)
    projection = project_pw_to_mixed(
        reference, [[shell] for _ in reference.atom_labels], g_cut=0.25 * g_max, drop_diffuse=False
    )
    ham = build_hamiltonian_from_projection(
        reference, projection, kind="pgto+pw", j1_param=None, j2_param=None, ewald_tol=1.0e-10
    )
    aos = ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data
    kinds = [type(child).__name__ for child in aos.children]
    assert kinds == ["AOs_cart_pgto_data", "AOs_plane_wave_data"], kinds
    n_up = int(ham.wavefunction_data.geminal_data.num_electron_up)
    r_all = jnp.asarray(_QE_FRAC @ np.asarray(ham.structure_data.cell))
    return ham, r_all[:n_up], r_all[n_up:]


def _scale_radial(aos, exponent_factor, coefficient_factor):
    children = []
    for child in aos.children:
        if hasattr(child, "radial"):
            radial = child.radial.replace(
                exponents=jnp.asarray(child.radial.exponents) * exponent_factor,
                coefficients=jnp.asarray(child.radial.coefficients) * coefficient_factor,
            )
            child = child.replace(radial=radial)
        children.append(child)
    return aos.replace(children=tuple(children))


def _joint_observables(ham, r_up, r_dn, s_geometry, s_basis):
    """(ln|Psi|, T, V) with geometry/electrons scaled by ``s_geometry`` and the pGTO radial set by ``s_basis``."""
    strained = ham._apply_strain(jnp.eye(3) * s_geometry)
    geminal = strained.wavefunction_data.geminal_data
    geminal = geminal.replace(
        **{
            side: getattr(geminal, side).replace(
                aos_data=_scale_radial(getattr(geminal, side).aos_data, s_basis**-2, s_basis**1.5)
            )
            for side in ("orb_data_up_spin", "orb_data_dn_spin")
        }
    )
    wavefunction = strained.wavefunction_data.replace(geminal_data=geminal)
    up, dn = r_up * s_geometry, r_dn * s_geometry
    potential = strained.coulomb_potential_data
    return jnp.stack(
        [
            jnp.real(evaluate_ln_wavefunction(wavefunction, up, dn)),
            jnp.real(compute_kinetic_energy(wavefunction, up, dn)),
            jnp.real(coulomb_module.compute_coulomb_potential(potential, up, dn, jnp.eye(3), wavefunction_data=wavefunction)),
        ]
    )


def _basis_chain(ham, grad_wavefunction):
    """``-2 sum(Z dX/dZ) + 1.5 sum(c dX/dc)`` over both spin channels' pGTO radial leaves."""
    total = 0.0
    for side in ("orb_data_up_spin", "orb_data_dn_spin"):
        grads = getattr(grad_wavefunction.geminal_data, side).aos_data.children
        values = getattr(ham.wavefunction_data.geminal_data, side).aos_data.children
        for grad, value in zip(grads, values):
            if hasattr(value, "radial"):
                total += -2.0 * np.sum(np.asarray(value.radial.exponents) * np.real(np.asarray(grad.radial.exponents)))
                total += 1.5 * np.sum(np.asarray(value.radial.coefficients) * np.real(np.asarray(grad.radial.coefficients)))
    return total


def _joint_scaling_stages(ham, r_up, r_dn):
    one = jnp.asarray(1.0)
    base = np.asarray(_joint_observables(ham, r_up, r_dn, one, one))
    finite = {s: np.asarray(_joint_observables(ham, r_up, r_dn, jnp.asarray(s), jnp.asarray(s))) for s in _JOINT_SCALES}
    partial_geometry, partial_basis = (
        np.asarray(p) for p in jax.jacrev(_joint_observables, argnums=(3, 4))(ham, r_up, r_dn, one, one)
    )
    RT = jnp.eye(3)
    flags = {"opt_eps": True, **_BASIS_MASK}
    grad_e = derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=True)
    grad_ln = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)
    return {
        "base": base,
        "finite": finite,
        "partial_geometry": partial_geometry,
        "partial_basis": partial_basis,
        "trace_e": float(np.trace(np.real(np.asarray(grad_e["eps"])))),
        "chain_e": _basis_chain(ham, grad_e["c"]),
        "trace_ln": float(np.trace(np.real(np.asarray(grad_ln["eps"])))),
        "chain_ln": _basis_chain(ham, grad_ln["c"]),
    }


def _check_joint_scaling(stages, checks=("A", "B", "C")):
    atol, rtol = get_tolerance_min(_JOINT_ZONES, "strict")
    base = stages["base"]
    lnpsi, kinetic, potential = base
    _assert_nonvacuous(base, "base", atol, rtol)
    if "A" in checks:
        for s, value in stages["finite"].items():
            _assert_close(value[0], lnpsi, f"A/s={s}/lnPsi", atol, rtol)
            _assert_close(s**2 * value[1], kinetic, f"A/s={s}/T", atol, rtol)
            _assert_close(s * value[2], potential, f"A/s={s}/V", atol, rtol)
    if "B" in checks:
        geometry, basis_part = stages["partial_geometry"], stages["partial_basis"]
        _assert_nonvacuous(basis_part[:2], "B/basis-partial", atol, rtol)
        # geometry partial == (closed-form total) - (basis partial); ln|Psi| has total 0.
        for name, actual, reference, terms in (
            ("lnPsi", geometry[0], -basis_part[0], (geometry[0], basis_part[0])),
            ("T", geometry[1], -2.0 * kinetic - basis_part[1], (geometry[1], basis_part[1], 2.0 * kinetic)),
            ("V", geometry[2], -potential - basis_part[2], (geometry[2], basis_part[2], potential)),
        ):
            b_atol, b_rtol = get_tolerance_cancelling_mean(atol, rtol, np.max(np.abs(terms)))
            _assert_close(actual, reference, f"B/{name}", b_atol, b_rtol)
    if "C" in checks:
        # production strain trace == (closed-form total) - (production basis chain); ln|Psi| has total 0.
        trace_e, chain_e, trace_ln, chain_ln = (stages[key] for key in ("trace_e", "chain_e", "trace_ln", "chain_ln"))
        for name, actual, reference, terms in (
            ("e_L", trace_e, -2.0 * kinetic - potential - chain_e, (trace_e, chain_e, 2.0 * kinetic + potential)),
            ("lnPsi", trace_ln, -chain_ln, (trace_ln, chain_ln)),
        ):
            _assert_nonvacuous(terms[1], f"C/{name}/basis-chain", atol, rtol)
            c_atol, c_rtol = get_tolerance_cancelling_mean(atol, rtol, np.max(np.abs(terms)))
            _assert_close(actual, reference, f"C/{name}", c_atol, c_rtol)


@pytest.mark.heavy
def test_composite_pgto_plane_wave_joint_scaling():
    """Composite pGTO + PW joint-scaling oracle, forward, AD and production analytic blocks.

    Fixture: tests/qe_example_files/H-ae_gamma (8 all-electron H, cubic 9.5
    bohr cell, Gamma, 4 up + 4 down) projected by ``project_pw_to_mixed`` onto
    one s pGTO shell per atom (exponents 0.8, 2.4; Cartesian periodic leaf,
    16 AOs) plus the plane waves below a quarter of the wavefunction cutoff,
    no Jastrow, one fixed asymmetric configuration.  The joint path (geometry
    ``F = s I`` through ``_apply_strain``, electrons ``s r``, pGTO exponents
    ``Z / s^2``, coefficients ``s^{3/2} c``; MO and PW data fixed) leaves Psi
    invariant.  Stage A: at s in {1.02, 0.97} ln|Psi|, s^2 T and s V equal
    their s = 1 values.  Stage B: jax.jacrev through the joint path (geometry
    and basis partials separately) sums to 0, -2 T and -V.  Stage C: the
    production analytic strain trace plus the production analytic basis
    chain ``-2 sum(Z dX/dZ) + 1.5 sum(c dX/dc)`` (``opt_lambda_basis_exp`` /
    ``opt_lambda_basis_coeff``) equals -2 T - V for e_L and 0 for ln|Psi|.
    Stages B and C are cancellations, so their tolerance is rebased onto the
    largest cancelling term (``get_tolerance_cancelling_mean``).  Strict zone
    policy; full and mixed are each judged in their own mode.
    """
    ham, r_up, r_dn = _composite_hamiltonian()
    _check_joint_scaling(_joint_scaling_stages(ham, r_up, r_dn))


@pytest.mark.heavy
def test_composite_joint_scaling_rejects_pgto_mutations(monkeypatch):
    """Joint-scaling negative controls: pGTO strain, image-T and normalization-derivative mutations.

    Same fixture.  Production mutations, one at a time: (a)
    ``_apply_strain_to_aos`` (as imported by ``determinant``) returns the
    periodic pGTO leaf unstrained -- stages A and B must fail; (b) the
    kernel-rank image-T correction is dropped from the pGTO Pulay strain
    contraction (``pbc_stress._pgto_stress_contrib`` returns ``term1``) --
    stage C must fail; (c) the same drop in the kinetic strain field
    contraction (``pbc_stress._field_stress``) -- stage C must fail; (d) the
    normalization derivative ``a = (2l + 3) / (4 Z)`` of the primitive
    exponent sensitivity is zeroed (``basis._prim_sens``) -- stage C must fail.
    """
    ham, r_up, r_dn = _composite_hamiltonian()
    mutations = {
        "pgto-unstrained": (determinant_module, "_apply_strain_to_aos", _unstrained_leaf("AOs_pGTO_base_data"), ("A", "B")),
        "pulay-image-T": (
            pbc_stress,
            "_pgto_stress_contrib",
            _mutated(pbc_stress._pgto_stress_contrib, "return term1 + term2", "return term1"),
            ("C",),
        ),
        "kinetic-image-T": (
            pbc_stress,
            "_field_stress",
            _mutated(pbc_stress._field_stress, "return term1 + term2", "return term1"),
            ("C",),
        ),
        "normalization": (
            basis,
            "_prim_sens",
            _mutated(
                basis._prim_sens,
                "a = ((2.0 * l + 3.0) / (4.0 * Z))[:, None]",
                "a = 0.0 * ((2.0 * l + 3.0) / (4.0 * Z))[:, None]",
            ),
            ("C",),
        ),
    }
    for label, (module, name, replacement, stages) in mutations.items():
        with _production_mutation(monkeypatch, module, name, replacement):
            observed = _joint_scaling_stages(ham, r_up, r_dn)
            for stage in stages:
                with pytest.raises(AssertionError, match=rf"(?s)Not equal to tolerance.*{stage}/"):
                    _check_joint_scaling(observed, checks=(stage,))


# --------------------------------------------------------------------------- estimator covariance identity
#: Zones on the e_L / ln|Psi| / stress path of the J2 plane-wave fixture.
_ESTIMATOR_ZONES = _KINETIC_ZONES + ("jastrow_eval", "jastrow_grad_lap", "wf_eval", "coulomb", "ewald", "local_energy")


def _estimator_tolerance():
    return get_tolerance_min(_ESTIMATOR_ZONES, "strict")


def _j2_hamiltonian():
    """The triclinic plane-wave fixture with an exponential J2, so the Pulay stress is non-zero."""
    ham, _, _ = build_hamiltonian()
    jastrow = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=_J2_PARAM, jastrow_2b_type="exp", structure_data=ham.structure_data
        ),
        jastrow_three_body_data=None,
        jastrow_dtn_data=None,
    )
    return ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jastrow))


def _estimator_configurations():
    """(steps, walkers) fixed configurations: one up and one down electron each, Cartesian."""
    rng = np.random.default_rng(_ESTIMATOR_SEED)
    frac = rng.uniform(size=(_ESTIMATOR_STEPS, _ESTIMATOR_WALKERS, 2, 3))
    return np.einsum("swek,kc->swec", frac, _CELL)


def _production_moments(ham, configs):
    """Production e_L, analytic stress_HF and stress_PP at every fixed configuration."""
    RT = jnp.eye(3)
    flags = {"opt_eps": True}
    energy = np.zeros(configs.shape[:2])
    hf = np.zeros(configs.shape[:2] + (3, 3))
    pp = np.zeros(configs.shape[:2] + (3, 3))
    for step, walker in np.ndindex(*configs.shape[:2]):
        r_up, r_dn = jnp.asarray(configs[step, walker, :1]), jnp.asarray(configs[step, walker, 1:])
        energy[step, walker] = float(np.real(compute_local_energy(ham, r_up, r_dn, RT)))
        hf[step, walker] = np.real(np.asarray(derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=True)["eps"]))
        pp[step, walker] = np.real(np.asarray(derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)["eps"]))
    return energy, hf, pp


def _moment_driver(ham, weights, energy, hf, pp):
    """A production MCMC object carrying the given moments (no sampling), as in the pressure reference."""
    driver = MCMC.__new__(MCMC)
    driver._mpi_comm, driver._mpi_rank, driver._mpi_size = MPI.COMM_SELF, 0, 1
    driver._MCMC__mcmc_counter, driver._MCMC__num_walkers = energy.shape
    driver._MCMC__comput_stress = True
    driver._MCMC__hamiltonian_data = ham
    driver._MCMC__stored_w_L = weights
    driver._MCMC__stored_e_L = energy
    driver._MCMC__stored_stress_HF = hf
    driver._MCMC__stored_stress_PP = pp
    driver._MCMC__stored_E_L_stress_PP = energy[..., None, None] * pp
    return driver


def _reweighted_energy_derivative(ham, configs, weights):
    """``jax.jacrev`` over eps of every delete-one reweighted empirical energy, shape (K, 3, 3)."""
    RT = jnp.eye(3)
    r_up = jnp.asarray(configs[:, :, :1].reshape(-1, 1, 3))
    r_dn = jnp.asarray(configs[:, :, 1:].reshape(-1, 1, 3))
    w_L = jnp.asarray(weights.reshape(-1))
    count = w_L.shape[0]
    ln_base = jnp.stack([jnp.real(evaluate_ln_wavefunction(ham.wavefunction_data, r_up[i], r_dn[i])) for i in range(count)])

    def delete_one_energies(eps):
        F = jnp.eye(3) + eps
        strained = ham._apply_strain(F)
        ln_psi, e_L = [], []
        for i in range(count):
            up, dn = r_up[i] @ F.T, r_dn[i] @ F.T
            ln_psi.append(jnp.real(evaluate_ln_wavefunction(strained.wavefunction_data, up, dn)))
            e_L.append(jnp.real(compute_local_energy(strained, up, dn, RT)))
        w = w_L * jnp.exp(2.0 * (jnp.stack(ln_psi) - ln_base))
        keep = 1.0 - jnp.eye(count)
        return (keep @ (w * jnp.stack(e_L))) / (keep @ w)

    return np.asarray(jax.jacrev(delete_one_energies)(jnp.zeros((3, 3))))


def _check_estimator_identity(actual_mean, actual_std, derivative, energy, hf, pp, volume):
    atol, rtol = _estimator_tolerance()
    replicates = 0.5 * (derivative + derivative.transpose(0, 2, 1)) / volume
    expected_mean = replicates.mean(axis=0)
    expected_std = np.sqrt((len(replicates) - 1) * np.mean((replicates - expected_mean) ** 2, axis=0))
    scale = max(np.max(np.abs(hf)), 2.0 * np.max(np.abs(energy[..., None, None] * pp))) / volume
    tensor_atol, tensor_rtol = get_tolerance_cancelling_mean(atol, rtol, scale)
    _assert_nonvacuous(expected_std, "estimator/std", tensor_atol, tensor_rtol)
    for label, actual, expected in (("mean", actual_mean, expected_mean), ("std", actual_std, expected_std)):
        _assert_close(actual, expected, f"estimator/{label}", tensor_atol, tensor_rtol)


def _estimator_case():
    ham = _j2_hamiltonian()
    configs = _estimator_configurations()
    energy, hf, pp = _production_moments(ham, configs)
    atol, rtol = _estimator_tolerance()
    _assert_nonvacuous(pp, "stress_PP", atol, rtol)
    derivative = _reweighted_energy_derivative(ham, configs, _ESTIMATOR_WEIGHTS)
    volume = abs(float(np.linalg.det(_CELL)))
    return ham, energy, hf, pp, derivative, volume


@pytest.mark.light
def test_get_aT_equals_ad_of_reweighted_energy():
    """Production get_aT equals the jackknifed AD derivative of the reweighted energy.

    Fixture: the triclinic plane-wave SD of ``test_stress_homogeneity`` with an
    exponential J2 (b = 1.1), so the Pulay stress is non-zero; 3 steps x 2
    walkers of fixed seeded configurations and fixed non-uniform weights w_L.
    Left side: production ``e_L``, analytic ``stress_HF`` and ``stress_PP`` per
    configuration, injected (with ``E_L_stress_PP = e_L stress_PP``) into a
    production ``MCMC`` object; ``get_aT(0, 3)`` (every configuration its own
    delete-one block).  Right side: ``jax.jacrev`` over the full 3x3 eps of
    each delete-one reweighted empirical energy
    ``sum_i w_L,i w_i(eps) e_L,eps(F x_i) / sum_i w_L,i w_i(eps)`` with
    ``w_i = |Psi_eps(F x_i)|^2 / |Psi_0(x_i)|^2`` through
    ``Hamiltonian_data._apply_strain``, symmetrised and divided by the volume;
    mean and jackknife std compared entry-wise.  This pins the sign, the
    factor 2 and the centering of the covariance term without statistics.
    Strict zone policy rebased on the largest summand
    (``get_tolerance_cancelling_mean``); full and mixed are each judged in
    their own mode; no statistical window.
    """
    ham, energy, hf, pp, derivative, volume = _estimator_case()
    actual_mean, actual_std = _moment_driver(ham, _ESTIMATOR_WEIGHTS, energy, hf, pp).get_aT(0, _ESTIMATOR_STEPS)
    _check_estimator_identity(actual_mean, actual_std, derivative, energy, hf, pp, volume)


@pytest.mark.light
def test_get_aT_reweighted_identity_rejects_combination_mutations(monkeypatch):
    """Covariance-identity negative controls: mutations of the get_aT combination formula must fail.

    Same fixture and moments.  ``MCMC.get_aT`` is recompiled from its
    production source with one edit each: (a) the Pulay factor ``2.0`` becomes
    ``1.0``; (b) the covariance term enters with a minus sign; (c) the
    centering product ``<e_L> <stress_PP>`` is dropped (the delete-one
    ``w_L e_L`` sum is multiplied by zero).  Each must fail the mean check.
    """
    ham, energy, hf, pp, derivative, volume = _estimator_case()
    mutations = {
        "factor": ("sigma_Pulay_jn_local = 2.0 * np.array(", "sigma_Pulay_jn_local = 1.0 * np.array("),
        "sign": (
            "sigma_jn_local = np.real(sigma_HF_jn_local + sigma_Pulay_jn_local)",
            "sigma_jn_local = np.real(sigma_HF_jn_local - sigma_Pulay_jn_local)",
        ),
        "centering": (
            "(w_L_e_L_binned_global_sum - w_L_e_L_binned_local[j])",
            "0.0 * (w_L_e_L_binned_global_sum - w_L_e_L_binned_local[j])",
        ),
    }
    for label, (old, new) in mutations.items():
        replacement = _mutated(MCMC.get_aT, old, new, owner="MCMC")
        with _production_mutation(monkeypatch, MCMC, "get_aT", replacement, traced=False):
            actual_mean, actual_std = _moment_driver(ham, _ESTIMATOR_WEIGHTS, energy, hf, pp).get_aT(0, _ESTIMATOR_STEPS)
            with pytest.raises(AssertionError, match=r"(?s)Not equal to tolerance.*estimator/mean"):
                _check_estimator_identity(actual_mean, actual_std, derivative, energy, hf, pp, volume)
