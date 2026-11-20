"""Individual tests: analytic ``jqmc.derivatives`` vs the ``_auto``
autograd twin, over all patterns.

The a-side goes through the PUBLIC flags interface (``compute_grad_*(..., flags,
analytic=True)[axis]`` -- STRICT, no silent ``_auto`` fallback).  The primary
parity tests send the same flags through ``analytic=False`` so the JAX oracle
differentiates exactly the production optimizer space; focused component tests
may call the internal per-axis ``_auto`` twin directly. The analytic path is
called with NO capability-gate pre-check: while a case's analytic path is not yet
implemented, strict dispatch RAISES ``NotImplementedError`` and the test FAILS
hard (RED, not xfail -- a mission remainder surfaced rather than hidden). There
is no capability-registry pre-check: every test asserts on the analytic run
itself (a==b vs JAX AD, or a finite/no-raise behavioral run), never on a registry
flag.
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

import contextlib
import dataclasses
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc import derivatives
from jqmc._precision import configure, get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from jqmc.coulomb_potential import Coulomb_potential_data
from jqmc.derivatives import _auto
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
    _phase_free_j3_orbitals,
)
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._test_data_cache import qe_example_files_root
from tests._ecp_core_electrons import core_electrons_of
from tests._mixed_ecp_ae_reference import (
    MIXED_AE_POSITIONS,
    MIXED_AE_R_UP,
    mixed_ecp_ae_potential_jnp,
    mixed_ecp_ae_system,
)
from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

_TREXIO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")


@contextlib.contextmanager
def _fp64_reference_policy():
    """Evaluate the enclosed JAX reference with every precision zone at fp64 (same-precision reference rule, fp64 kernels).

    ``jqmc.derivatives`` kernels are fp64 by construction in both modes, so a
    reference they are judged against must be fp64 too, not the mixed policy's
    fp32 ``ao_eval`` / ``jastrow_eval`` forward values. In mixed mode the full
    policy is activated for the enclosed block and the mixed policy restored
    afterwards; in full mode this is a no-op, so full-mode references are
    numerically unchanged.
    """
    was_mixed = is_mixed_precision_enabled()
    if was_mixed:
        configure("full")
    try:
        yield
    finally:
        if was_mixed:
            configure("mixed")


@pytest.fixture(autouse=True)
def _clear_jax_caches_between_derivative_tests():
    """Release large AD executables before the next derivative matrix case.

    This module exercises many unrelated Hamiltonian pytrees in one worker.
    Retaining every compiled Jacobian until worker teardown can exhaust the
    compiler on CI even when the cases pass individually. The fixture changes
    test-process cache lifetime only; production dispatch and evaluation order
    are unaffected.
    """
    yield
    jax.clear_caches()


# --------------------------------------------------------------------------- #
# Complex-twist ALL-ELECTRON (twist-AE) source (H-ae_k025) for the MCMC-side
# analytic-vs-_auto twist coverage.
#
# The committed lithium_k025.h5 twist fixture is ECP (ccECP); it never exercises
# the AE electron-ion branch (bare 1/r Coulomb + Ewald e-i, no ECP local/non-local)
# under a complex Bloch orbital. The genuine twist-AE fixture is the committed QE
# save tests/qe_example_files/H-ae_k025 (hydrogen ALL-ELECTRON, H.coulomb-ae.UPF,
# k_frac = (1/4, 1/4, 1/4)). The Hamiltonian is built AT TEST TIME via the same QE
# converter the light test_qe_to_jqmc.py / test_integration_qe_non_gamma.py tests
# use (temporary, discarded -- NO derived h5 committed). Projection kind is the
# SIMPLEST supported analytic basis: plain pGTO (project_pw_to_pgto), which at a
# non-Gamma twist yields an AOs_cart_pgto_complex_data leaf (the AE UPF carries no
# PSWFC to fit, so UPFExponentInit.shells_for_element returns an even-tempered AE
# pGTO ladder). ecp_by_element=None -> bare all-electron Coulomb + Ewald.
# --------------------------------------------------------------------------- #
_H_AE_SAVE = os.path.join(qe_example_files_root(), "H-ae_k025", "out", "h_k025.save")
_H_AE_UPF = os.path.join(_H_AE_SAVE, "H.coulomb-ae.UPF")
_H_AE_AVAILABLE = all(
    os.path.isfile(os.path.join(_H_AE_SAVE, f)) for f in ("data-file-schema.xml", "wfc1.hdf5", "H.coulomb-ae.UPF")
)
requires_h_ae = pytest.mark.skipif(not _H_AE_AVAILABLE, reason="H-ae_k025 QE all-electron twist fixture not present")


# Windowed exponent range for the twist pgto(+)PW COMPOSITE (mirrors the 8-set harness
# _QE_PGTO_ALPHA_WINDOW / _QE_PW_G_CUT_FRACTION: keep the pGTO block well-conditioned
# against the low-|G| PW block, so the analytic-vs-_auto compare is clean).
_TWIST_COMPOSITE_ALPHA_WINDOW = (1.0, 8.0)
_TWIST_COMPOSITE_PW_G_CUT_FRACTION = 0.25


def _window_shells(shells_per_atom, lo: float, hi: float):
    """Keep only the pGTO primitives with ``lo <= exponent <= hi`` (shells left empty are dropped)."""
    out = []
    for shells in shells_per_atom:
        kept = []
        for s in shells:
            pairs = [(e, c) for e, c in zip(s.exponents, s.coefficients) if lo <= e <= hi]
            if pairs:
                kept.append(
                    dataclasses.replace(s, exponents=tuple(p[0] for p in pairs), coefficients=tuple(p[1] for p in pairs))
                )
        out.append(kept)
    return out


@functools.lru_cache(maxsize=2)
def _twist_ae_pieces(kind: str):
    """(structure, complex-twist AO leaf, MO-geminal, coulomb) from the committed H-ae_k025 QE output,
    built ONCE per (process, kind). ``kind="pgto"`` -> single-leaf ``AOs_cart_pgto_complex_data``;
    ``kind="pgto+pw"`` -> a twist pgto(+)PW COMPOSITE (windowed complex cart-pGTO child + complex PW
    child), following the 8-set harness composite recipe. ALL-ELECTRON (ecp_by_element=None -> bare
    1/r e-i + Ewald, NO ECP branch)."""
    from jqmc.pw_pgto_projection import project_pw_to_mixed, project_pw_to_pgto
    from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction, build_wavefunction_from_projection
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import UPFExponentInit

    ref = load_qe_reference(_H_AE_SAVE)
    g_max = float(np.sqrt(2.0 * ref.ecutwfc))
    strat = UPFExponentInit({"H": _H_AE_UPF}, g_target=g_max)
    shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]
    if kind == "pgto":
        proj = project_pw_to_pgto(ref, shells)
    else:  # "pgto+pw" twist composite
        shells = _window_shells(shells, *_TWIST_COMPOSITE_ALPHA_WINDOW)
        proj = project_pw_to_mixed(ref, shells, g_cut=_TWIST_COMPOSITE_PW_G_CUT_FRACTION * g_max, drop_diffuse=False)
    structure_data, aos_data, geminal_mo_data = build_wavefunction_from_projection(ref, proj, kind=kind)
    ham = build_hamiltonian_from_wavefunction(
        structure_data,
        aos_data,
        geminal_mo_data,
        j1_param=1.0,
        j2_param=1.0,
        ecp_by_element=None,  # ALL-ELECTRON: bare 1/r electron-ion + Ewald, no ECP local/non-local.
    )
    return structure_data, aos_data, geminal_mo_data, ham.coulomb_potential_data


def _make_twist_j3(orb_data) -> Jastrow_three_body_data:
    """An ACTIVE three-body Jastrow on ``orb_data`` (a complex-twist AO leaf or an MOs_data wrapping
    one). Random symmetric coupling block + a linear final column (mirrors the 8-set _make_j3).
    The J3 basis is the phase-free copy shared by all twists (no J3 at the twist); the determinant keeps its twist."""
    rng = np.random.default_rng(5)
    n = int(orb_data._num_orb)
    jm = rng.normal(size=(n, n + 1)) * 0.1
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)  # bosonic coupling block must be symmetric
    return Jastrow_three_body_data(orb_data=_phase_free_j3_orbitals(orb_data), j_matrix=jm)


@functools.lru_cache(maxsize=4)
def _build_twist_ae_ham_cached(kind: str, j3: str | None, fixture_source: str) -> Hamiltonian_data:
    """(J1+J2[, J3], MO-geminal) complex-twist ALL-ELECTRON Hamiltonian from the committed QE
    H-ae_k025 output. ``kind`` selects the geminal AO leaf (single-leaf ``"pgto"`` cart-complex, or
    the ``"pgto+pw"`` twist composite). ``j3`` optionally attaches an active three-body Jastrow whose
    orbital leaf is the AO leaf itself (``"ao"``) or that leaf wrapped in an identity ``MOs_data``
    (``"mo"``). All-electron: bare 1/r electron-ion + Ewald, no ECP local/non-local branch."""
    if fixture_source != _H_AE_SAVE:
        raise ValueError(f"unexpected twist-AE fixture source: {fixture_source}")
    structure_data, aos_data, geminal_mo_data, coulomb_potential_data = _twist_ae_pieces(kind)
    n_atom = len(structure_data.positions)
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure_data
    )
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=core_electrons_of(coulomb_potential_data, n_atom),
        jastrow_1b_type="exp",
    )
    if j3 == "mo":
        j3_orb = MOs_data(num_mo=aos_data.num_ao, aos_data=aos_data, mo_coefficients=np.eye(aos_data.num_ao))
    elif j3 == "ao":
        j3_orb = aos_data
    else:
        j3_orb = None
    j3_data = _make_twist_j3(j3_orb) if j3_orb is not None else None
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=j3_data, jastrow_dtn_data=None
    )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    return Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)


def _build_twist_ae_ham(kind: str = "pgto", j3: str | None = None) -> Hamiltonian_data:
    """Return the cached Hamiltonian while re-emitting its source provenance.

    Every governed pytest case must realize the committed QE source in its own
    receipt.  Keeping this tiny wrapper uncached exposes the source path on
    every call without rebuilding the expensive projected Hamiltonian.
    """
    fixture_source = _H_AE_SAVE
    ham = _build_twist_ae_ham_cached(kind, j3, fixture_source)
    return ham


# Covering set is built IN-TEST from committed sources (no hamiltonian_data.h5). R1..R7
# add their own systems/fixtures as they land.
_SYSTEMS = ["H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5"]

# Basis-parameter optimization is covered by the focused radial-basis tests below.
# It is kept off in the general c-axis parity matrix as a coverage partition;
# joint-MO/basis optimization has its own explicit blocking evidence below.
_BASIS_OFF = dict(
    opt_lambda_basis_exp=False,
    opt_lambda_basis_coeff=False,
)


def _ref_mask(stem):
    # Radial-basis derivatives are verified separately for molecular GTO, periodic
    # pGTO, and pGTO+PW composite fixtures.  The broad parity matrix always uses
    # the production-valid direct-MO/lambda mode.
    return _BASIS_OFF


def _flags(wrt, mask=None):
    """Map a ``wrt=``/``mask=`` public-dispatch request onto the whitelist ``flags``.

    Physical axes (r/R/eps) never use the mask -- it is dropped. For the c-axis the
    ``mask`` follows the ``with_param_grad_mask`` keyword convention. ``None`` selects
    every non-basis production block while leaving the two geminal radial-basis
    blocks off. Focused tests exercise radial-basis and joint-MO/basis modes.
    A partial dict is expanded with the
    ``with_param_grad_mask`` defaults for absent keys (J1/J2/J3/geminal/DTN True,
    basis False) and then expressed as the explicit whitelist expected by the flags
    interface (absent key => False)."""
    if wrt in ("r", "R", "eps"):
        return {f"opt_{wrt}": True}
    assert wrt == "c", f"unknown axis {wrt!r}"
    if mask is None:
        return {k: "lambda_basis" not in k for k in derivatives._PARAM_FLAG_KEYS}
    flags = {k: not ("basis" in k) for k in derivatives._PARAM_FLAG_KEYS}
    flags.update({k: bool(v) for k, v in dict(mask).items()})
    return flags


_PARAM_GRAD_TO_FLAG = {
    "j1_param": "opt_J1_param",
    "j2_param": "opt_J2_param",
    "j3_matrix": "opt_J3_param",
    "mo_coefficients_up": "opt_geminal",
    "mo_coefficients_dn": "opt_geminal",
    "lambda_matrix": "opt_geminal",
    "lambda_diag": "opt_geminal",
    "lambda_unpaired": "opt_geminal",
    "j3_basis_exp": "opt_J3_basis_exp",
    "j3_basis_coeff": "opt_J3_basis_coeff",
    "lambda_basis_exp": "opt_lambda_basis_exp",
    "lambda_basis_coeff": "opt_lambda_basis_coeff",
    "j3_poly_alpha": "opt_JDTN_u_param",
    "j3_poly_beta": "opt_JDTN_chi_param",
    "j3_poly_gamma": "opt_JDTN_f_param",
}


def _active_param_grads(wf, grad_wf, flags):
    """Reduce a raw analytic/autograd pytree to the flagged optimizer blocks.

    The analytic kernels intentionally assemble every supported raw derivative;
    production performs this block filtering in ``derivatives.bridge``.  Parity
    tests using the eager public axis API must apply the same reduction before
    comparing a production-valid subset.
    """
    grads = wf.collect_param_grads(grad_wf)

    def _flag_for(name):
        if name.startswith("j3_") and name.endswith("_basis_exp"):
            return "opt_J3_basis_exp"
        if name.startswith("j3_") and name.endswith("_basis_coeff"):
            return "opt_J3_basis_coeff"
        return _PARAM_GRAD_TO_FLAG[name]

    return {name: value for name, value in grads.items() if flags.get(_flag_for(name), False)}


def _c_parameter_blocks(
    ham: Hamiltonian_data,
    r_up: jax.Array,
    r_dn: jax.Array,
    quantity: str,
    flags: dict[str, bool],
    mask: dict[str, bool] | None,
    *,
    analytic: bool,
) -> dict[str, np.ndarray]:
    """Evaluate active wavefunction-parameter derivative blocks through one route."""
    if quantity == "ln_Psi" and analytic:
        raw = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)["c"]
    elif quantity == "ln_Psi":
        raw = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="c", mask=mask)
    elif analytic:
        raw = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=True)["c"]
    else:
        raw = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="c", mask=mask)
    return {name: np.asarray(value) for name, value in _active_param_grads(ham.wavefunction_data, raw, flags).items()}


def _assert_parameter_derivative_mixed_fidelity(
    mixed: np.ndarray,
    full: np.ndarray,
    atol: float,
    rtol: float,
) -> None:
    """Require one logical parameter block to stay within its mixed error model."""
    assert np.abs(mixed - full).max() <= atol + rtol * np.abs(full).max()


def _api_remainder_compare(mixed, full):
    """Preserve each homogeneous field's central parameter-derivative budget."""
    from tests._precision_fidelity import mixed_fidelity_blocks, out_of_budget_mixed_block

    assert set(mixed) == set(full)
    assert full
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    compared = 0
    for field in full:
        assert np.asarray(mixed[field]).shape == np.asarray(full[field]).shape
        assert np.isfinite(mixed[field]).all() and np.isfinite(full[field]).all()
        if not np.asarray(full[field]).size:
            continue
        for mixed_block, full_block in mixed_fidelity_blocks(mixed[field], full[field]):
            compared += 1
            assert np.max(np.abs(mixed_block - full_block)) <= atol + rtol * np.max(np.abs(full_block)), field
            changed = out_of_budget_mixed_block(mixed_block, full_block, atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                assert np.max(np.abs(changed - full_block)) <= atol + rtol * np.max(np.abs(full_block))
    assert compared > 0


def _api_remainder_fidelity(route, *, stem=None, quantity=None, wrt=None):
    """Fresh same-backend analytical models from the unchanged original inputs."""
    import tempfile
    from pathlib import Path

    fixtures = [stem, "lithium.h5"] if route == "twist-radial" else [stem]
    for fixture in fixtures:
        entry = "mixed" if is_mixed_precision_enabled() else "full"
        with tempfile.TemporaryDirectory(prefix="jqmc_api_remainder_") as temporary:
            source = Path(temporary) / "full_physical_source.h5"
            try:
                configure("full")
                if route.startswith("twist-ae"):
                    # These test-only caches otherwise retain the policy of a
                    # previous case. Both reloads below remain independent.
                    _twist_ae_pieces.cache_clear()
                    _build_twist_ae_ham_cached.cache_clear()
                    original = _build_twist_ae_ham(kind="pgto", j3="ao" if route == "twist-ae-j3" else None)
                elif route == "dtn":
                    from jqmc.jastrow_factor import Jastrow_DTN_data

                    structure, _aos, _mu, _md, gem, cpd = read_trexio_file(os.path.join(_TREXIO, fixture), store_tuple=True)
                    z_eff, pseudized = _dtn_species_policy(structure, cpd)
                    dtn = Jastrow_DTN_data.build(
                        structure_data=structure, z_eff_per_species=z_eff, pseudized_per_species=pseudized
                    )
                    jas = Jastrow_data(
                        jastrow_one_body_data=None,
                        jastrow_two_body_data=None,
                        jastrow_three_body_data=None,
                        jastrow_dtn_data=dtn,
                    )
                    original = Hamiltonian_data(
                        structure_data=structure,
                        coulomb_potential_data=cpd,
                        wavefunction_data=Wavefunction_data(jastrow_data=jas, geminal_data=gem),
                    )
                else:
                    original = _build_ham(fixture)
                up, down = _draw(original)
                fixed_up, fixed_down = np.asarray(up).copy(), np.asarray(down).copy()
                original.save_to_hdf5(source)
            finally:
                configure(entry)

            def factory():
                return Hamiltonian_data.load_from_hdf5(source)

            def evaluate(ham):
                wf = ham.wavefunction_data
                up, down = jnp.asarray(fixed_up), jnp.asarray(fixed_down)
                if route.startswith("twist"):
                    assert ham.structure_data.pbc_flag
                if route.startswith("twist-ae"):
                    assert not ham.coulomb_potential_data.ecp_flag
                if route in ("twist-ae-radial", "twist-ae-j3"):
                    from jqmc.atomic_orbital import AOs_cart_pgto_complex_data

                    assert isinstance(wf.geminal_data.orb_data_up_spin.aos_data, AOs_cart_pgto_complex_data)
                if route == "dtn":
                    result = derivatives.compute_grad_ln_Psi(ham, up, down, _flags("c"), analytic=True)["c"]
                    values = result.jastrow_data.jastrow_dtn_data
                    keys = ("alpha_indep", "beta_indep", "gamma_indep")
                    return {key: np.asarray(getattr(values, key)) for key in keys}, {"dtn": keys}
                axes = ("c", "r", "R", "eps") if route == "twist-ecp" else (wrt or "c",)
                quantities = ("ln_Psi", "e_L") if route in ("twist-ecp", "twist-radial") else (quantity,)
                blocks = {}
                keysets = {}
                for q in quantities:
                    for axis in axes:
                        flags = _flags(axis)
                        if route == "twist-radial":
                            flags = {"opt_lambda_basis_exp": True, "opt_lambda_basis_coeff": True}
                        elif route == "twist-ae" and axis == "c":
                            flags = {key: True for key in derivatives._PARAM_FLAG_KEYS}
                            flags["opt_geminal"] = False
                        raw = (
                            derivatives.compute_grad_ln_Psi(ham, up, down, flags, analytic=True)[axis]
                            if q == "ln_Psi"
                            else derivatives.compute_grad_e_L(ham, up, down, jnp.eye(3), flags, analytic=True)[axis]
                        )
                        if axis == "c":
                            values = wf.collect_param_grads(raw)
                            keysets[q + "/c"] = tuple(sorted(values))
                            if route in ("twist-radial", "twist-ae"):
                                values = _active_param_grads(wf, raw, flags)
                                assert {"lambda_basis_exp", "lambda_basis_coeff"} <= set(values)
                                if route == "twist-radial":
                                    assert set(values) == {"lambda_basis_exp", "lambda_basis_coeff"}
                            elif route in ("twist-ae-radial", "twist-ae-j3"):
                                required = (
                                    ("lambda_basis_exp", "lambda_basis_coeff")
                                    if route == "twist-ae-radial"
                                    else ("j3_basis_exp", "j3_basis_coeff")
                                )
                                assert set(required) <= set(values)
                                values = {key: values[key] for key in required}
                            else:
                                # The original twist ECP case freezes basis
                                # fields and judges all non-basis fields.
                                values = {key: value for key, value in values.items() if "basis" not in key}
                            for key, value in values.items():
                                blocks[q + "/c/" + key] = np.asarray(value)
                        else:
                            for i, value in enumerate(jax.tree_util.tree_leaves(raw)):
                                blocks[q + "/" + axis + "/" + str(i)] = np.asarray(value)
                return blocks, keysets

            mixed, full = mixed_vs_full(factory, evaluate)
        assert mixed[1] == full[1]
        _api_remainder_compare(mixed[0], full[0])


def _api_joint_fidelity(stem, geminal, basis, quantity):
    """Retain joint/single analytical flag consistency and cross-policy fidelity."""
    import tempfile
    from pathlib import Path

    from jqmc.derivatives.bridge import _reduce_c_param_grads
    from tests.test_jqmc_kernel._generated_matrix_base import _apply_gem

    flags = {
        "opt_geminal": True,
        "opt_lambda_basis_exp": basis in ("exponents", "both"),
        "opt_lambda_basis_coeff": basis in ("coefficients", "both"),
    }
    entry = "mixed" if is_mixed_precision_enabled() else "full"
    with tempfile.TemporaryDirectory(prefix="jqmc_api_joint_") as temporary:
        source = Path(temporary) / "full_physical_source.h5"
        try:
            configure("full")
            original = _apply_gem(_build_ham(stem), geminal)
            up, down = _draw(original, seed=0)
            fixed_up, fixed_down = np.asarray(up).copy(), np.asarray(down).copy()
            original.save_to_hdf5(source)
        finally:
            configure(entry)

        def factory():
            return Hamiltonian_data.load_from_hdf5(source)

        def evaluate(ham):
            wf = ham.wavefunction_data
            up, down = jnp.asarray(fixed_up), jnp.asarray(fixed_down)

            def gradient(selected):
                raw = (
                    derivatives.compute_grad_ln_Psi(ham, up, down, selected, analytic=True)["c"]
                    if quantity == "ln_Psi"
                    else derivatives.compute_grad_e_L(ham, up, down, jnp.eye(3), selected, analytic=True)["c"]
                )
                batched = jax.tree_util.tree_map(lambda value: jnp.expand_dims(value, 0), raw)
                flat = _reduce_c_param_grads(wf, batched, 1, selected)
                return {
                    name: np.asarray(value)[0] for name, value in flat.items() if selected.get(_PARAM_GRAD_TO_FLAG[name], False)
                }

            joint = gradient(flags)
            mo_only = gradient({"opt_geminal": True})
            basis_only = gradient({**flags, "opt_geminal": False})
            assert "mo_coefficients_up" in joint
            assert set(joint) == set(mo_only) | set(basis_only)
            level = "medium" if is_mixed_precision_enabled() else "strict"
            atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), level)
            for block, isolated in {**mo_only, **basis_only}.items():
                assert np.all(np.isfinite(joint[block])), block
                np.testing.assert_allclose(joint[block], isolated, atol=atol, rtol=rtol, err_msg=f"single block {block}")
            for block in ("mo_coefficients_up", *basis_only):
                assert np.max(np.abs(joint[block])) > atol, f"inactive requested block {block}"
            return joint

        mixed, full = mixed_vs_full(factory, evaluate)
    _api_remainder_compare(mixed, full)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["H2_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5"])
@pytest.mark.parametrize("geminal", ["SD", "AGPn"])
@pytest.mark.parametrize("basis", ["exponents", "coefficients", "both"])
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_joint_mo_basis_gradients_match_single_block_and_analytic(stem, geminal, basis, quantity):
    """Joint MO/radial optimization preserves each single-block derivative and matches the analytical route.

    The committed H2 Cartesian/spherical GTO fixture is realized as SD or AGPn,
    with the existing seed-zero electron coordinates. MO coefficients and the
    requested radial exponents/contractions must all carry a finite signal.
    Both routes use the same active full/mixed policy and central derivative
    tolerance, without finite differences or cross-precision comparisons.
    """
    if is_mixed_precision_enabled():
        _api_joint_fidelity(stem, geminal, basis, quantity)
        return
    from tests.test_jqmc_kernel._generated_matrix_base import _apply_gem
    from jqmc.derivatives.bridge import _reduce_c_param_grads

    ham = _apply_gem(_build_ham(stem), geminal)
    r_up, r_dn = _draw(ham, seed=0)
    flags = {
        "opt_geminal": True,
        "opt_lambda_basis_exp": basis in ("exponents", "both"),
        "opt_lambda_basis_coeff": basis in ("coefficients", "both"),
    }
    wf = ham.wavefunction_data
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), level)

    def gradient(selected, analytic):
        if quantity == "ln_Psi":
            raw = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, selected, analytic=analytic)["c"]
        else:
            raw = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), selected, analytic=analytic)["c"]
        # AGPn.collect_param_grads exposes a legacy dense-lambda view rather
        # than its optimizer's direct-MO leaves. Use the actual driver reducer
        # for both representations, then retain exactly the requested blocks.
        batched = jax.tree_util.tree_map(lambda value: jnp.expand_dims(value, 0), raw)
        flat = _reduce_c_param_grads(wf, batched, 1, selected)
        return {name: np.asarray(value)[0] for name, value in flat.items() if selected.get(_PARAM_GRAD_TO_FLAG[name], False)}

    joint = gradient(flags, False)
    analytical = gradient(flags, True)
    mo_only = gradient({"opt_geminal": True}, False)
    basis_only = gradient({**flags, "opt_geminal": False}, False)
    assert "mo_coefficients_up" in joint
    assert set(joint) == set(analytical) == set(mo_only) | set(basis_only)
    for block, isolated in {**mo_only, **basis_only}.items():
        actual = np.asarray(joint[block])
        assert np.all(np.isfinite(actual)), block
        np.testing.assert_allclose(actual, np.asarray(isolated), atol=atol, rtol=rtol, err_msg=f"single block {block}")
        np.testing.assert_allclose(actual, np.asarray(analytical[block]), atol=atol, rtol=rtol, err_msg=f"analytical {block}")
    for block in ("mo_coefficients_up", *basis_only):
        assert np.max(np.abs(np.asarray(joint[block]))) > atol, f"inactive requested block {block}"


def _build_ham(stem: str) -> Hamiltonian_data:
    (structure_data, aos_data, mos_up, mos_dn, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        os.path.join(_TREXIO, stem), store_tuple=True
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure_data
    )
    n_atom = len(structure_data.positions)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=core_electrons_of(coulomb_potential_data, n_atom),
        jastrow_1b_type="exp",
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=j1,
        jastrow_two_body_data=j2,
        jastrow_three_body_data=None,
        jastrow_dtn_data=None,
    )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    return Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wf,
    )


def _draw(ham: Hamiltonian_data, seed: int = 0):
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(scale=0.7, size=(nu, 3))),
        jnp.asarray(rng.normal(scale=0.7, size=(nd, 3))),
    )


def _dtn_species_policy(
    structure_data: Structure_data,
    coulomb_potential_data: Coulomb_potential_data,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    """Collapse the Coulomb block's per-atom ECP policy to DTN species."""
    atomic_numbers = np.asarray(structure_data.atomic_numbers)
    unique_atomic_numbers = np.unique(atomic_numbers)
    effective_charges = np.asarray(coulomb_potential_data._effective_charges)
    return (
        tuple(float(np.mean(effective_charges[atomic_numbers == Z])) for Z in unique_atomic_numbers),
        tuple(bool(np.all(coulomb_potential_data.pseudized_per_atom[atomic_numbers == Z])) for Z in unique_atomic_numbers),
    )


def _assert_molecular_mixed_derivative_block(mixed_block, full_block, atol, rtol, label):
    """Apply the existing numerical policy once per homogeneous derivative block."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max(), label


def _molecular_gradient_fidelity(stem, quantity, wrt, route):
    """Rebuild the original public, fresh bridge or legacy driver route per policy."""
    from jqmc.derivatives import bridge
    from jqmc.derivatives.driver import make_param_grad_closures

    def factory():
        bridge.cache_clear()
        ham = _build_ham(stem)
        r_up, r_dn = _draw(ham)
        return ham, r_up, r_dn

    def evaluate(case):
        ham, r_up, r_dn = case
        mask = _ref_mask(stem)
        flags = _flags(wrt, mask)
        if route == "driver":
            assert quantity == "both" and wrt == "c"
            up, down, rotation = r_up[None], r_dn[None], jnp.eye(3)[None]
            o_k, de_L = make_param_grad_closures(ham)
            actual = {
                "ln_Psi": _covered_leaves(o_k(ham.wavefunction_data, up, down, None)),
                "e_L": _covered_leaves(de_L(ham, up, down, rotation).wavefunction_data),
            }
        else:
            if route == "public":
                if quantity == "ln_Psi":
                    gradient = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)[wrt]
                else:
                    assert quantity == "e_L"
                    gradient = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=True)[wrt]
            else:
                assert route == "bridge"
                if quantity == "ln_Psi":
                    batched = bridge.batched_grad_ln_Psi(wrt=wrt, mask=mask)(ham, r_up[None], r_dn[None])
                else:
                    assert quantity == "e_L"
                    batched = bridge.batched_grad_e_L(wrt=wrt, mask=mask)(ham, r_up[None], r_dn[None], jnp.eye(3)[None])
                gradient = jax.tree_util.tree_map(lambda value: value[0], batched)
            # Match the original public collector/filter, within this policy.
            # The bridge retains its original fresh factorization convention.
            actual = _active_param_grads(ham.wavefunction_data, gradient, flags) if wrt == "c" else gradient
        return {"actual": actual, "r_up": r_up, "r_dn": r_dn}

    mixed, full = mixed_vs_full(factory, evaluate)
    assert np.array_equal(mixed["r_up"], full["r_up"]) and np.array_equal(mixed["r_dn"], full["r_dn"]), (
        "full/mixed molecular fixture coordinates differ"
    )
    blocks = mixed_fidelity_blocks(mixed["actual"], full["actual"])
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    label = f"{stem} {route} {quantity}/{wrt} mixed vs full"
    for mixed_block, full_block in blocks:
        _assert_molecular_mixed_derivative_block(mixed_block, full_block, atol, rtol, label)
    faulty = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _assert_molecular_mixed_derivative_block(faulty, blocks[0][1], atol, rtol, label)


@pytest.mark.light
@pytest.mark.parametrize("stem", _SYSTEMS)
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
@pytest.mark.parametrize("wrt", ["c", "R", "eps"])
def test_analytic_vs_auto(stem, quantity, wrt):
    """Analytic vs auto."""
    if is_mixed_precision_enabled():
        _molecular_gradient_fidelity(stem, quantity, wrt, "public")
        return
    ham = _build_ham(stem)
    r_up, r_dn = _draw(ham)
    RT = jnp.eye(3)

    # The general matrix uses direct-MO/lambda mode. Radial-basis mode has
    # independent focused parity tests.
    mask = _ref_mask(stem)
    flags = _flags(wrt, mask)
    if quantity == "ln_Psi":
        a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)[wrt]
        b = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False)[wrt]
    else:
        a = derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=True)[wrt]
        b = derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=False)[wrt]

    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    wf = ham.wavefunction_data
    if wrt == "c":  # compare the variational-parameter blocks (the meaningful O_k / dE_L/dc)
        da = _active_param_grads(wf, a, flags)
        db = _active_param_grads(wf, b, flags)
        assert set(da) == set(db)
        for k in da:
            np.testing.assert_allclose(np.asarray(da[k]), np.asarray(db[k]), atol=atol, rtol=rtol)
    else:  # R/eps/r: compare the assembled axis grad leaf-for-leaf
        for la, lb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
            np.testing.assert_allclose(np.asarray(la), np.asarray(lb), atol=atol, rtol=rtol)


# The production drivers call the analytic grads as jit(vmap(...)) over a walker
# batch with the hamiltonian on a None axis (TRACED, not a closure constant). This
# guards that the analytic path is fully jnp (no host-side numpy on traced leaves)
# so it compiles in that shape and still matches _auto. Spherical-GTO is covered too:
# the bridge hoists the sphe->cart basis conversion out of the per-walker jit
# (derivatives.hoist.cartesianize_orbitals), so sphe R/eps compile and match the
# ORIGINAL spherical _auto (the hoist preserves the MO functions -> identical grads).


@pytest.mark.light
@pytest.mark.parametrize("stem", _SYSTEMS)
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
@pytest.mark.parametrize("wrt", ["c", "R", "eps"])
def test_bridge_jit_vs_auto(stem, quantity, wrt):
    """Bridge jit vs auto."""
    if is_mixed_precision_enabled():
        _molecular_gradient_fidelity(stem, quantity, wrt, "bridge")
        return
    from jqmc.derivatives import bridge

    ham = _build_ham(stem)
    r_up, r_dn = _draw(ham)

    rub, rdb, rtb = r_up[None], r_dn[None], jnp.eye(3)[None]
    mask = _ref_mask(stem)
    flags = _flags(wrt, mask)
    if quantity == "ln_Psi":
        out = bridge.batched_grad_ln_Psi(wrt=wrt, mask=mask)(ham, rub, rdb)
        b = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False)[wrt]
    else:
        out = bridge.batched_grad_e_L(wrt=wrt, mask=mask)(ham, rub, rdb, rtb)
        b = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=False)[wrt]
    a = jax.tree_util.tree_map(lambda x: x[0], out)  # un-batch the single walker

    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    wf = ham.wavefunction_data
    if wrt == "c":
        da = _active_param_grads(wf, a, flags)
        db = _active_param_grads(wf, b, flags)
        assert set(da) == set(db)
        for k in da:
            np.testing.assert_allclose(np.asarray(da[k]), np.asarray(db[k]), atol=atol, rtol=rtol)
    else:
        for la, lb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
            np.testing.assert_allclose(np.asarray(la), np.asarray(lb), atol=atol, rtol=rtol)


# Regression guard for the "traced-ham trap": the PRODUCTION
# MCMC/GFMC drivers wrap ``compute_grad_e_L(analytic=True)`` in ``jit(vmap(_single,
# in_axes=(None, 0, 0, 0)))`` with the WHOLE hamiltonian as a TRACED arg -- and, unlike
# ``test_bridge_jit_vs_auto`` above, they do NOT route through ``bridge`` /
# ``cartesianize_orbitals``, so nothing hoists the spherical->Cartesian basis conversion
# out of the trace. For a SPHE-gto (molecular) OR SPHE-pgto (PBC) wavefunction the R (force)
# and eps (stress) axes take Cartesian-GTO AO derivatives via the sphe->cart conversion
# (``_aos_sphe_to_cart`` on the traced ``radial.exponents``; ``MOs_data.to_cartesian``'s host
# ``np.asarray(mo_coefficients)`` matmul), both of which raise TracerArrayConversionError
# under this bridge if they run on a traced leaf (at DISTINCT sites: ``pgto_deriv.sphe_leaf_to_cart`` for PBC,
# ``ao_deriv.orb_to_cart_pieces`` for molecular). This guard reproduces exactly that closure shape
# and asserts it TRACES (``jax.eval_shape`` -- no XLA compile/exec, so it stays in the fast suite)
# for BOTH force and stress on BOTH a molecular and a PBC (real + complex-twist) spherical basis,
# catching any re-introduction of a host-numpy-on-a-traced-leaf op without the heavy integration lane.
def _build_sphe_jnone_ham(stem: str) -> Hamiltonian_data:
    (structure_data, aos_data, mos_up, mos_dn, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        os.path.join(_TREXIO, stem), store_tuple=True
    )
    # Minimal (no-Jastrow) ham: the trap lives in the geminal orbital sphe->cart kinetic/stress
    # path, independent of the Jastrow blocks -- keeping jnone makes the trace cheap.
    wf = Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=geminal_mo_data)
    return Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)


@pytest.mark.light
@pytest.mark.parametrize(
    "stem",
    ["H2_ae_ccpvdz_sphe.h5", "lithium.h5", "lithium_k025.h5"],  # molecular sphe-gto; real + twist sphe-pgto
)
@pytest.mark.parametrize("wrt", ["R", "eps"])  # force + stress -- the two sphe->cart trap sites
def test_sphe_production_bridge_traces(stem, wrt):
    """Sphe production bridge traces."""
    path = os.path.join(_TREXIO, stem)
    if not os.path.exists(path):
        pytest.skip(f"fixture {stem} not present")
    ham = _build_sphe_jnone_ham(stem)
    gd = ham.wavefunction_data.geminal_data
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(0)
    rub = jnp.asarray(rng.normal(scale=0.7, size=(1, nu, 3)))
    rdb = jnp.asarray(rng.normal(scale=0.7, size=(1, nd, 3)))
    rtb = jnp.eye(3)[None]

    # Exactly the production closure shape (jqmc_mcmc._cached_grad_e_L_rR / _eps): the ham is a
    # traced (None-axis) arg, so its basis exponents are tracers -- no bridge hoist in between.
    def _single(h, a, b, RT):
        return derivatives.compute_grad_e_L(h, a, b, RT, {f"opt_{wrt}": True}, analytic=True)[wrt]

    fn = jax.jit(jax.vmap(_single, in_axes=(None, 0, 0, 0)))
    # eval_shape traces the whole closure (triggering any host-numpy-on-a-traced-leaf op) without
    # lowering to XLA -- so a re-introduced traced-ham trap raises TracerArrayConversionError here.
    out = jax.eval_shape(fn, ham, rub, rdb, rtb)
    n_atom = int(np.asarray(ham.structure_data.positions).shape[0])
    expected = (1, 3, 3) if wrt == "eps" else (1, n_atom, 3)  # eps: 3x3 stress; R: per-atom force
    assert tuple(out.shape) == expected


# The analytic driver adapters (derivatives.driver) must produce param
# grads bit-equal to the production MCMC autograd closures (jqmc_mcmc module
# globals) on the COVERED variational blocks, so the loop can swap closures behind
# a flag. This legacy adapter check compares its raw J1/J2 leaves. SD's
# structural ``lambda_unpaired`` is deliberately absent: JAX can differentiate
# that stored field mathematically, but it is not an optimizer block. Production
# MO-block coverage lives in the route matrix and exact-block evidence.
def _covered_leaves(wf_grad):
    return {
        "j1_param": np.asarray(wf_grad.jastrow_data.jastrow_one_body_data.jastrow_1b_param),
        "j2_param": np.asarray(wf_grad.jastrow_data.jastrow_two_body_data.jastrow_2b_param),
    }


@pytest.mark.light
@pytest.mark.parametrize("stem", _SYSTEMS)
def test_driver_param_closures_vs_production_autograd(stem):
    """Driver param closures vs production autograd."""
    if is_mixed_precision_enabled():
        _molecular_gradient_fidelity(stem, "both", "c", "driver")
        return
    from jqmc.derivatives.driver import make_param_grad_closures

    ham = _build_ham(stem)
    r_up, r_dn = _draw(ham)
    wf = ham.wavefunction_data
    rub, rdb, rtb = r_up[None], r_dn[None], jnp.eye(3)[None]
    o_k, de_L = make_param_grad_closures(ham)

    # The "production autograd" reference is now sourced from the production BRIDGE
    # c-axis autograd twin: ``compute_grad_{ln_Psi,e_L}(..., flags, analytic=False)["c"]``
    # (analytic=False == the pre-C1 masked autograd numerically), differentiating every
    # parameter block (flags all-True -> nothing frozen), vmapped over the one-walker
    # batch. This yields the RAW ``Wavefunction_data``-shaped grad MCMC.run's c-axis
    # closure emits (before ``_reduce_c_param_grads`` collapses it), so ``_covered_leaves``
    # extracts the legacy adapter's raw J1/J2 leaves.
    c_flags = _flags("c")
    prod_ok_wf = jax.jit(
        jax.vmap(lambda a, b: derivatives.compute_grad_ln_Psi(ham, a, b, c_flags, analytic=False)["c"], in_axes=(0, 0))
    )(rub, rdb)
    prod_el_wf = jax.jit(
        jax.vmap(lambda a, b, rt: derivatives.compute_grad_e_L(ham, a, b, rt, c_flags, analytic=False)["c"], in_axes=(0, 0, 0))
    )(rub, rdb, rtb)

    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    # O_k = d ln|Psi| / dc : production bridge autograd twin vs analytic
    prod_ok = _covered_leaves(prod_ok_wf)
    ana_ok = _covered_leaves(o_k(wf, rub, rdb, None))
    # dE_L/dc : production bridge autograd twin vs analytic
    prod_el = _covered_leaves(prod_el_wf)
    ana_el = _covered_leaves(de_L(ham, rub, rdb, rtb).wavefunction_data)
    for k in prod_ok:
        if prod_ok[k].size:
            np.testing.assert_allclose(ana_ok[k], prod_ok[k], atol=atol, rtol=rtol)
            np.testing.assert_allclose(ana_el[k], prod_el[k], atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize(
    "stem",
    [
        "N2_ecp_ccpvtz_cart.h5",
        "H2_ecp_ccpvdz_cart.h5",
    ],
)
def test_ecp_e_L_dr_vs_auto(stem):
    """ECP molecular drift d e_L/dr = dT/dr + dV_bare/dr + dV_ecp/dr. The ECP non-local electron
    gradient (ecp_force.grad_V_ecp_wrt_r): explicit projector/weight geometry + mesh-numerator via
    the drift field O at the quadrature config + denominator, min-image. vs _auto.

    Both committed N2/ccpvtz and H2/ccpvdz configurations compare within the current
    precision mode at the zone-derived strict tolerance, without conditioning
    reselection or a precision exclusion."""
    if is_mixed_precision_enabled():
        _api_first17_ecp_fidelity(stem, "r")
        return
    ham = _build_ham(stem)
    r_up, r_dn = _draw(ham)
    assert ham.coulomb_potential_data.ecp_flag
    a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("r"), analytic=True)["r"]
    b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="r", mask=_ref_mask(stem))
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    al = np.concatenate([np.asarray(x) for x in a], 0)
    bl = np.concatenate([np.asarray(x) for x in jax.tree_util.tree_leaves(b)], 0)
    np.testing.assert_allclose(al, bl, atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize(
    "stem",
    [
        "N2_ecp_ccpvtz_cart.h5",
        "H2_ecp_ccpvdz_cart.h5",
    ],
)
def test_ecp_e_L_deps_vs_auto(stem):
    """MOLECULAR ECP stress d e_L/d eps = T + V_bare affine virial (_stress_eL) + ECP local/non-local
    strain (ecp_force.grad_V_ecp_wrt_strain_mol: local affine virial + non-local geometry strain +
    non-local wf-ratio MOLECULAR Pulay stress in CLOSED FORM off the rank-1 config drift/force fields --
    NO autodiff-through-pinv). This is the LAST production ``_auto`` fallback removed (molecular-ECP
    e_L/eps). vs _auto's strained forward (_real_local_energy_strained; electrons r@F.T, nuclei via
    _apply_strain(F), quadrature rotation RT held fixed). Cross-graph strict tolerance, mirroring test_ecp_e_L_dr_vs_auto."""
    if is_mixed_precision_enabled():
        _api_first17_ecp_fidelity(stem, "eps")
        return
    ham = _build_ham(stem)
    r_up, r_dn = _draw(ham)
    assert ham.coulomb_potential_data.ecp_flag
    a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("eps"), analytic=True)["eps"]
    b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="eps", mask=_ref_mask(stem))
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["H2_ecp_ccpvdz_cart.h5"])
def test_ecp_e_L_eps_shared_inverse_no_svd(stem):
    """The MOLECULAR ECP e_L/eps closed-form Pulay stress reuses the running geminal inverse (no
    autodiff-through-pinv). With the per-walker ``geminal_inv`` threaded, the lowered StableHLO carries
    ZERO ``lapack_dgesdd_ffi`` for e_L/{r,R,eps} (was exactly ONE each without the shared inverse).
    Numerically inert: providing the inverse agrees with the SVD-pinv fallback to fp."""
    from jqmc.derivatives import bridge
    from jqmc.derivatives.fast import geminal_pinv
    from jqmc.derivatives.hoist import cartesianize_orbitals
    from jqmc.determinant import compute_geminal_all_elements

    ham = _build_ham(stem)
    assert ham.coulomb_potential_data.ecp_flag and not ham.structure_data.pbc_flag
    gd = ham.wavefunction_data.geminal_data
    ru0, rd0 = _draw(ham, seed=0)
    ru1, rd1 = _draw(ham, seed=1)
    rub = jnp.stack([ru0, ru1])
    rdb = jnp.stack([rd0, rd1])
    RTb = jnp.tile(jnp.eye(3)[None], (rub.shape[0], 1, 1))
    gi_b = jax.jit(jax.vmap(lambda a, b: geminal_pinv(compute_geminal_all_elements(gd, a, b))))(rub, rdb)
    jax.block_until_ready(gi_b)

    hc = cartesianize_orbitals(ham)
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), level)
    for wrt in ("r", "R", "eps"):
        txt = bridge._jit_e_L_giv(wrt, False).lower(hc, rub, rdb, RTb, gi_b).as_text()
        assert txt.count("lapack_dgesdd_ffi") == 0, (
            f"molecular ECP e_L/{wrt}: {txt.count('lapack_dgesdd_ffi')} SVD FFI calls, expected 0"
        )
        a = np.asarray(bridge.batched_grad_e_L(wrt=wrt, with_geminal_inv=False)(ham, rub, rdb, RTb))
        b = np.asarray(bridge.batched_grad_e_L(wrt=wrt, with_geminal_inv=True)(ham, rub, rdb, RTb, gi_b))
        np.testing.assert_allclose(b, a, atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["H2_ecp_ccpvdz_cart.h5", "H2_ecp_ccpvtz_cart.h5"])
def test_ecp_e_L_dc_vs_auto(stem):
    """ECP non-local dE_L/dc = dT/dc + sum_g V_grid_g O_k(r'_g) - sumV O_k(r), with the
    production NN-truncated mesh. Covered-set e_L/c (basis-opt off) vs _auto."""
    if is_mixed_precision_enabled():
        _api_first17_ecp_fidelity(stem, "c")
        return
    ham = _build_ham(stem)
    r_up, r_dn = _draw(ham)
    assert ham.coulomb_potential_data.ecp_flag
    c_flags = _flags("c")
    a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), c_flags, analytic=True)["c"]
    b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="c", mask=_ref_mask(stem))
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    wf = ham.wavefunction_data
    da = _active_param_grads(wf, a, c_flags)
    db = _active_param_grads(wf, b, c_flags)
    assert set(da) == set(db)
    for k in da:
        np.testing.assert_allclose(np.asarray(da[k]), np.asarray(db[k]), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["N2_ecp_ccpvtz_cart.h5"])
def test_dtn_o_k_vs_auto(stem):
    """DTN polynomial Jastrow O_k = d ln|Psi| / d (alpha/beta/gamma indep). DTN built
    at test time from the trexio structure (Jastrow_DTN_data.build); vs _auto."""
    if is_mixed_precision_enabled():
        _api_remainder_fidelity("dtn", stem=stem)
        return
    from jqmc.jastrow_factor import Jastrow_DTN_data

    (structure_data, _aos, _mu, _md, gem, cpd) = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    z_eff, pseudized = _dtn_species_policy(structure_data, cpd)
    dtn = Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=z_eff,
        pseudized_per_species=pseudized,
    )
    jastrow = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=dtn
    )
    wf = Wavefunction_data(jastrow_data=jastrow, geminal_data=gem)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=cpd, wavefunction_data=wf)
    r_up, r_dn = _draw(ham)
    a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags("c"), analytic=True)["c"]
    b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="c", mask=None)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    da = a.jastrow_data.jastrow_dtn_data
    db = b.jastrow_data.jastrow_dtn_data
    for k in ("alpha_indep", "beta_indep", "gamma_indep"):
        np.testing.assert_allclose(np.asarray(getattr(da, k)), np.asarray(getattr(db, k)), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5"])
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_pbc_real_c_axis_vs_auto(stem, quantity):
    """Real Gamma-point periodic (ECP, sphe-pgto) c-axis derivatives.

    Full precision proves the analytical route against autodiff. Mixed
    precision instead bounds the deviation of that same analytical production
    route from an independently rebuilt full-precision case, one logical
    parameter block at a time.

    The J1/J2 c-grads use the Williamson smooth periodic distance; for dE_L/dc the J1/J2
    kinetic uses that distance's closed-form grad+laplacian (kinetic._dist_and_derivs PBC
    branch -- the Euclidean f''+2f'/r radial-laplacian identity breaks on the torus). Ewald
    & bare Coulomb are c-independent (= 0); the geminal kinetic is pbc-aware; dV_ecp/dc
    reuses the basis/pbc-agnostic O_k(r') mesh identity. This test selects the
    sphe-pgto radial-basis mode with direct-MO optimization disabled. Twisted
    (complex-Bloch, k != 0) PBC delegates to _auto."""
    # sphe-pgto lambda radial basis-opt is now analytic (R5) -> verify it ACTIVE (lithium has no J3,
    # so the separate J3-basis radial opt is moot and kept off).
    mask = {
        "opt_geminal": False,
        "opt_lambda_basis_exp": True,
        "opt_lambda_basis_coeff": True,
        "opt_J3_basis_exp": False,
        "opt_J3_basis_coeff": False,
    }
    c_flags = _flags("c", mask)

    if is_mixed_precision_enabled():

        def case_factory() -> tuple[Hamiltonian_data, jax.Array, jax.Array]:
            ham = _build_ham(stem)
            assert ham.structure_data.pbc_flag
            return ham, *_draw(ham)

        def evaluate(case: tuple[Hamiltonian_data, jax.Array, jax.Array]) -> dict[str, np.ndarray]:
            ham, r_up, r_dn = case
            return _c_parameter_blocks(ham, r_up, r_dn, quantity, c_flags, mask, analytic=True)

        da, db = mixed_vs_full(case_factory, evaluate)
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        assert set(da) == set(db)
        assert {"lambda_basis_exp", "lambda_basis_coeff"} <= set(da)
        for k in da:
            if not np.all(np.isfinite(da[k])) or not np.all(np.isfinite(db[k])):
                raise AssertionError("mixed-fidelity parameter blocks must be finite")
            _assert_parameter_derivative_mixed_fidelity(da[k], db[k], atol, rtol)
        return

    ham = _build_ham(stem)
    assert ham.structure_data.pbc_flag
    r_up, r_dn = _draw(ham)
    da = _c_parameter_blocks(ham, r_up, r_dn, quantity, c_flags, mask, analytic=True)
    db = _c_parameter_blocks(ham, r_up, r_dn, quantity, c_flags, mask, analytic=False)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    assert set(da) == set(db)
    assert {"lambda_basis_exp", "lambda_basis_coeff"} <= set(da)
    for k in da:
        np.testing.assert_allclose(np.asarray(da[k]), np.asarray(db[k]), atol=atol, rtol=rtol)


# The geminal radial-basis blocks
# (``basis.grad_ln_det_wrt_basis`` for ln_Psi, ``basis.grad_T_wrt_basis`` for e_L) on the
# REAL periodic sphe-pGTO leaf (lithium.h5, ccECP, Gamma) WITH a Jastrow. The e_L block
# carries the Jastrow-coupled kinetic term (the Jastrow gradient enters the geminal
# reverse sweep), so each case also shows that this term is live: the autodiff block
# with the Jastrow differs from the same block without it.
_PBC_BASIS_BLOCKS = ("lambda_basis_exp", "lambda_basis_coeff")
_PBC_BASIS_MASK = {
    "opt_geminal": False,
    "opt_lambda_basis_exp": True,
    "opt_lambda_basis_coeff": True,
    "opt_J3_basis_exp": False,
    "opt_J3_basis_coeff": False,
}


def _pbc_basis_ham(jastrow: str) -> Hamiltonian_data:
    """lithium.h5 with the committed-source J1+J2 (exp) or J1+J2+J3 (sphe-pGTO J3 orbital)."""
    return _build_ham("lithium.h5") if jastrow == "J1J2" else _build_ham_j3("lithium.h5")


def _without_jastrow(ham: Hamiltonian_data) -> Hamiltonian_data:
    wf = ham.wavefunction_data
    return ham.replace(wavefunction_data=wf.replace(jastrow_data=Jastrow_data()))


def _pbc_basis_blocks(ham: Hamiltonian_data, r_up, r_dn, quantity: str, analytic: bool) -> dict:
    """The two geminal radial-basis blocks of d ln_Psi/dc or dE_L/dc (analytic or ``_auto``)."""
    flags = _flags("c", _PBC_BASIS_MASK)
    blocks = _c_parameter_blocks(ham, r_up, r_dn, quantity, flags, _PBC_BASIS_MASK, analytic=analytic)
    assert set(_PBC_BASIS_BLOCKS) <= set(blocks), sorted(blocks)
    return {name: blocks[name] for name in _PBC_BASIS_BLOCKS}


def _assert_pbc_basis_blocks_match(analytic: dict, reference: dict, atol: float, rtol: float) -> None:
    expected_tolerance = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    assert (atol, rtol) == expected_tolerance
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    for name in _PBC_BASIS_BLOCKS:
        actual, expected = analytic[name], reference[name]
        assert np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)), name
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=f"periodic pGTO {name}")


@pytest.mark.light
@pytest.mark.parametrize("jastrow", ["J1J2", "J1J2J3"])
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_pbc_real_pgto_basis_grads_with_jastrow_vs_auto(jastrow, quantity):
    """Periodic real sphe-pGTO radial-basis derivative correctness and fidelity.

    lithium.h5 (ccECP, Gamma) with J1+J2 or J1+J2+J3. Both lambda_basis_exp and
    lambda_basis_coeff of d ln_Psi/dc and dE_L/dc must match ``_auto`` under
    full precision. Under mixed precision each analytical production block is
    instead compared with the same route rebuilt in full precision. Every
    block must be nonzero above its applicable comparison atol.
    For e_L the autodiff block with the Jastrow must also differ from the block of the
    same wavefunction without the Jastrow by more than atol, so the Jastrow-coupled
    kinetic term of ``basis.grad_T_wrt_basis`` is part of what is compared.
    """
    if is_mixed_precision_enabled():

        def case_factory() -> tuple[Hamiltonian_data, jax.Array, jax.Array]:
            current_ham = _pbc_basis_ham(jastrow)
            return current_ham, *_draw(current_ham)

        def evaluate(case: tuple[Hamiltonian_data, jax.Array, jax.Array]) -> dict[str, np.ndarray]:
            current_ham, current_r_up, current_r_dn = case
            return _pbc_basis_blocks(current_ham, current_r_up, current_r_dn, quantity, analytic=True)

        analytic, full_reference = mixed_vs_full(case_factory, evaluate)
        atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
        for name in _PBC_BASIS_BLOCKS:
            if not np.all(np.isfinite(analytic[name])) or not np.all(np.isfinite(full_reference[name])):
                raise AssertionError("mixed-fidelity parameter blocks must be finite")
            _assert_parameter_derivative_mixed_fidelity(analytic[name], full_reference[name], atol, rtol)
        ham = _pbc_basis_ham(jastrow)
        r_up, r_dn = _draw(ham)
        reference = _pbc_basis_blocks(ham, r_up, r_dn, quantity, analytic=False)
    else:
        ham = _pbc_basis_ham(jastrow)
        r_up, r_dn = _draw(ham)
        atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
        analytic = _pbc_basis_blocks(ham, r_up, r_dn, quantity, analytic=True)
        reference = _pbc_basis_blocks(ham, r_up, r_dn, quantity, analytic=False)
        _assert_pbc_basis_blocks_match(analytic, reference, atol, rtol)

    assert ham.structure_data.pbc_flag
    assert type(ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data).__name__ == "AOs_sphe_pgto_data"
    jd = ham.wavefunction_data.jastrow_data
    assert jd.jastrow_one_body_data is not None and jd.jastrow_two_body_data is not None
    assert (jd.jastrow_three_body_data is not None) == (jastrow == "J1J2J3")
    for name in _PBC_BASIS_BLOCKS:
        assert np.max(np.abs(analytic[name])) > atol, f"inactive periodic pGTO block {name}"
    if quantity == "e_L":
        bare = _pbc_basis_blocks(_without_jastrow(ham), r_up, r_dn, quantity, analytic=False)
        for name in _PBC_BASIS_BLOCKS:
            coupling = np.max(np.abs(reference[name] - bare[name]))
            assert coupling > atol, f"Jastrow-coupled kinetic term of {name} is not live ({coupling:.3e})"


def _assert_pbc_mutated_parameter_blocks(mutated, reference):
    """Compare the original ordered blocks inside one expected-error invocation."""
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    for name in _PBC_BASIS_BLOCKS:
        if not np.all(np.isfinite(mutated[name])) or not np.all(np.isfinite(reference[name])):
            raise AssertionError("mixed-fidelity parameter blocks must be finite")
        assert np.abs(mutated[name] - reference[name]).max() <= atol + rtol * np.abs(reference[name]).max()


@pytest.mark.light
@pytest.mark.parametrize("jastrow", ["J1J2", "J1J2J3"])
def test_pbc_real_pgto_basis_grads_reject_scaled_jastrow_kinetic_term(jastrow, monkeypatch):
    """Negative control: scaling the Jastrow-coupled part of ``basis.grad_T_wrt_basis`` by 1.05 must fail.

    The mutant keeps the Jastrow-free kinetic basis term and multiplies only the
    difference made by the Jastrow. The comparison of
    ``test_pbc_real_pgto_basis_grads_with_jastrow_vs_auto`` must then reject the
    analytical dE_L/dc radial-basis block.
    """
    from jqmc.derivatives import basis as basis_module

    original = basis_module.grad_T_wrt_basis
    outer_mixed = is_mixed_precision_enabled()

    def scaled_jastrow_term(gd, jastrow_data, r_up, r_dn, geminal_inv=None, *, holomorphic=False):
        full = original(gd, jastrow_data, r_up, r_dn, geminal_inv=geminal_inv, holomorphic=holomorphic)
        bare = original(gd, Jastrow_data(), r_up, r_dn, geminal_inv=geminal_inv, holomorphic=holomorphic)
        if not outer_mixed or is_mixed_precision_enabled():
            return jax.tree_util.tree_map(lambda f, b: b + 1.05 * (f - b), full, bare)
        return full

    jax.clear_caches()
    monkeypatch.setattr(basis_module, "grad_T_wrt_basis", scaled_jastrow_term)
    try:
        if is_mixed_precision_enabled():

            def case_factory() -> tuple[Hamiltonian_data, jax.Array, jax.Array]:
                current_ham = _pbc_basis_ham(jastrow)
                return current_ham, *_draw(current_ham)

            def evaluate(case: tuple[Hamiltonian_data, jax.Array, jax.Array]) -> dict[str, np.ndarray]:
                current_ham, current_r_up, current_r_dn = case
                return _pbc_basis_blocks(current_ham, current_r_up, current_r_dn, "e_L", analytic=True)

            mutated, reference = mixed_vs_full(case_factory, evaluate)
        else:
            ham = _pbc_basis_ham(jastrow)
            r_up, r_dn = _draw(ham)
            reference = _pbc_basis_blocks(ham, r_up, r_dn, "e_L", analytic=False)
            mutated = _pbc_basis_blocks(ham, r_up, r_dn, "e_L", analytic=True)
            atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    finally:
        monkeypatch.undo()
        jax.clear_caches()
    if is_mixed_precision_enabled():
        with pytest.raises(AssertionError):
            _assert_pbc_mutated_parameter_blocks(mutated, reference)
    else:
        with pytest.raises(AssertionError, match="periodic pGTO lambda_basis_"):
            _assert_pbc_basis_blocks_match(mutated, reference, atol, rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5", "lithium_k025.h5"])
def test_pbc_pgto_ao_derivs_vs_production(stem):
    """Periodic pgto AO derivative kernels (value/grad/lap/gradlap/Hessian), folded to
    molecular rank, vs production compute_AOs* and autodiff. Foundation for the PBC kinetic
    force/stress. Covers the real (Gamma) and complex (twist) leaves.

    Precision: the derivative kernels are fp64 by construction in both
    modes, so the production references are evaluated with every precision zone at fp64
    (``_fp64_reference_policy``; a no-op in full mode). In mixed mode the AO value
    reference would otherwise be the fp32 ``ao_eval`` output."""
    from jqmc.atomic_orbital import compute_AOs, compute_AOs_grad, compute_AOs_laplacian
    from jqmc.derivatives import pgto_deriv as P

    (_sd, _aos, _mu, _md, gem, _cpd) = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    a = gem.orb_data_up_spin.aos_data
    cart, T = P.sphe_leaf_to_cart(a)
    rng = np.random.default_rng(2)
    r = jnp.asarray(rng.normal(scale=0.8, size=(2, 3)))
    atol, rtol = get_tolerance_min(("ao_eval",), "strict")

    is_complex = P._is_complex_pgto(cart)
    with _fp64_reference_policy():
        value_ref = np.asarray(compute_AOs(a, r))
        gx, gy, gz = compute_AOs_grad(a, r)
        grad_ref = np.stack([np.asarray(gx), np.asarray(gy), np.asarray(gz)], 0)
        lap_ref = np.asarray(compute_AOs_laplacian(a, r))
        # gradlap / Hessian vs autodiff of the production grad / laplacian. The fold is
        # order-agnostic (value/grad/lap already validate it for the complex leaf too), so
        # the autodiff cross-check uses the real leaf (jax.jacrev of a complex AO is non-holomorphic).
        if not is_complex:
            J = np.asarray(jax.vmap(jax.jacrev(lambda x: compute_AOs_laplacian(a, x[None, :])[:, 0]))(r))  # (N,nao,3)

            def g1(x):
                gx_, gy_, gz_ = compute_AOs_grad(a, x[None, :])
                return jnp.stack([gx_[:, 0], gy_[:, 0], gz_[:, 0]], 0)

            Jh = np.asarray(jax.vmap(jax.jacrev(g1))(r))  # (N,3,nao,3)
    assert np.dtype(value_ref.dtype).itemsize == (16 if is_complex else 8)

    np.testing.assert_allclose(np.asarray(T @ P.pgto_value(cart, r)), value_ref, atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.einsum("sc,dcn->dsn", T, P.pgto_grad(cart, r)), grad_ref, atol=atol, rtol=rtol)
    np.testing.assert_allclose(np.asarray(T @ P.pgto_laplacian(cart, r)), lap_ref, atol=atol, rtol=rtol)
    if not is_complex:  # the gradlap / Hessian autodiff cross-check is defined on the real leaf only (above)
        np.testing.assert_allclose(
            np.einsum("sc,dcn->dsn", T, P.pgto_gradlap(cart, r)), np.transpose(J, (2, 1, 0)), atol=atol, rtol=rtol
        )
        np.testing.assert_allclose(
            np.einsum("sc,decn->desn", T, P.pgto_hessian(cart, r)),
            np.transpose(Jh, (3, 1, 2, 0)),
            atol=atol,
            rtol=rtol,
        )


def _api_pbc_fidelity(stem, route, with_j1):
    """Compare analytical policies from one Full physical source and fixed inputs."""
    import tempfile
    from pathlib import Path

    from jqmc._precision import configure
    from jqmc._setting import get_tolerance_mixed_fidelity
    from jqmc.derivatives import force, pbc_stress
    from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

    entry = "mixed" if is_mixed_precision_enabled() else "full"
    with tempfile.TemporaryDirectory(prefix="jqmc_api_pbc_source_") as temporary:
        source = Path(temporary) / "full_physical_source.h5"
        try:
            configure("full")
            if route in ("energy-stress", "log-stress"):
                original = _build_ham(stem)
                up, down = _draw(original)
            else:
                structure, _aos, _mu, _md, gem, cpd = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
                if route == "orbital-stress":
                    j1 = j2 = None
                else:
                    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
                        jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure
                    )
                    j1 = (
                        Jastrow_one_body_data.init_jastrow_one_body_data(
                            jastrow_1b_param=1.0,
                            structure_data=structure,
                            core_electrons=core_electrons_of(cpd, len(structure.positions)),
                            jastrow_1b_type="exp",
                        )
                        if with_j1
                        else None
                    )
                jas = Jastrow_data(
                    jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None, jastrow_dtn_data=None
                )
                original = Hamiltonian_data(
                    structure_data=structure,
                    coulomb_potential_data=cpd,
                    wavefunction_data=Wavefunction_data(jastrow_data=jas, geminal_data=gem),
                )
                up = np.random.default_rng(0).normal(scale=0.7, size=(int(gem.num_electron_up), 3))
                down = np.random.default_rng(1).normal(scale=0.7, size=(int(gem.num_electron_dn), 3))
            fixed_up, fixed_down = np.asarray(up).copy(), np.asarray(down).copy()
            original.save_to_hdf5(source)
        finally:
            configure(entry)

        def factory():
            return Hamiltonian_data.load_from_hdf5(source)

        def evaluate(ham):
            assert ham.structure_data.pbc_flag
            up, down = jnp.asarray(fixed_up), jnp.asarray(fixed_down)
            wf = ham.wavefunction_data
            if route == "kinetic-force":
                assert (wf.jastrow_data.jastrow_one_body_data is not None) == with_j1
                return force.grad_T_wrt_R_withJ(wf.geminal_data, wf.jastrow_data, up, down)
            if route == "kinetic-stress":
                assert wf.jastrow_data.jastrow_one_body_data is not None
                assert wf.jastrow_data.jastrow_two_body_data is not None
                return pbc_stress.stress_eL_kinetic_pbc(ham, up, down)
            if route == "orbital-stress":
                assert wf.jastrow_data.jastrow_one_body_data is None
                assert wf.jastrow_data.jastrow_two_body_data is None
                return pbc_stress.grad_T_wrt_strain_orbital(wf.geminal_data, wf.jastrow_data, up, down)
            if route == "energy-stress":
                assert ham.coulomb_potential_data.ecp_flag
                return derivatives.compute_grad_e_L(ham, up, down, jnp.eye(3), _flags("eps"), analytic=True)["eps"]
            assert route == "log-stress"
            return derivatives.compute_grad_ln_Psi(ham, up, down, _flags("eps"), analytic=True)["eps"]

        mixed, full = mixed_vs_full(factory, evaluate)
    assert np.asarray(mixed).shape == np.asarray(full).shape
    assert np.asarray(full).size > 0
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    for mixed_block, full_block in mixed_fidelity_blocks(mixed, full):
        assert np.max(np.abs(mixed_block - full_block)) <= atol + rtol * np.max(np.abs(full_block))
        changed = out_of_budget_mixed_block(mixed_block, full_block, atol=atol, rtol=rtol)
        with pytest.raises(AssertionError):
            assert np.max(np.abs(changed - full_block)) <= atol + rtol * np.max(np.abs(full_block))


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5", "lithium_k025.h5"])
@pytest.mark.parametrize("with_j1", [False, True])
def test_pbc_kinetic_force_vs_grad(stem, with_j1):
    """Periodic kinetic force dT/dR vs jax.grad of the periodic kinetic energy. Covers the
    det/orbital side (folded pgto AO grad/gradlap/Hessian, force._orbital_side_pbc) and, with
    J1, the e-N J1 side (smooth periodic distance grad/Hessian/laplacian/grad-of-laplacian,
    force._j1_side_pbc; J2 has no R-dependence). Real (Gamma) and complex (twist) leaves."""
    if is_mixed_precision_enabled():
        _api_pbc_fidelity(stem, "kinetic-force", with_j1)
        return
    from jqmc.derivatives import force
    from jqmc.jqmc_mcmc import _inject_active_positions
    from jqmc.wavefunction import compute_kinetic_energy

    (structure_data, _aos, _mu, _md, gem, _cpd) = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    assert structure_data.pbc_flag
    n_atom = len(structure_data.positions)
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure_data
    )
    j1 = (
        Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=core_electrons_of(_cpd, n_atom),
            jastrow_1b_type="exp",
        )
        if with_j1
        else None
    )
    jas = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None, jastrow_dtn_data=None)
    wf = Wavefunction_data(jastrow_data=jas, geminal_data=gem)
    r_up = jnp.asarray(np.random.default_rng(0).normal(scale=0.7, size=(int(gem.num_electron_up), 3)))
    r_dn = jnp.asarray(np.random.default_rng(1).normal(scale=0.7, size=(int(gem.num_electron_dn), 3)))
    ana = force.grad_T_wrt_R_withJ(gem, jas, r_up, r_dn)
    pos = jnp.asarray(gem.orb_data_up_spin.aos_data.structure_data.positions)
    ref = jax.grad(lambda P: jnp.real(compute_kinetic_energy(_inject_active_positions(wf, {"positions": P}), r_up, r_dn)))(pos)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5"])
def test_pbc_ewald_force_vs_grad(stem):
    """Ewald electrostatic HF force d(E_ii + E_ei)/dR (+ d E_ei/dr) vs jax.grad of the
    periodic potential. Self-contained PBC piece (no AO kernels); the full e_L/R also needs
    the periodic kinetic force + periodic ECP force, so e_L/R itself still delegates."""
    from jqmc.coulomb_potential import compute_ewald_potential_ei, compute_ewald_potential_ii
    from jqmc.derivatives import ewald

    (structure_data, _aos, _mu, _md, gem, cpd) = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    assert structure_data.pbc_flag and cpd.ewald_data is not None
    ewd = cpd.ewald_data
    rng = np.random.default_rng(0)
    r_up = jnp.asarray(rng.normal(scale=0.7, size=(int(gem.num_electron_up), 3)))
    r_dn = jnp.asarray(rng.normal(scale=0.7, size=(int(gem.num_electron_dn), 3)))
    pos = jnp.asarray(ewd.structure_data.positions)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")

    ref_ii = jax.grad(
        lambda P: compute_ewald_potential_ii(cpd, ewd.replace(structure_data=ewd.structure_data.replace(positions=P)))
    )(pos)
    np.testing.assert_allclose(np.asarray(ewald.grad_Eii_wrt_R(cpd, ewd)), np.asarray(ref_ii), atol=atol, rtol=rtol)

    def Eei(P, ru, rd):
        cd2 = cpd.replace(structure_data=cpd.structure_data.replace(positions=P))
        ewd2 = ewd.replace(structure_data=ewd.structure_data.replace(positions=P))
        eu, ed = compute_ewald_potential_ei(cd2, ru, rd, ewd2)
        return jnp.sum(eu) + jnp.sum(ed)

    np.testing.assert_allclose(
        np.asarray(ewald.grad_Eei_wrt_R(cpd, ewd, r_up, r_dn)),
        np.asarray(jax.grad(lambda P: Eei(P, r_up, r_dn))(pos)),
        atol=atol,
        rtol=rtol,
    )
    ref_r = np.concatenate(
        [
            np.asarray(jax.grad(lambda r: Eei(pos, r, r_dn))(r_up)),
            np.asarray(jax.grad(lambda r: Eei(pos, r_up, r))(r_dn)),
        ],
        axis=0,
    )
    np.testing.assert_allclose(np.asarray(ewald.grad_Eei_wrt_r(cpd, ewd, r_up, r_dn)), ref_r, atol=atol, rtol=rtol)

    # Ewald STRESS d(E_ii + E_ei)/d eps vs jax.grad under affine strain (electrons, ions, cell)
    sdw = ewd.structure_data
    va, vb, vc = jnp.asarray(sdw.vec_a), jnp.asarray(sdw.vec_b), jnp.asarray(sdw.vec_c)

    def _strain(sdx, F):
        return sdx.replace(positions=jnp.asarray(sdx.positions) @ F.T, vec_a=F @ va, vec_b=F @ vb, vec_c=F @ vc)

    def fii(eps):
        F = jnp.eye(3) + eps
        return compute_ewald_potential_ii(cpd, ewd.replace(structure_data=_strain(sdw, F)))

    np.testing.assert_allclose(
        np.asarray(ewald.grad_Eii_wrt_strain(cpd, ewd)), np.asarray(jax.grad(fii)(jnp.zeros((3, 3)))), atol=atol, rtol=rtol
    )

    def fei(eps):
        F = jnp.eye(3) + eps
        cd2 = cpd.replace(structure_data=_strain(cpd.structure_data, F))
        eu, ed = compute_ewald_potential_ei(cd2, r_up @ F.T, r_dn @ F.T, ewd.replace(structure_data=_strain(sdw, F)))
        return jnp.sum(eu) + jnp.sum(ed)

    np.testing.assert_allclose(
        np.asarray(ewald.grad_Eei_wrt_strain(cpd, ewd, r_up, r_dn)),
        np.asarray(jax.grad(fei)(jnp.zeros((3, 3)))),
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium_k025.h5"])
def test_pbc_twist_o_k_vs_auto(stem):
    """Complex-Bloch (twisted, k != 0) periodic analytic grads vs _auto across ALL axes: O_k =
    d ln|Psi|/dc, drift d ln|Psi|/dr, Pulay force d ln|Psi|/dR, and e_L/{c,r,R,eps}. O_k is
    holomorphic (D17: no conjugation), so the same analytic kernels are exact for the complex
    orbitals; lnPsi grads are Re(grad). e_L grads keep the ECP wf-ratio / det Pulay stress
    holomorphic and take Re of the assembled PRODUCT at the end (Re(ratio*O) != Re(ratio)*Re(O)).
    sphe-pgto basis frozen (mask _BASIS_OFF)."""
    if is_mixed_precision_enabled():
        _api_remainder_fidelity("twist-ecp", stem=stem)
        return
    ham = _build_ham(stem)
    r_up, r_dn = _draw(ham)
    assert ham.structure_data.pbc_flag
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    wf = ham.wavefunction_data
    # O_k (ln_Psi/c): production-variational (non-basis) blocks
    a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags("c"), analytic=True)["c"]
    b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="c", mask=_ref_mask(stem))
    da = wf.collect_param_grads(a)
    db = wf.collect_param_grads(b)
    assert set(da) == set(db)
    # sphe-pgto basis is frozen (not optimized); compare the production-variational blocks.
    # (For complex MOs the imaginary part leaks into the frozen-basis reconstruction, which is
    # irrelevant since basis-opt is unsupported for pgto -- see SUPPORTED_BASIS_OPT_AO.)
    for k in da:
        if "basis" in k:
            continue
        np.testing.assert_allclose(np.asarray(da[k]), np.asarray(db[k]), atol=atol, rtol=rtol)
    # drift (ln_Psi/r) = Re(grad ln Psi); Pulay force (ln_Psi/R); periodic Pulay stress (ln_Psi/eps)
    for wrt in ("r", "R", "eps"):
        ax = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags(wrt), analytic=True)[wrt]
        bx = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt=wrt, mask=_ref_mask(stem))
        for la, lb in zip(jax.tree_util.tree_leaves(ax), jax.tree_util.tree_leaves(bx)):
            np.testing.assert_allclose(np.asarray(la), np.asarray(lb), atol=atol, rtol=rtol)
    # e_L/c: holomorphic kinetic dT/dc + ECP O_k(r'), reduced to Re (e_L is real-part optimized)
    ae = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("c"), analytic=True)["c"]
    be = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="c", mask=_ref_mask(stem))
    dae = wf.collect_param_grads(ae)
    dbe = wf.collect_param_grads(be)
    for k in dae:
        if "basis" in k:
            continue
        np.testing.assert_allclose(np.asarray(dae[k]), np.asarray(dbe[k]), atol=atol, rtol=rtol)
    # e_L/{r,R,eps}: periodic kinetic drift/force/stress + Ewald (e-e/e-i/ii) + periodic ECP, with
    # the ECP wf-ratio and det Pulay stress kept holomorphic and Re taken on the assembled product.
    for wrt in ("r", "R", "eps"):
        ax = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags(wrt), analytic=True)[wrt]
        bx = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt=wrt, mask=_ref_mask(stem))
        for la, lb in zip(jax.tree_util.tree_leaves(ax), jax.tree_util.tree_leaves(bx)):
            np.testing.assert_allclose(np.asarray(la), np.asarray(lb), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium_k025.h5", "lithium_k025_cart.h5"])
def test_pbc_twist_basis_opt_now_analytic(stem):
    """Complex-twist single-leaf cart/sphe-pGTO radial basis derivatives match same-mode autodiff.

    Both committed lithium twist ccECP orbital families exercise the analytical
    exponent/coefficient blocks, alongside the existing real-Gamma sibling.
    Complex-twist single-leaf radial basis-opt (opt_lambda_basis_*) is ANALYTIC
    The O_k value basis is holomorphic (Re-projected for the direct ln_Psi/c), the
    kinetic dT/dc basis is a real energy derivative that matches _auto, and the ECP non-local
    dV_nl/dc pairs the complex holomorphic basis O_k with the complex (project_real=False) per-grid
    weight -- capturing the Im(wf_ratio).Im(O_k) cross term the old real-projected weight dropped.
    The analytic basis-opt c-axis path must equal the autodiff oracle and return a finite gradient for
    both quantities on the twist fixture AND the real-Gamma sibling (lithium.h5). Full precision uses
    the strict tolerance; mixed precision uses the centrally defined medium tolerance because the
    autodiff oracle differentiates the fp32 ``ao_eval`` forward while the analytic radial derivative
    remains fp64. This keeps both precision modes blocking without a fixture-specific tolerance. (The
    complex-twist COMPOSITE basis-opt still routes to _auto -- no fixture verifies its kinetic basis
    grad -- but that is exercised elsewhere, not here.) The a==b oracle for the twist lambda-basis grad
    lives in test_pbc_twist_ae_cart_lambda_basis_opt_analytic_vs_auto + the ECP twist basis rows."""
    if is_mixed_precision_enabled():
        _api_remainder_fidelity("twist-radial", stem=stem)
        return
    flags = {"opt_lambda_basis_exp": True, "opt_lambda_basis_coeff": True}
    twist = _build_ham(stem)
    assert twist.structure_data.pbc_flag
    real = _build_ham("lithium.h5")
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), level)
    for fixture_name, ham in ((stem, twist), ("lithium.h5", real)):
        r_up, r_dn = _draw(ham)
        for quantity in ("ln_Psi", "e_L"):
            if quantity == "ln_Psi":
                out = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)["c"]
                reference = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=False)["c"]
            else:
                out = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=True)["c"]
                reference = derivatives.compute_grad_e_L(
                    ham,
                    r_up,
                    r_dn,
                    jnp.eye(3),
                    flags,
                    analytic=False,
                )["c"]
            actual_blocks = _active_param_grads(ham.wavefunction_data, out, flags)
            reference_blocks = _active_param_grads(ham.wavefunction_data, reference, flags)
            expected_blocks = {"lambda_basis_exp", "lambda_basis_coeff"}
            assert set(actual_blocks) == expected_blocks
            assert set(reference_blocks) == expected_blocks
            for block_name in sorted(expected_blocks):
                actual = np.asarray(actual_blocks[block_name])
                expected = np.asarray(reference_blocks[block_name])
                assert np.all(np.isfinite(actual)), f"{quantity}/c {block_name} analytic grad not finite"
                assert np.all(np.isfinite(expected)), f"{quantity}/c {block_name} autodiff grad not finite"
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=atol,
                    rtol=rtol,
                    err_msg=f"fixture {fixture_name}; quantity {quantity}; block {block_name}",
                )


# --------------------------------------------------------------------------- #
# Complex-twist ALL-ELECTRON (twist-AE) analytic-vs-_auto coverage. The committed
# lithium_k025.h5 twist rows above are ECP-only (bare 1/r e-i replaced by ccECP
# local + non-local); NONE exercise the AE electron-ion branch (bare 1/r Coulomb +
# Ewald e-i) under a complex Bloch orbital. The QE H-ae_k025 fixture (built at test
# time by _build_twist_ae_ham) closes that: hydrogen all-electron at k=(1/4,1/4,1/4),
# complex cart-pGTO leaf, no ECP. Every analytic axis it exercises PASSES vs _auto.
# The CANONICAL grad-comparison zone set (matches _grad_fidelity_harness._assert_axis_allclose):
# the weakest Precision Zone an analytic-vs-_auto grad can cross.  Full uses
# the central strict policy and mixed uses the central medium policy; both
# compare analytical and autodiff routes under the same active precision.
_TWIST_AE_ZONES = ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic")


@pytest.mark.light
@requires_h_ae
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
@pytest.mark.parametrize("wrt", ["c", "r", "R", "eps"])
def test_pbc_twist_ae_analytic_vs_auto(quantity, wrt):
    """Complex-twist ALL-ELECTRON periodic analytic grads vs _auto, across every axis the
    analytic path supports for this basis. AE means: bare 1/r electron-ion + Ewald (NO ECP
    local/non-local branch) under complex Bloch orbitals. Mirrors test_pbc_twist_o_k_vs_auto
    (its ECP sibling) but on the AE Hamiltonian:

    * ``r``/``R``/``eps`` (drift / Pulay force / periodic stress): full ln_Psi and e_L axis
      grad -- the AE e_L/{r,R,eps} routes the periodic kinetic drift/force/stress + Ewald
      (e-e/e-i/i-i) with the det Pulay stress kept holomorphic and Re taken on the product;
      the ECP-mesh term is simply absent (ecp_flag False).
    * ``c`` (O_k = d ln|Psi|/dc and dE_L/dc): the cart-pGTO complex-twist
      radial BASIS-opt mode (opt_lambda_basis_exp/coeff with opt_geminal off),
      which is analytic for this leaf -- see
      test_pbc_twist_ae_cart_lambda_basis_opt_analytic_vs_auto.
    """
    if is_mixed_precision_enabled():
        _api_remainder_fidelity("twist-ae", quantity=quantity, wrt=wrt)
        return
    ham = _build_twist_ae_ham()
    r_up, r_dn = _draw(ham)
    assert ham.structure_data.pbc_flag
    assert not ham.coulomb_potential_data.ecp_flag  # all-electron: bare 1/r e-i + Ewald, no ECP branch
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(_TWIST_AE_ZONES, level)

    if wrt == "c":
        # cart-pgto complex-twist basis-opt is analytic. This is
        # the radial-basis mode, so direct-MO optimization is disabled: the two
        # blocks are exercised independently here, with joint coverage elsewhere.
        flags = {k: True for k in derivatives._PARAM_FLAG_KEYS}
        flags["opt_geminal"] = False
        if quantity == "ln_Psi":
            a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)["c"]
            b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="c", mask=flags)
        else:
            a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=True)["c"]
            b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="c", mask=flags)
        wf = ham.wavefunction_data
        da = _active_param_grads(wf, a, flags)
        db = _active_param_grads(wf, b, flags)
        assert set(da) == set(db)
        assert "lambda_basis_exp" in da and "lambda_basis_coeff" in da  # cart-complex basis now emitted
        for k in da:
            np.testing.assert_allclose(np.asarray(da[k]), np.asarray(db[k]), atol=atol, rtol=rtol, err_msg=f"block {k}")
        return

    if quantity == "ln_Psi":
        a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags(wrt), analytic=True)[wrt]
        b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt=wrt, mask=None)
    else:
        a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags(wrt), analytic=True)[wrt]
        b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt=wrt, mask=None)
    for la, lb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(np.asarray(la), np.asarray(lb), atol=atol, rtol=rtol)


@pytest.mark.light
@requires_h_ae
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_pbc_twist_ae_cart_lambda_basis_opt_analytic_vs_auto(quantity):
    """Complex-twist single-leaf cart-pGTO radial LAMBDA basis-opt is
    ANALYTIC and matches _auto. The QE H all-electron k=(1/4,1/4,1/4) projection yields an
    ``AOs_cart_pgto_complex_data`` geminal leaf; ``opt_lambda_basis_exp``/``opt_lambda_basis_coeff``
    are served by the SAME shared radial kernels as the sphe-complex leaf
    (basis._value/kinetic_general_leaf via ``_cart_prim_value``/``_prim_sens`` + the pgto image unfold
    + the wrap-restore phase). O_k value (ln_Psi/c) is the holomorphic d ln|det|/d radial Re-projected
    for the direct block; kinetic dT/dc (e_L/c) keeps the full complex (holomorphic) grad ln D. This
    fixture is ALL-ELECTRON, so the e_L/c basis channel covers KINETIC dT/dc only (no ECP non-local
    dV_nl/dc -- the cart-complex ECP-nl channel has no committed fixture; its holomorphic seam is
    verified on the sphe-complex lithium_k025 ECP fixture and the cart leaf shares the identical
    kernel/seam). A ``basis grads off by O(10)`` symptom means the assemble gate silently skips the
    block (-> zero = the full missing gradient), NOT a kernel math error; the leaf is registered in BOTH
    ``_capability`` and ``assemble._geminal_has_opt_basis``."""
    if is_mixed_precision_enabled():
        _api_remainder_fidelity("twist-ae-radial", quantity=quantity)
        return
    ham = _build_twist_ae_ham()
    from jqmc.atomic_orbital import AOs_cart_pgto_complex_data

    assert isinstance(ham.wavefunction_data.geminal_data.orb_data_up_spin.aos_data, AOs_cart_pgto_complex_data)
    r_up, r_dn = _draw(ham)
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(_TWIST_AE_ZONES, level)
    if quantity == "ln_Psi":
        a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags("c"), analytic=True)["c"]
        b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="c", mask=None)
    else:
        a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("c"), analytic=True)["c"]
        b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="c", mask=None)
    wf = ham.wavefunction_data
    da = wf.collect_param_grads(a)
    db = wf.collect_param_grads(b)
    assert "lambda_basis_exp" in da and "lambda_basis_coeff" in da
    for k in ("lambda_basis_exp", "lambda_basis_coeff"):
        np.testing.assert_allclose(np.asarray(da[k]), np.asarray(db[k]), atol=atol, rtol=rtol, err_msg=f"block {k}")


@pytest.mark.light
@requires_h_ae
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_pbc_twist_ae_cart_j3_basis_opt_analytic_vs_auto(quantity):
    """Complex-twist single-leaf cart-pGTO J3-ORBITAL radial basis-opt is
    ANALYTIC and matches _auto. An active three-body Jastrow whose orbital leaf is the
    ``AOs_cart_pgto_complex_data`` leaf: the J3 O_k VALUE derivative (ln_Psi/c) is a REAL energy
    derivative (basis.grad_J3_wrt_basis Re-projects, since the three-body Jastrow consumes Re(orb)),
    and the J3 KINETIC dT/dc basis (e_L/c, basis.grad_T_wrt_J3_basis) Re-projects the kinetic cotangent
    (cot = Re(grad ln D) + grad J). Both ``opt_J3_basis_exp``/``opt_J3_basis_coeff`` match the _auto
    twin to machine precision."""
    if is_mixed_precision_enabled():
        _api_remainder_fidelity("twist-ae-j3", quantity=quantity)
        return
    ham = _build_twist_ae_ham(kind="pgto", j3="ao")
    r_up, r_dn = _draw(ham)
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(_TWIST_AE_ZONES, level)
    if quantity == "ln_Psi":
        a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags("c"), analytic=True)["c"]
        b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="c", mask=None)
    else:
        a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("c"), analytic=True)["c"]
        b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="c", mask=None)
    wf = ham.wavefunction_data
    da = wf.collect_param_grads(a)
    db = wf.collect_param_grads(b)
    assert "j3_basis_exp" in da and "j3_basis_coeff" in da
    for k in ("j3_basis_exp", "j3_basis_coeff"):
        np.testing.assert_allclose(np.asarray(da[k]), np.asarray(db[k]), atol=atol, rtol=rtol, err_msg=f"block {k}")


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5", "lithium_k025.h5"])
def test_pbc_kinetic_stress_vs_grad(stem):
    """Full periodic kinetic stress dT/d eps (orbital + Jastrow, J1+J2) vs jax.grad of the
    strained kinetic energy under jqmc's _apply_strain convention (strains the geminal AO and J1
    cells but NOT the J2 e-e cell). J1 = smooth-distance lattice-strain (_smooth_strain); J2 =
    fixed-cell affine virial (_smooth_strain_fixed_cell via force._smooth_dist_kinematics).
    Real (Gamma) + complex (twist)."""
    if is_mixed_precision_enabled():
        _api_pbc_fidelity(stem, "kinetic-stress", True)
        return
    from jqmc.derivatives import pbc_stress
    from jqmc.wavefunction import compute_kinetic_energy

    (structure_data, _aos, _mu, _md, gem, cpd) = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    assert structure_data.pbc_flag
    n_atom = len(structure_data.positions)
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure_data
    )
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=core_electrons_of(cpd, n_atom),
        jastrow_1b_type="exp",
    )
    jas = Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=None, jastrow_dtn_data=None)
    wf = Wavefunction_data(jastrow_data=jas, geminal_data=gem)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=cpd, wavefunction_data=wf)
    r_up = jnp.asarray(np.random.default_rng(0).normal(scale=0.7, size=(int(gem.num_electron_up), 3)))
    r_dn = jnp.asarray(np.random.default_rng(1).normal(scale=0.7, size=(int(gem.num_electron_dn), 3)))
    ana = pbc_stress.stress_eL_kinetic_pbc(ham, r_up, r_dn)

    def f(eps):
        F = jnp.eye(3) + eps
        return jnp.real(compute_kinetic_energy(wf._apply_strain(F), r_up @ F.T, r_dn @ F.T))

    ref = jax.grad(f)(jnp.zeros((3, 3)))
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5", "lithium_k025.h5"])
def test_pbc_kinetic_stress_orbital_jnone_vs_grad(stem):
    """Periodic kinetic stress dT/d eps -- orbital/det side (no Jastrow) vs jax.grad of the
    strained kinetic energy. The kinetic VJP cotangents (bar_orb/bar_lap/bar_gou) contracted with
    the strain response of the AO value/laplacian/gradient fields (folded-grad virial + kernel-
    rank image-T, pbc_stress.grad_T_wrt_strain_orbital). Real (Gamma) + complex (twist). The
    Jastrow side (smooth-distance strain) is added separately."""
    if is_mixed_precision_enabled():
        _api_pbc_fidelity(stem, "orbital-stress", False)
        return
    from jqmc.derivatives import pbc_stress
    from jqmc.structure import _strain_structure_data
    from jqmc.wavefunction import compute_kinetic_energy

    (structure_data, _aos, _mu, _md, gem, _cpd) = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    assert structure_data.pbc_flag
    jas = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=None
    )
    r_up = jnp.asarray(np.random.default_rng(0).normal(scale=0.7, size=(int(gem.num_electron_up), 3)))
    r_dn = jnp.asarray(np.random.default_rng(1).normal(scale=0.7, size=(int(gem.num_electron_dn), 3)))
    ana = pbc_stress.grad_T_wrt_strain_orbital(gem, jas, r_up, r_dn)

    def f(eps):
        F = jnp.eye(3) + eps

        def s(orb):
            return orb.replace(
                aos_data=orb.aos_data.replace(structure_data=_strain_structure_data(orb.aos_data.structure_data, F))
            )

        gd2 = gem.replace(orb_data_up_spin=s(gem.orb_data_up_spin), orb_data_dn_spin=s(gem.orb_data_dn_spin))
        return jnp.real(compute_kinetic_energy(Wavefunction_data(jastrow_data=jas, geminal_data=gd2), r_up @ F.T, r_dn @ F.T))

    ref = jax.grad(f)(jnp.zeros((3, 3)))
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    np.testing.assert_allclose(np.asarray(ana), np.asarray(ref), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5"])
def test_pbc_real_stress_eL_vs_auto(stem):
    """Real Gamma-point periodic HF stress d e_L/d eps, dispatched. = periodic kinetic stress
    (orbital + Jastrow) + Ewald (ii+ei+ee) stress + ECP stress (local affine virial + non-local:
    the geometry strain dV_grid/deps plus the wf-ratio d ln ratio/deps = stress_lnPsi_pbc at the
    mesh configs minus base, plus the non-affine mesh-radius correction). vs _auto. Completes the
    eps axis for real PBC (all four axes c/r/R/eps are analytic for real Gamma periodic)."""
    if is_mixed_precision_enabled():
        _api_pbc_fidelity(stem, "energy-stress", None)
        return
    ham = _build_ham(stem)
    r_up, r_dn = _draw(ham)
    assert ham.structure_data.pbc_flag and ham.coulomb_potential_data.ecp_flag
    a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("eps"), analytic=True)["eps"]
    b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="eps", mask=_ref_mask(stem))
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5"])
def test_pbc_real_stress_lnpsi_vs_auto(stem):
    """Real Gamma-point periodic Pulay stress d ln|Psi| / d eps, dispatched. The det image-T
    correction (term2: kernel-rank AO grad weighted by T_n) on top of the folded-grad affine
    virial, plus the Jastrow cell-strain terms: J1 = smooth-distance lattice-length stress
    (_dd_dstrain, since jqmc strains the J1 cell), J2 = affine virial f'(d) grad_d (x) disp
    (jqmc does NOT strain the J2 e-e cell). vs _auto (production strained forward)."""
    if is_mixed_precision_enabled():
        _api_pbc_fidelity(stem, "log-stress", None)
        return
    ham = _build_ham(stem)
    r_up, r_dn = _draw(ham)
    assert ham.structure_data.pbc_flag
    a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags("eps"), analytic=True)["eps"]
    b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="eps", mask=_ref_mask(stem))
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


def _native_api_first5_dtn_case(stem):
    from jqmc.jastrow_factor import Jastrow_DTN_data

    structure_data, _aos, _mu, _md, gem, cpd = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    z_eff, pseudized = _dtn_species_policy(structure_data, cpd)
    dtn = Jastrow_DTN_data.build(structure_data=structure_data, z_eff_per_species=z_eff, pseudized_per_species=pseudized)
    jastrow = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=dtn
    )
    wf = Wavefunction_data(jastrow_data=jastrow, geminal_data=gem)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=cpd, wavefunction_data=wf)
    r_up, r_dn = _draw(ham)
    return (ham, r_up, r_dn)


def _assert_native_api_parameter_fidelity(mixed_block, full_block, atol, rtol):
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max()


def _native_api_first5_case(stem, variant):
    if variant == "dtn-parameters":
        return _native_api_first5_dtn_case(stem)
    ham = _build_ham(stem)
    up, down = _draw(ham)
    if variant == "periodic-ecp":
        assert ham.structure_data.pbc_flag and ham.coulomb_potential_data.ecp_flag
    elif variant == "periodic":
        assert ham.structure_data.pbc_flag
    else:
        assert variant == "ecp"
        assert ham.coulomb_potential_data.ecp_flag
    return ham, up, down


def _native_api_first5_evaluate(case, quantity, axis, variant):
    ham, up, down = case
    if quantity == "ln_Psi":
        result = derivatives.compute_grad_ln_Psi(ham, up, down, _flags(axis), analytic=True)[axis]
    else:
        assert quantity == "e_L"
        result = derivatives.compute_grad_e_L(ham, up, down, jnp.eye(3), _flags(axis), analytic=True)[axis]
    if axis == "c":
        result = ham.wavefunction_data.collect_param_grads(result)
        assert {"j3_poly_alpha", "j3_poly_beta", "j3_poly_gamma"} <= set(result)
    return {"blocks": result, "up": up, "down": down}


def _compare_native_api_first5(stem, quantity, axis, variant):
    mixed, full = mixed_vs_full(
        lambda: _native_api_first5_case(stem, variant),
        lambda case: _native_api_first5_evaluate(case, quantity, axis, variant),
    )
    assert np.array_equal(mixed["up"], full["up"]) and np.array_equal(mixed["down"], full["down"])
    blocks = mixed_fidelity_blocks(mixed["blocks"], full["blocks"])
    if variant == "dtn-parameters":
        assert set(mixed["blocks"]) == set(full["blocks"])
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    for mixed_block, full_block in blocks:
        _assert_native_api_parameter_fidelity(mixed_block, full_block, atol, rtol)
    faulty = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _assert_native_api_parameter_fidelity(faulty, blocks[0][1], atol, rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5"])
def test_pbc_real_eL_dr_vs_auto(stem):
    """Real Gamma-point periodic drift d e_L/dr = dT/dr + d(E_ee+E_ei)/dr + dV_ecp/dr, dispatched.
    The periodic kinetic drift (folded pgto AO orbital side + periodic-distance J1/J2 drift), the
    Ewald e-e/e-i electron gradients, and the ECP electron gradient. vs _auto."""
    if is_mixed_precision_enabled():
        _compare_native_api_first5(stem, "e_L", "r", "periodic-ecp")
        return
    ham = _build_ham(stem)
    r_up, r_dn = _draw(ham)
    assert ham.structure_data.pbc_flag and ham.coulomb_potential_data.ecp_flag
    a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("r"), analytic=True)["r"]
    b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="r", mask=_ref_mask(stem))
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    al = np.concatenate([np.asarray(x) for x in a], 0)
    bl = np.concatenate([np.asarray(x) for x in jax.tree_util.tree_leaves(b)], 0)
    np.testing.assert_allclose(al, bl, atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5"])
def test_pbc_real_force_eL_vs_auto(stem):
    """Real Gamma-point periodic HF force d e_L/dR = dT/dR + dV_ewald/dR + dV_ecp/dR, dispatched.
    The periodic kinetic force (folded pgto AO higher derivatives + smooth-distance J1 side), the
    Ewald HF force, and the periodic ECP force (min-image via _get_min_dist_rel_R_cart_jnp -- the
    same image selection as jqmc's forward, NOT the numpy _debug twin) vs _auto. Complex-Bloch
    (twist) ECP force needs the conj/Re convention and still delegates."""
    if is_mixed_precision_enabled():
        _compare_native_api_first5(stem, "e_L", "R", "periodic-ecp")
        return
    ham = _build_ham(stem)
    r_up, r_dn = _draw(ham)
    assert ham.structure_data.pbc_flag and ham.coulomb_potential_data.ecp_flag
    a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("R"), analytic=True)["R"]
    b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="R", mask=_ref_mask(stem))
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5"])
def test_pbc_real_force_lnpsi_vs_auto(stem):
    """Real Gamma-point periodic Pulay force d ln|Psi| / dR. The Bloch translation identity
    d phi/dR = -d phi/dr holds at Gamma (image phase depends on the integer image index, not
    R); the periodic det grad folds lattice-image AOs to molecular rank and scatters by the
    ragged home atom (_pgto_home_atom); the J1 e-N force uses the periodic distance gradient
    (grad_J1_wrt_R PBC branch). Twisted (complex-Bloch) PBC delegates to _auto. (e_L/R adds
    the Ewald HF lattice-sum force + periodic kinetic force and still delegates.)"""
    if is_mixed_precision_enabled():
        _compare_native_api_first5(stem, "ln_Psi", "R", "periodic")
        return
    ham = _build_ham(stem)
    r_up, r_dn = _draw(ham)
    assert ham.structure_data.pbc_flag
    a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags("R"), analytic=True)["R"]
    b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="R", mask=_ref_mask(stem))
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["N_ae_ccpvdz_cart.h5"])
def test_dtn_e_L_dc_vs_auto(stem):
    """DTN dE_L/dc = dT/dc for the u/chi/f channels (kinetic; dV_local/dc = 0). The
    f-channel (e-e-N, ``gamma``) kinetic is the 3-body cross-term derivation. DTN built
    at test time from the trexio structure; AE system (no ECP mesh) so the eager DTN O_k
    is not vmapped. Open-shell N (5+2) exercises the f-channel with many electrons. The
    DTN ``*_indep`` grads surface through collect_param_grads as ``j3_poly_{alpha,beta,gamma}``."""
    if is_mixed_precision_enabled():
        _compare_native_api_first5(stem, "e_L", "c", "dtn-parameters")
        return
    from jqmc.jastrow_factor import Jastrow_DTN_data

    (structure_data, _aos, _mu, _md, gem, cpd) = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    z_eff, pseudized = _dtn_species_policy(structure_data, cpd)
    dtn = Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=z_eff,
        pseudized_per_species=pseudized,
    )
    jastrow = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=dtn
    )
    wf = Wavefunction_data(jastrow_data=jastrow, geminal_data=gem)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=cpd, wavefunction_data=wf)
    r_up, r_dn = _draw(ham)
    a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("c"), analytic=True)["c"]
    b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="c", mask=None)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    da = wf.collect_param_grads(a)
    db = wf.collect_param_grads(b)
    assert {"j3_poly_alpha", "j3_poly_beta", "j3_poly_gamma"} <= set(da)
    assert set(da) == set(db)
    for k in da:
        np.testing.assert_allclose(np.asarray(da[k]), np.asarray(db[k]), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["N2_ecp_ccpvtz_cart.h5"])
def test_ecp_force_vs_auto(stem):
    """ECP nuclear force d e_L/dR = dT/dR + dV_bare/dR + dV_ecp/dR (local all-pairs +
    non-local NN=1 wf-ratio + mesh-motion + projector geometry). Eager dispatch vs _auto."""
    if is_mixed_precision_enabled():
        _compare_native_api_first5(stem, "e_L", "R", "ecp")
        return
    ham = _build_ham(stem)
    r_up, r_dn = _draw(ham)
    assert ham.coulomb_potential_data.ecp_flag
    a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("R"), analytic=True)["R"]
    b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="R", mask=_ref_mask(stem))
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


def _mixed_ecp_ae_force_assert_bound(mixed_block, full_block, atol, rtol, label):
    """Check one electron or nuclear ECP force block against the central bound."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max(), label


def _mixed_ecp_ae_force_fidelity(ecp_atom):
    """Rebuild both nearest-ECP layouts under independent analytical policies."""
    from jqmc.derivatives.ecp_force import grad_V_ecp_wrt_r, grad_V_ecp_wrt_R
    from tests._precision_fidelity import mixed_fidelity_blocks, out_of_budget_mixed_block

    def factory():
        cp, wf = mixed_ecp_ae_system(ecp_atom)
        ham = Hamiltonian_data(structure_data=cp.structure_data, coulomb_potential_data=cp, wavefunction_data=wf)
        return ham, MIXED_AE_R_UP.copy(), np.empty((0, 3))

    def evaluate(case):
        ham, r_up, r_down = case
        electron_force = grad_V_ecp_wrt_r(ham, r_up, r_down, RT=np.eye(3), Nv=6)
        nuclear_force = grad_V_ecp_wrt_R(ham, r_up, r_down, RT=np.eye(3), Nv=6)
        assert electron_force.shape == MIXED_AE_R_UP.shape
        assert nuclear_force.shape == MIXED_AE_POSITIONS.shape
        return {"r": electron_force, "R": nuclear_force, "electrons": r_up, "nuclei": ham.structure_data.positions}

    mixed, full = mixed_vs_full(factory, evaluate)
    for name in ("electrons", "nuclei"):
        assert np.array_equal(mixed[name], full[name]), f"policy-dependent mixed ECP/AE physical input: {name}"
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    checked = []
    for axis in ("r", "R"):
        for mixed_block, full_block in mixed_fidelity_blocks(mixed[axis], full[axis]):
            _mixed_ecp_ae_force_assert_bound(mixed_block, full_block, atol, rtol, f"ECP nucleus {ecp_atom} force {axis}")
            checked.append((mixed_block, full_block))
    assert checked, "mixed ECP/AE force comparison would be vacuous"
    faulty = out_of_budget_mixed_block(*checked[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _mixed_ecp_ae_force_assert_bound(faulty, checked[0][1], atol, rtol, "ECP/AE excess-error control")


@pytest.mark.light
@pytest.mark.parametrize("ecp_atom", [0, 1], ids=("trailing_ae", "leading_ae"))
def test_ecp_force_mixed_ecp_all_electron_matches_independent_derivative(ecp_atom):
    """XLA ECP electron and nuclear forces of a mixed ECP/AE molecule equal an independent derivative.

    Force route. The system and the independent V_ecp are
    in ``tests/_mixed_ecp_ae_reference.py`` (shared with the XLA/Rust non-local
    energy detector in ``tests/test_rs_kernel/test_ecp_local.py``). This test
    calls only the JAX force kernels, so it lives in the XLA suite.

    The production ECP force setup (``ecp_force._ecp_nl_common``, shared by
    ``grad_V_ecp_wrt_r``, ``grad_V_ecp_wrt_R`` and the strain derivatives) places
    each electron's NN=1 non-local projector on its nearest ECP nucleus; an
    all-electron nucleus never hosts it. Electron 0 is nearer the all-electron
    nucleus in both layouts, so ranking all nuclei would drop or misplace its
    projector. Both public forces (fixed quadrature rotation, Nv=6) are compared
    with ``jax.grad`` of ``mixed_ecp_ae_potential_jnp``. The reference is the
    ideal float64 value and retains its strict mathematical check in Full. Mixed
    compares independently rebuilt Full/Mixed analytical forces under the central fidelity budget.
    """
    if is_mixed_precision_enabled():
        _mixed_ecp_ae_force_fidelity(ecp_atom)
        return
    from jqmc.derivatives.ecp_force import grad_V_ecp_wrt_r, grad_V_ecp_wrt_R

    cp, wf = mixed_ecp_ae_system(ecp_atom)
    ham = Hamiltonian_data(structure_data=cp.structure_data, coulomb_potential_data=cp, wavefunction_data=wf)
    r_up = MIXED_AE_R_UP.copy()
    r_dn = np.empty((0, 3))
    r_ref = jnp.asarray(r_up, dtype=jnp.float64)
    nuclei_ref = jnp.asarray(MIXED_AE_POSITIONS, dtype=jnp.float64)
    expected_r = np.asarray(jax.grad(mixed_ecp_ae_potential_jnp, 0)(r_ref, nuclei_ref, ecp_atom))
    expected_R = np.asarray(jax.grad(mixed_ecp_ae_potential_jnp, 1)(r_ref, nuclei_ref, ecp_atom))
    got_r = np.asarray(grad_V_ecp_wrt_r(ham, r_up, r_dn, RT=np.eye(3), Nv=6))
    got_R = np.asarray(grad_V_ecp_wrt_R(ham, r_up, r_dn, RT=np.eye(3), Nv=6))
    atol, rtol = get_tolerance_min(("coulomb", "ao_eval", "ao_grad_lap", "det_eval", "det_grad_lap", "det_ratio"), "strict")
    for got, expected in ((got_r, expected_r), (got_R, expected_R)):
        assert got.shape == expected.shape
        assert np.all(np.isfinite(got))
        np.testing.assert_allclose(got, expected, atol=atol, rtol=rtol)


# Exotic periodic bases (plane-wave / B-spline / pgto(+)PW composite) have NO committed trexio fixture
# -- they require QE plane-wave output. They are therefore assembled AT TEST TIME from the committed
# QE ``*.save`` outputs (tests/qe_example_files) through the jqmc projection pipeline
# (``load_qe_reference`` -> ``project_pw_to_*`` -> ``build_hamiltonian_from_projection``), exactly as
# the light test_integration_qe_periodic.py assembly does; the
# hamiltonian is discarded after the compare (NO derived hamiltonian_data.h5 is committed, so these
# always run from the committed QE outputs). Their c-axis (param-grad O_k/c +
# dE_L/dc) is BASIS-AGNOSTIC: it differentiates det/T w.r.t. the PARAMS using jqmc's FORWARD orbital
# value/grad/lap fields, never an analytic AO position-derivative.
_QE_DIR = os.fspath(qe_example_files_root())
_LI_GAMMA_SAVE = os.path.join(_QE_DIR, "Li_gamma", "out", "li.save")  # real-Gamma ccECP Li
_LI_GAMMA_UPF = os.path.join(_LI_GAMMA_SAVE, "Li.ccECP.upf")
_LI_K025_SAVE = os.path.join(_QE_DIR, "Li_k025", "out", "li_k025.save")  # complex twist (k=1/4) ccECP Li
_LI_K025_UPF = os.path.join(_LI_K025_SAVE, "Li.ccECP.upf")

# The bench hamiltonian tree is retained ONLY for test_composite_pgtopw_twist_basis_opt_stays_auto
# (the complex-twist composite basis-opt closure owns that fixture); every other exotic test below
# builds from the committed QE saves above.
_BENCH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "bench", "hamiltonians")


def _window_shells(shells_per_atom, lo, hi):
    """Keep only pGTO primitives with ``lo <= exponent <= hi`` (mirrors
    _grad_fidelity_harness._window_shells): the PW block already spans the diffuse lower edge exactly
    and the UPF ladder's ultra-tight primitive is unresolved on the cell grid, so the window conditions
    the composite overlap."""
    out = []
    for shells in shells_per_atom:
        kept = []
        for s in shells:
            pairs = [(e, c) for e, c in zip(s.exponents, s.coefficients) if lo <= e <= hi]
            if pairs:
                kept.append(
                    dataclasses.replace(s, exponents=tuple(p[0] for p in pairs), coefficients=tuple(p[1] for p in pairs))
                )
        out.append(kept)
    return out


@functools.lru_cache(maxsize=None)
def _build_pw_ham(save_dir, upf_path, element, g_cut_fraction):
    """Pure plane-wave (``kind='pw'``) periodic Hamiltonian from a committed QE save, truncated to
    ``|G| <= g_cut_fraction * g_max`` for a light basis (~33 PW AOs). ``element=None`` -> all-electron
    (bare 1/r e-i + Ewald); otherwise the full analytic ccECP is parsed from ``upf_path``. A Gamma save
    yields a real ``AOs_plane_wave_data`` leaf; a non-Gamma save a complex ``AOs_plane_wave_complex_data``
    leaf. Built once per (args) process via ``lru_cache``."""
    from jqmc.pw_pgto_projection import project_pw_to_pw
    from jqmc.qe_to_jqmc import build_hamiltonian_from_projection
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import parse_gaussinfo

    ref = load_qe_reference(save_dir)
    g_max = float(np.sqrt(2.0 * ref.ecutwfc))
    proj = project_pw_to_pw(ref, g_cut=g_cut_fraction * g_max)
    ecp = {element: parse_gaussinfo(upf_path)} if element is not None else None
    return build_hamiltonian_from_projection(
        ref, proj, kind="pw", j1_param=None, j2_param=1.0, j2_type="pade", ecp_by_element=ecp
    )


@functools.lru_cache(maxsize=None)
def _build_composite_pgtopw_ham(save_dir, upf_path, element):
    """Real-Gamma pgto(+)PW COMPOSITE (``kind='pgto+pw'``) from a committed QE save -- the windowed
    even-tempered pGTO ladder (exponent window [1, 8]) plus the low-|G| (<= 0.25 g_max) PW block, the
    same construction as _grad_fidelity_harness._qe_composite_pieces. Full analytic ccECP; yields an
    ``AOs_composite_data`` leaf with pGTO + plane-wave children."""
    from jqmc.pw_pgto_projection import project_pw_to_mixed
    from jqmc.qe_to_jqmc import build_hamiltonian_from_projection
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import UPFExponentInit, parse_gaussinfo

    ref = load_qe_reference(save_dir)
    g_max = float(np.sqrt(2.0 * ref.ecutwfc))
    strat = UPFExponentInit({element.lower(): upf_path}, g_target=g_max)
    shells = _window_shells([strat.shells_for_element(lbl) for lbl in ref.atom_labels], 1.0, 8.0)
    proj = project_pw_to_mixed(ref, shells, g_cut=0.25 * g_max, drop_diffuse=False)
    return build_hamiltonian_from_projection(
        ref,
        proj,
        kind="pgto+pw",
        j1_param=None,
        j2_param=1.0,
        j2_type="pade",
        ecp_by_element={element: parse_gaussinfo(upf_path)},
    )


@functools.lru_cache(maxsize=None)
def _build_bspline_ham(save_dir, upf_path, element):
    """Periodic B-spline (``kind='bspline'``) Hamiltonian from a committed QE save (per-band whole-cell
    quintic spline, ``mesh_factor=1.0``), the same assembly as
    test_integration_qe_periodic._build_ham_bspline. Yields an ``AOs_bspline_data`` leaf. Full analytic
    ccECP."""
    from jqmc.pw_pgto_projection import project_pw_to_spline
    from jqmc.qe_to_jqmc import build_hamiltonian_from_projection
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import parse_gaussinfo

    ref = load_qe_reference(save_dir)
    proj = project_pw_to_spline(ref, mesh_factor=1.0, order=5)
    return build_hamiltonian_from_projection(
        ref,
        proj,
        kind="bspline",
        j1_param=None,
        j2_param=1.0,
        j2_type="pade",
        ecp_by_element={element: parse_gaussinfo(upf_path)},
    )


# Pure plane-wave FULL-axis cases: real Gamma + complex twist (both ccECP Li, ~33 PW AOs each). The
# former exercises the real ``AOs_plane_wave_data`` leaf, the latter the complex
# ``AOs_plane_wave_complex_data`` leaf (the twist kinetic/stress kernels).
_EXOTIC_PW_CASES = [
    pytest.param(_LI_GAMMA_SAVE, _LI_GAMMA_UPF, "Li", 0.25, id="pw_real_gamma_ecp"),
    pytest.param(_LI_K025_SAVE, _LI_K025_UPF, "Li", 0.25, id="pw_twist_ecp"),
]


def _draw_physical(ham, seed=0, scale=0.15):
    """Electrons near random nuclei -- a well-conditioned config for a large determinant (random
    scale-0.7 makes the 96-electron det near-singular, blowing up _geminal_reverse's plain inv)."""
    gd = ham.wavefunction_data.geminal_data
    R = np.asarray(gd.orb_data_up_spin.aos_data.structure_data.positions)
    nu, nd = int(gd.num_electron_up), int(gd.num_electron_dn)
    rng = np.random.default_rng(seed)
    ru = jnp.asarray(R[rng.integers(0, len(R), nu)] + rng.normal(scale=scale, size=(nu, 3)))
    rd = jnp.asarray(R[rng.integers(0, len(R), nd)] + rng.normal(scale=scale, size=(nd, 3)))
    return ru, rd


def _native_api_exotic_case(variant, save_dir, upf_path, element, g_cut_fraction):
    if variant == "pw":
        from jqmc.derivatives._capability import jit_ao_serveable

        _build_pw_ham.cache_clear()
        ham = _build_pw_ham(save_dir, upf_path, element, g_cut_fraction)
        assert jit_ao_serveable(ham)
    elif variant == "composite-gamma":
        _build_composite_pgtopw_ham.cache_clear()
        ham = _build_composite_pgtopw_ham(_LI_GAMMA_SAVE, _LI_GAMMA_UPF, "Li")
        _assert_pgtopw_composite(ham.wavefunction_data.geminal_data)
    elif variant == "composite-twist":
        _twist_ae_pieces.cache_clear()
        _build_twist_ae_ham_cached.cache_clear()
        ham = _build_twist_ae_ham(kind="pgto+pw", j3="mo")
        _assert_pgtopw_composite(ham.wavefunction_data.geminal_data)
    else:
        assert variant == "composite-bspline-j3"
        ham = _build_ham_j3_composite_bspline()
        aos = ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data.aos_data
        assert type(aos).__name__ == "AOs_composite_data"
        child_kinds = [type(child).__name__ for child in aos.children]
        assert any("pgto" in kind for kind in child_kinds) and any("bspline" in kind for kind in child_kinds)
    return ham, *_draw(ham)


def _native_api_exotic_evaluate(case, variant, quantity):
    ham, up, down = case
    wf = ham.wavefunction_data
    blocks = {}
    labels = []
    if variant == "pw":
        from jqmc.derivatives import bridge

        rub, rdb, rtb = up[None], down[None], jnp.eye(3)[None]
        for axis in ("r", "R", "eps"):
            blocks["eager/ln_Psi/" + axis] = derivatives.compute_grad_ln_Psi(ham, up, down, _flags(axis), analytic=True)[axis]
            blocks["eager/e_L/" + axis] = derivatives.compute_grad_e_L(ham, up, down, jnp.eye(3), _flags(axis), analytic=True)[
                axis
            ]
            if axis != "r":
                outP = bridge.batched_grad_ln_Psi(wrt=axis)(ham, rub, rdb)
            else:
                outP = derivatives.compute_grad_ln_Psi(ham, up, down, _flags(axis), analytic=True)[axis]
            # Preserve the original ln-Psi bridge completion without inventing
            # a new numerical comparison that the old test did not assert.
            jax.block_until_ready(outP)
            outL = bridge.batched_grad_e_L(wrt=axis)(ham, rub, rdb, rtb)
            blocks["bridge/e_L/" + axis] = jax.tree_util.tree_map(lambda value: value[0], outL)
    elif variant == "composite-gamma":
        mask = {
            "opt_geminal": False,
            "opt_lambda_basis_exp": True,
            "opt_lambda_basis_coeff": True,
            "opt_J3_basis_exp": False,
            "opt_J3_basis_coeff": False,
        }
        flags = _flags("c", mask)
        for selected_quantity in ("ln_Psi", "e_L"):
            if selected_quantity == "ln_Psi":
                result = derivatives.compute_grad_ln_Psi(ham, up, down, flags, analytic=True)["c"]
            else:
                result = derivatives.compute_grad_e_L(ham, up, down, jnp.eye(3), flags, analytic=True)["c"]
            result = _active_param_grads(wf, result, flags)
            assert {"lambda_basis_exp", "lambda_basis_coeff"} <= set(result)
            blocks[selected_quantity] = result
    else:
        if quantity == "ln_Psi":
            result = derivatives.compute_grad_ln_Psi(ham, up, down, _flags("c"), analytic=True)["c"]
        else:
            assert quantity == "e_L"
            result = derivatives.compute_grad_e_L(ham, up, down, jnp.eye(3), _flags("c"), analytic=True)["c"]
        if variant == "composite-twist":
            towers = (
                ("gem_up", result.geminal_data.orb_data_up_spin.aos_data),
                ("gem_dn", result.geminal_data.orb_data_dn_spin.aos_data),
                ("j3", result.jastrow_data.jastrow_three_body_data.orb_data.aos_data),
            )
        else:
            towers = (("j3", result.jastrow_data.jastrow_three_body_data.orb_data.aos_data),)
        for tag, aos in towers:
            compared = 0
            for index, child in enumerate(aos.children):
                if "gto" not in type(child).__name__:
                    if variant == "composite-bspline-j3":
                        assert getattr(child, "radial", None) is None
                    continue
                for field in ("exponents", "coefficients"):
                    label = f"{tag}/{index}/{field}"
                    blocks[label] = np.asarray(getattr(child.radial, field))
                    labels.append(label)
                compared += 1
            assert compared > 0, f"{tag}: no GTO child compared -- vacuous"
        if variant == "composite-twist":
            assert labels == [
                f"{tag}/0/{field}" for tag in ("gem_up", "gem_dn", "j3") for field in ("exponents", "coefficients")
            ]
    if variant == "pw":
        assert set(blocks) == {
            "eager/ln_Psi/r",
            "eager/ln_Psi/R",
            "eager/ln_Psi/eps",
            "eager/e_L/r",
            "eager/e_L/R",
            "eager/e_L/eps",
            "bridge/e_L/r",
            "bridge/e_L/R",
            "bridge/e_L/eps",
        }
    return {"blocks": blocks, "up": up, "down": down, "labels": labels}


def _assert_native_api_exotic_fidelity(mixed_block, full_block, atol, rtol):
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max()


def _compare_native_api_exotic(variant, quantity=None, save_dir=None, upf_path=None, element=None, g_cut_fraction=None):
    mixed, full = mixed_vs_full(
        lambda: _native_api_exotic_case(variant, save_dir, upf_path, element, g_cut_fraction),
        lambda case: _native_api_exotic_evaluate(case, variant, quantity),
    )
    assert np.array_equal(mixed["up"], full["up"]) and np.array_equal(mixed["down"], full["down"])
    assert set(mixed["blocks"]) == set(full["blocks"])
    assert mixed["labels"] == full["labels"]
    blocks = mixed_fidelity_blocks(mixed["blocks"], full["blocks"])
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    for mixed_block, full_block in blocks:
        _assert_native_api_exotic_fidelity(mixed_block, full_block, atol, rtol)
    faulty = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _assert_native_api_exotic_fidelity(faulty, blocks[0][1], atol, rtol)


@pytest.mark.light
@pytest.mark.parametrize("save_dir, upf_path, element, g_cut_fraction", _EXOTIC_PW_CASES)
def test_exotic_pw_full_axes_vs_auto(save_dir, upf_path, element, g_cut_fraction):
    """Plane-wave (pure ``kind='pw'``, real Gamma AND complex twist) is FULL-axis analytic: the
    position/strain kernels (ao.pw_* unified real/complex + the PW kinetic stress + the dAO/dR=0 /
    affine-invariant-value simplifications) make ln_Psi/{r,R,eps} + e_L/{r,R,eps} exact vs _auto, not
    just the c-axis. ALSO jit-serveable: the kernels are pure jnp and compile through the production
    bridge (``jit(vmap(...))`` with the hamiltonian as a TRACED arg -- geometry via the _cell_jnp /
    _positions_cart_jnp twins, sphe->cart hoist skipped). The Hamiltonian is built from the committed QE
    save via the pure-PW side of the projection pipeline (``project_pw_to_pw``), discarded after."""
    if is_mixed_precision_enabled():
        _compare_native_api_exotic("pw", save_dir=save_dir, upf_path=upf_path, element=element, g_cut_fraction=g_cut_fraction)
        return
    from jqmc.derivatives import bridge
    from jqmc.derivatives._capability import jit_ao_serveable

    ham = _build_pw_ham(save_dir, upf_path, element, g_cut_fraction)
    assert jit_ao_serveable(ham)  # periodic/exotic is served through the jit bridge
    r_up, r_dn = _draw(ham)
    RT = jnp.eye(3)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    rub, rdb, rtb = r_up[None], r_dn[None], RT[None]  # nw=1 walker batch for the bridge
    for wrt in ("r", "R", "eps"):
        for a, b in (
            (
                derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags(wrt), analytic=True)[wrt],
                _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt=wrt, mask=_BASIS_OFF),
            ),
            (
                derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, _flags(wrt), analytic=True)[wrt],
                _auto.compute_grad_e_L(ham, r_up, r_dn, RT, wrt=wrt, mask=_BASIS_OFF),
            ),
        ):
            for la, lb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
                np.testing.assert_allclose(np.asarray(la), np.asarray(lb), atol=atol, rtol=rtol)
        # the actual "ported" proof: the SAME analytic compiles under jit(vmap) over a walker batch
        # (hamiltonian on a None axis = traced) and matches the eager/_auto result un-batched.
        if wrt != "r":
            outP = bridge.batched_grad_ln_Psi(wrt=wrt)(ham, rub, rdb)
            outL = bridge.batched_grad_e_L(wrt=wrt)(ham, rub, rdb, rtb)
        else:
            outP = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags(wrt), analytic=True)[
                wrt
            ]  # ln_Psi/r not a bridge axis
            outL = bridge.batched_grad_e_L(wrt=wrt)(ham, rub, rdb, rtb)
        ref = _auto.compute_grad_e_L(ham, r_up, r_dn, RT, wrt=wrt, mask=_BASIS_OFF)
        a0 = jax.tree_util.tree_map(lambda x: x[0], outL)
        for la, lb in zip(jax.tree_util.tree_leaves(a0), jax.tree_util.tree_leaves(ref)):
            np.testing.assert_allclose(np.asarray(la), np.asarray(lb), atol=atol, rtol=rtol)


def _assert_pgtopw_composite(gd):
    """Sanity: the fixture geminal is an MO-over-composite with pgto + PW children."""
    aos = gd.orb_data_up_spin.aos_data if type(gd.orb_data_up_spin).__name__ == "MOs_data" else gd.orb_data_up_spin
    assert type(aos).__name__ == "AOs_composite_data"
    child_kinds = [type(c).__name__ for c in aos.children]
    assert any("pgto" in k for k in child_kinds) and any("plane_wave" in k for k in child_kinds)
    return child_kinds


@pytest.mark.light
def test_composite_pgtopw_basis_opt_vs_auto():
    """Composite (pgto(+)PW) basis-parameter optimization is ANALYTIC on the c-axis for the REAL-Gamma
    composite: the geminal routes the radial exp/coeff grads per child (only the (p)gto child
    contributes; the PW child is a zero no-basis-param leaf), so O_k = d ln|Psi| / d radial.{exp,coeff}
    AND dE_L / d radial.{exp,coeff} match _auto to machine precision. Turn basis-opt ON via the mask
    and compare the concatenated up+dn ``lambda_basis_exp`` / ``lambda_basis_coeff`` blocks
    (collect_param_grads) vs _auto with the SAME mask -- a==b for the composite with basis active."""
    if is_mixed_precision_enabled():
        _compare_native_api_exotic("composite-gamma")
        return
    ham = _build_composite_pgtopw_ham(_LI_GAMMA_SAVE, _LI_GAMMA_UPF, "Li")  # real Gamma pgto(+)PW composite, ECP
    wf = ham.wavefunction_data
    gd = wf.geminal_data
    _assert_pgtopw_composite(gd)

    mask = {
        "opt_geminal": False,
        "opt_lambda_basis_exp": True,
        "opt_lambda_basis_coeff": True,
        "opt_J3_basis_exp": False,
        "opt_J3_basis_coeff": False,
    }
    c_flags = _flags("c", mask)
    r_up, r_dn = _draw(ham)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")

    for quantity in ("ln_Psi", "e_L"):
        # the composite basis-opt collection is analytic: the analytic path runs and matches _auto.
        if quantity == "ln_Psi":
            a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, c_flags, analytic=True)["c"]
            b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="c", mask=mask)
        else:
            a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), c_flags, analytic=True)["c"]
            b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="c", mask=mask)
        da = _active_param_grads(wf, a, c_flags)
        db = _active_param_grads(wf, b, c_flags)
        # the composite basis blocks are now emitted (up+dn concatenated per child, PW child empty)
        assert set(da) == set(db)
        assert "lambda_basis_exp" in da and "lambda_basis_coeff" in da
        for k in da:
            np.testing.assert_allclose(np.asarray(da[k]), np.asarray(db[k]), atol=atol, rtol=rtol)


@pytest.mark.light
@requires_h_ae
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_composite_pgtopw_twist_basis_opt_analytic_vs_auto(quantity):
    """COMPLEX-twist pgto(+)PW COMPOSITE radial basis-opt is
    ANALYTIC and matches _auto. The twist composite
    (a complex cart-pGTO child + a complex PW child) is built AT TEST TIME from the committed QE
    H-ae_k025 all-electron save (windowed pGTO [1,8] (+) 0.25*g_max PW -- the 8-set composite recipe,
    which keeps the pGTO/PW overlap conditioned), so NO derived bench h5 is needed.

    Both the geminal LAMBDA basis (opt_lambda_basis_*) and an MO-backed J3 orbital's basis
    (opt_J3_basis_*) are verified: the shared per-child general contraction kernels
    (basis._basis_contract_*_general iterating pgto_deriv.orbital_leaves) route the complex cart-pGTO
    child's O_k + kinetic dT/dc through the SAME holomorphic/real-cotangent path as the single-leaf
    twist, while the complex PW child carries no radial params (zero basis grad). The flat-dict
    reducer publishes per-child J3 blocks; this test additionally compares the RAW grad-pytree
    radial leaves of every GTO child (as the 8-set test_composite_j3_basis_grads_vs_auto does)."""
    if is_mixed_precision_enabled():
        _compare_native_api_exotic("composite-twist", quantity)
        return
    ham = _build_twist_ae_ham(kind="pgto+pw", j3="mo")
    _assert_pgtopw_composite(ham.wavefunction_data.geminal_data)
    r_up, r_dn = _draw(ham)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    if quantity == "ln_Psi":
        a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags("c"), analytic=True)["c"]
        b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="c", mask=None)
    else:
        a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("c"), analytic=True)["c"]
        b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="c", mask=None)

    compared_fields = []

    def _cmp_children(ga, gb, tag):
        atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
        compared = 0
        for i, (ca, cb) in enumerate(zip(ga.children, gb.children)):
            if "gto" not in type(ca).__name__:  # PW child: no radial params
                continue
            for fld in ("exponents", "coefficients"):
                comparison_label = f"{tag}/{i}/{fld}"
                np.testing.assert_allclose(
                    np.asarray(getattr(ca.radial, fld)),
                    np.asarray(getattr(cb.radial, fld)),
                    atol=atol,
                    rtol=rtol,
                    err_msg=f"{quantity} {comparison_label}",
                )
                compared_fields.append(comparison_label)
            compared += 1
        assert compared > 0, f"{tag}: no GTO child compared -- vacuous"

    # LAMBDA basis (geminal AO leaf, up + dn) and J3 basis (MO -> composite).
    _cmp_children(a.geminal_data.orb_data_up_spin.aos_data, b.geminal_data.orb_data_up_spin.aos_data, "gem_up")
    _cmp_children(a.geminal_data.orb_data_dn_spin.aos_data, b.geminal_data.orb_data_dn_spin.aos_data, "gem_dn")
    _cmp_children(
        a.jastrow_data.jastrow_three_body_data.orb_data.aos_data,
        b.jastrow_data.jastrow_three_body_data.orb_data.aos_data,
        "j3",
    )
    assert compared_fields == [
        f"{tag}/0/{field}" for tag in ("gem_up", "gem_dn", "j3") for field in ("exponents", "coefficients")
    ]


def _native_api_bspline_stress_case():
    from jqmc.derivatives._capability import jit_ao_serveable

    _build_bspline_ham.cache_clear()
    ham = _build_bspline_ham(_LI_GAMMA_SAVE, _LI_GAMMA_UPF, "Li")
    assert jit_ao_serveable(ham)
    return ham, *_draw_physical(ham)


def _native_api_bspline_stress_evaluate(case):
    from jqmc.derivatives import pbc_stress

    ham, up, down = case
    return {"stress": np.asarray(pbc_stress.stress_eL_kinetic_pbc(ham, up, down)), "up": up, "down": down}


def _assert_native_api_bspline_stress_fidelity(mixed_block, full_block, atol, rtol):
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max()


def _compare_native_api_bspline_stress():
    mixed, full = mixed_vs_full(_native_api_bspline_stress_case, _native_api_bspline_stress_evaluate)
    assert np.array_equal(mixed["up"], full["up"]) and np.array_equal(mixed["down"], full["down"])
    blocks = mixed_fidelity_blocks(mixed["stress"], full["stress"])
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    for mixed_block, full_block in blocks:
        _assert_native_api_bspline_stress_fidelity(mixed_block, full_block, atol, rtol)
    faulty = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _assert_native_api_bspline_stress_fidelity(faulty, blocks[0][1], atol, rtol)


@pytest.mark.light
def test_exotic_bspline_kinetic_vs_grad():
    """B-spline FULL-axis pieces, verified in ISOLATION (the full e_L/eps with the ECP-stress mesh is
    too slow on a B-spline fixture):
    the closed-form kinetic kernels (grad/grad-of-lap/Hessian) vs the forward + jacrev, and the
    metric-strain kinetic stress vs jax.grad of the strained kinetic energy. Physical (near-nuclei)
    config. Gate: B-spline is full-axis analytic AND jit-serveable (pure jnp; bridge-compiles).
    The ``AOs_bspline_data`` leaf is built from the committed QE save via ``project_pw_to_spline`` (the
    same per-band whole-cell quintic spline assembly as test_integration_qe_periodic), discarded after."""
    from jqmc.derivatives import ao as A
    from jqmc.derivatives import pbc_stress
    from jqmc.derivatives._capability import jit_ao_serveable
    from jqmc.wavefunction import compute_kinetic_energy

    ham = _build_bspline_ham(_LI_GAMMA_SAVE, _LI_GAMMA_UPF, "Li")
    assert jit_ao_serveable(ham)  # B-spline is served through the jit bridge
    wf = ham.wavefunction_data
    aos = wf.geminal_data.orb_data_up_spin.aos_data
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    ru, rd = _draw_physical(ham)
    # kinetic kernels vs forward + jacrev (few electrons -> cheap)
    rk = ru[:4]
    grad, gradlap, hess = A._bspline_kinetic_fields(aos, rk)
    gx, gy, gz = aos._compute_AOs_grad(rk)
    np.testing.assert_allclose(grad, np.stack([np.asarray(gx), np.asarray(gy), np.asarray(gz)], 0), atol=atol, rtol=rtol)
    Jg = np.asarray(jax.jacrev(lambda r: jnp.stack(aos._compute_AOs_grad(r), 0))(rk))
    np.testing.assert_allclose(hess, np.einsum("ckiid->cdki", Jg), atol=atol, rtol=rtol)
    Jl = np.asarray(jax.jacrev(lambda r: aos._compute_AOs_laplacian(r))(rk))
    np.testing.assert_allclose(gradlap, np.einsum("kiid->dki", Jl), atol=atol, rtol=rtol)
    # kinetic stress (metric strain) vs jax.grad of the strained kinetic energy
    if is_mixed_precision_enabled():
        _compare_native_api_bspline_stress()
        return
    ana = np.asarray(pbc_stress.stress_eL_kinetic_pbc(ham, ru, rd))
    ref = np.asarray(
        jax.grad(
            lambda eps: jnp.real(
                compute_kinetic_energy(wf._apply_strain(jnp.eye(3) + eps), ru @ (jnp.eye(3) + eps).T, rd @ (jnp.eye(3) + eps).T)
            )
        )(jnp.zeros((3, 3)))
    )
    np.testing.assert_allclose(ana, ref, rtol=rtol, atol=atol * np.max(np.abs(ref)))


def _native_api_next_dtn_case(stem):
    from jqmc.jastrow_factor import Jastrow_DTN_data

    structure_data, _aos, _mu, _md, gem, cpd = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    z_eff, pseudized = _dtn_species_policy(structure_data, cpd)
    dtn = Jastrow_DTN_data.build(structure_data=structure_data, z_eff_per_species=z_eff, pseudized_per_species=pseudized)
    jastrow = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=dtn
    )
    wf = Wavefunction_data(jastrow_data=jastrow, geminal_data=gem)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=cpd, wavefunction_data=wf)
    r_up, r_dn = _draw(ham)
    return (ham, r_up, r_dn)


def _native_api_next_dtn_ok_case():
    from jqmc.derivatives import dtn
    from jqmc.jastrow_factor import Jastrow_DTN_data

    structure_data, _aos, _mu, _md, gem, cpd = read_trexio_file(os.path.join(_TREXIO, "lithium.h5"), store_tuple=True)
    d = Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=(1.0,),
        pseudized_per_species=(True,),
        use_u=True,
        use_chi=True,
        use_f=True,
        cutoff_smoothness=4,
        n_order_u=4,
        n_order_chi=4,
        n_order_eN=2,
        n_order_ee=2,
        L_u=None,
        L_chi=None,
        L_f=None,
        random_init=True,
        seed=13,
    )
    jastrow = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=d
    )
    wf = Wavefunction_data(jastrow_data=jastrow, geminal_data=gem)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=cpd, wavefunction_data=wf)
    assert ham.structure_data.pbc_flag and d is not None
    r_up, r_dn = _draw(ham)
    return (ham, r_up, r_dn)


def _assert_native_api_next_fidelity(mixed_block, full_block, atol, rtol):
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max()


def _native_api_next_case(stem, variant):
    if variant == "dtn-primitive":
        return _native_api_next_dtn_ok_case()
    if variant == "dtn-force":
        return _native_api_next_dtn_case(stem)
    if variant in ("dtn-closed", "dtn-stress"):
        ham = _build_pbc_dtn_ham(stem)
        assert ham.structure_data.pbc_flag and ham.wavefunction_data.jastrow_data.jastrow_dtn_data is not None
    else:
        assert variant in ("j3-molecular", "j3-periodic")
        ham = _build_ham_j3(stem)
        if variant == "j3-periodic":
            assert ham.structure_data.pbc_flag
    return ham, *_draw(ham)


def _native_api_next_evaluate(case, variant, selected_axis):
    ham, up, down = case
    wf = ham.wavefunction_data
    if variant == "dtn-primitive":
        from jqmc.derivatives import dtn

        blocks = dtn.grad_dtn_wrt_param(wf.jastrow_data.jastrow_dtn_data, up, down)
        assert {"alpha_indep", "beta_indep", "gamma_indep"} <= set(blocks)
        assert all(np.asarray(blocks[key]).size for key in ("alpha_indep", "beta_indep", "gamma_indep"))
        return {"blocks": blocks, "up": up, "down": down}
    tasks = {
        "j3-molecular": (("ln_Psi", "c"), ("e_L", "c"), ("e_L", "r"), ("e_L", "R"), ("ln_Psi", "R")),
        "j3-periodic": (("e_L", "R"), ("e_L", "eps"), ("ln_Psi", "R"), ("ln_Psi", "eps")),
        "dtn-force": (("ln_Psi", "R"), ("e_L", "R")),
        "dtn-closed": (("e_L", selected_axis),),
        "dtn-stress": (("ln_Psi", "eps"), ("e_L", "eps")),
    }[variant]
    blocks = {}
    for quantity, axis in tasks:
        flags = _flags(axis)
        if quantity == "e_L":
            result = derivatives.compute_grad_e_L(ham, up, down, jnp.eye(3), flags, analytic=True)[axis]
        else:
            result = derivatives.compute_grad_ln_Psi(ham, up, down, flags, analytic=True)[axis]
        if axis == "c":
            if variant == "j3-molecular":
                result = _active_param_grads(wf, result, flags)
                assert {"j3_matrix", "j3_basis_exp", "j3_basis_coeff"} <= set(result)
            else:
                d = result.jastrow_data.jastrow_dtn_data
                result = {key: getattr(d, key) for key in ("alpha_indep", "beta_indep", "gamma_indep")}
        blocks[quantity + "/" + axis] = result
    if variant == "dtn-stress":
        from jqmc.derivatives import dtn as dtn_module

        # Keep the original nontrivial comoving-cutoff liveness predicate at both policies.
        policy = float(dtn_module.dtn_cutoff_policy_value(wf.jastrow_data.jastrow_dtn_data, up, down))
        assert abs(policy) > 1.0e-3
    return {"blocks": blocks, "up": up, "down": down}


def _compare_native_api_next(stem, variant, selected_axis=None):
    mixed, full = mixed_vs_full(
        lambda: _native_api_next_case(stem, variant),
        lambda case: _native_api_next_evaluate(case, variant, selected_axis),
    )
    assert np.array_equal(mixed["up"], full["up"]) and np.array_equal(mixed["down"], full["down"])
    assert set(mixed["blocks"]) == set(full["blocks"])
    blocks = mixed_fidelity_blocks(mixed["blocks"], full["blocks"])
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    for mixed_block, full_block in blocks:
        _assert_native_api_next_fidelity(mixed_block, full_block, atol, rtol)
    faulty = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _assert_native_api_next_fidelity(faulty, blocks[0][1], atol, rtol)


@pytest.mark.light
def test_dtn_okc_pbc_vs_auto():
    """PBC DTN O_k = d ln_Psi/dc (u/chi/f channels) under the min-image distance convention
    (jastrow_factor._dtn_scalar_dist_with_derivs: PBC = minimum-image Euclidean). dtn._md applies the same ((u@inv - round)@cell) min-image, so the
    feature sums match the forward. vs _auto on a real-Gamma PBC DTN system: the committed periodic
    ccECP-Li sphe-pgto trexio (lithium.h5) wrapped with a DTN-only Jastrow built at test time (u/chi/f
    channels, L=None auto-resolves the cutoff from the WS-inscribed radius so the hard cutoff pins a
    unique min-image), exactly as the DTN tests in test_gfmc_force_analytic._build_ham_dtn."""
    if is_mixed_precision_enabled():
        _compare_native_api_next(None, "dtn-primitive")
        return
    from jqmc.derivatives import dtn
    from jqmc.jastrow_factor import Jastrow_DTN_data

    (structure_data, _aos, _mu, _md, gem, cpd) = read_trexio_file(os.path.join(_TREXIO, "lithium.h5"), store_tuple=True)
    d = Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=(1.0,),  # Li ccECP: Z=3, two core electrons
        pseudized_per_species=(True,),
        use_u=True,
        use_chi=True,
        use_f=True,
        cutoff_smoothness=4,
        n_order_u=4,
        n_order_chi=4,
        n_order_eN=2,
        n_order_ee=2,
        L_u=None,
        L_chi=None,
        L_f=None,
        random_init=True,
        seed=13,
    )
    jastrow = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=d
    )
    wf = Wavefunction_data(jastrow_data=jastrow, geminal_data=gem)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=cpd, wavefunction_data=wf)
    assert ham.structure_data.pbc_flag and d is not None
    r_up, r_dn = _draw(ham)
    ana = dtn.grad_dtn_wrt_param(d, r_up, r_dn)
    b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="c", mask=_BASIS_OFF).jastrow_data.jastrow_dtn_data
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    for k in ("alpha_indep", "beta_indep", "gamma_indep"):
        av, bv = ana.get(k), getattr(b, k, None)
        if av is not None and bv is not None and np.asarray(bv).size:
            np.testing.assert_allclose(np.asarray(av), np.asarray(bv), atol=atol, rtol=rtol)


# --------------------------------------------------------------------------- #
# Coverage backfill: active-J3 and DTN force/stress. These analytic
# paths were only exercised in throwaway dev scripts; the tests below give them a
# committed analytic-vs-_auto (or vs jax.grad) regression, matching the idioms above.
# --------------------------------------------------------------------------- #


def _build_ham_j3(stem: str, seed: int = 5) -> Hamiltonian_data:
    """``_build_ham`` twin with an ACTIVE three-body Jastrow built on the fixture's AO leaf.

    A random symmetric coupling block is used (j_matrix[:, :-1] symmetric bosonic + a final
    linear column) so every J3 kinetic / force / drift / O_k term is genuinely nonzero.
    """
    from jqmc.jastrow_factor import Jastrow_three_body_data

    (structure_data, aos_data, _mu, _md, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        os.path.join(_TREXIO, stem), store_tuple=True
    )
    rng = np.random.default_rng(seed)
    jm = rng.normal(size=(aos_data.num_ao, aos_data.num_ao + 1)) * 0.1
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)  # bosonic coupling block must be symmetric
    j3 = Jastrow_three_body_data(orb_data=aos_data, j_matrix=jm)
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure_data
    )
    n_atom = len(structure_data.positions)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=core_electrons_of(coulomb_potential_data, n_atom),
        jastrow_1b_type="exp",
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=j3, jastrow_dtn_data=None
    )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    return Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["H2_ae_ccpvdz_cart.h5", "N_ae_ccpvdz_cart.h5", "H2_ae_ccpvdz_sphe.h5"])
def test_j3_molecular_axes_vs_auto(stem):
    """Active three-body Jastrow (molecular, cart/sphe-GTO AE) analytic vs _auto over all axes.

    The J3 coupling-block O_k (``j3_matrix``), the J3 kinetic (e_L/c), and the J3 force/drift
    (e_L/R, e_L/r, ln_Psi/R) are analytic and match _auto to machine precision. The J3 *basis*-opt
    blocks ``j3_basis_exp`` / ``j3_basis_coeff`` (the radial exponents/coeffs of the J3 orbital
    basis, opt_J3_basis) are ALSO analytic for a single-leaf cart/sphe GTO J3 orbital
    (assemble.grad_J3_wrt_basis for O_k, grad_T_wrt_J3_basis for the kinetic dT/dc), so the c-axis
    compare includes them. Geminal radial-basis optimization is a separate mode
    and is covered by its dedicated tests."""
    if is_mixed_precision_enabled():
        _compare_native_api_next(stem, "j3-molecular")
        return
    ham = _build_ham_j3(stem)
    r_up, r_dn = _draw(ham)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")

    # c-axis: variational-parameter blocks (O_k for ln_Psi, dE_L/dc for e_L), j3_basis INCLUDED.
    c_flags = _flags("c")
    for quantity in ("ln_Psi", "e_L"):
        if quantity == "ln_Psi":
            a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, c_flags, analytic=True)["c"]
            b = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, c_flags, analytic=False)["c"]
        else:
            a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), c_flags, analytic=True)["c"]
            b = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), c_flags, analytic=False)["c"]
        wf = ham.wavefunction_data
        da = _active_param_grads(wf, a, c_flags)
        db = _active_param_grads(wf, b, c_flags)
        keys = set(da) & set(db)
        assert "j3_matrix" in keys  # the meaningful J3 coupling block is actually exercised
        assert {"j3_basis_exp", "j3_basis_coeff"} <= keys  # J3 radial basis-opt now analytic
        for k in keys:
            np.testing.assert_allclose(np.asarray(da[k]), np.asarray(db[k]), atol=atol, rtol=rtol)

    # r/R axes: assembled position/force grads leaf-for-leaf (no basis blocks here).
    for quantity, wrt in (("e_L", "r"), ("e_L", "R"), ("ln_Psi", "R")):
        if quantity == "e_L":
            a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags(wrt), analytic=True)[wrt]
            b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt=wrt, mask=None)
        else:
            a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags(wrt), analytic=True)[wrt]
            b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt=wrt, mask=None)
        for la, lb in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
            np.testing.assert_allclose(np.asarray(la), np.asarray(lb), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5"])
def test_pbc_j3_force_stress_vs_auto(stem):
    """Active three-body Jastrow on a real-Gamma periodic (sphe-pgto SD ECP) system: J3 force
    (e_L/R, ln_Psi/R) and stress (e_L/eps, ln_Psi/eps) vs _auto. Exercises the periodic-J3 basis
    derivatives (ao_deriv.j3_basis_derivs periodic branch) and the periodic-J3 stress
    (pbc_stress._j3_kstress) on top of the det Pulay + Ewald/ECP pieces."""
    if is_mixed_precision_enabled():
        _compare_native_api_next(stem, "j3-periodic")
        return
    ham = _build_ham_j3(stem)
    r_up, r_dn = _draw(ham)
    assert ham.structure_data.pbc_flag
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    for quantity, wrt in (("e_L", "R"), ("e_L", "eps"), ("ln_Psi", "R"), ("ln_Psi", "eps")):
        if quantity == "e_L":
            a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags(wrt), analytic=True)[wrt]
            b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt=wrt, mask=_ref_mask(stem))
        else:
            a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags(wrt), analytic=True)[wrt]
            b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt=wrt, mask=_ref_mask(stem))
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


def _build_ham_j3_composite_bspline(stem: str = "lithium.h5", seed: int = 5) -> Hamiltonian_data:
    """``_build_ham_j3`` twin whose J3 orbital is a COMPOSITE (pGTO (+) B-spline).

    The composite is built at test time from the committed periodic ``lithium.h5`` sphe-pGTO AO leaf
    plus a synthetic 2-band periodic B-spline leaf on the SAME cell (``AOs_composite_data`` via
    ``+``), then wrapped in an identity ``MOs_data`` so the forward J3 kinetic path (MO-backed)
    supports it. The pGTO child carries the radial basis-opt params; the B-spline child carries none
    (dAO/dc = 0), so it contributes zero J3-basis gradient -- but the composite field concatenation in
    ``ao_deriv.j3_basis_derivs`` still needs its field tower.
    """
    from jqmc.atomic_orbital import AOs_bspline_data
    from jqmc.pw_pgto_projection import bspline_prefilter_periodic

    (structure_data, aos_data, _mu, _md, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        os.path.join(_TREXIO, stem), store_tuple=True
    )
    # Synthetic periodic B-spline leaf (2 bands) on the fixture cell -- smooth trig fields, prefiltered.
    ns, order = (8, 8, 8), 5
    grids = [np.arange(n) / n for n in ns]
    fx, fy, fz = np.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
    x, y, z = fx, fy, fz
    tp = 2.0 * np.pi
    b0 = np.sin(tp * x) + 0.5 * np.cos(tp * y) + 0.25 * np.sin(tp * z)
    b1 = np.cos(tp * (x + y)) + 0.3 * np.sin(tp * z)
    coeffs = bspline_prefilter_periodic(np.stack([b0, b1], axis=0).astype(np.complex128), order)
    bspline_leaf = AOs_bspline_data.build(structure_data, coeffs.reshape(coeffs.shape[0], -1), ns, order)

    composite = aos_data + bspline_leaf  # pGTO (+) B-spline composite J3 orbital leaf
    j3_orb = MOs_data(num_mo=composite.num_ao, aos_data=composite, mo_coefficients=np.eye(composite.num_ao))
    rng = np.random.default_rng(seed)
    jm = rng.normal(size=(composite.num_ao, composite.num_ao + 1)) * 0.1
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)  # bosonic coupling block must be symmetric
    j3 = Jastrow_three_body_data(orb_data=j3_orb, j_matrix=jm)
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure_data
    )
    n_atom = len(structure_data.positions)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=structure_data,
        core_electrons=core_electrons_of(coulomb_potential_data, n_atom),
        jastrow_1b_type="exp",
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=j3, jastrow_dtn_data=None
    )
    wf = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    return Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=coulomb_potential_data, wavefunction_data=wf)


@pytest.mark.light
@pytest.mark.parametrize("quantity", ["ln_Psi", "e_L"])
def test_composite_bspline_j3_basis_grads_vs_auto(quantity):
    """Composite J3 orbital with a B-SPLINE child: analytic J3-basis grad == the autodiff twin.

    Sibling of ``_grad_fidelity_harness.test_composite_j3_basis_grads_vs_auto`` (pGTO (+) PW), but
    the composite carries a B-spline child instead of a plane-wave one. The c-axis J3 value O_k
    (basis.grad_J3_wrt_basis) and kinetic dT/dc (basis.grad_T_wrt_J3_basis, which calls
    ao_deriv.j3_basis_derivs on the composite) route the pGTO child's radial exp/coeff grad through the
    per-child general contraction; the B-spline child carries NO radial params (zero J3-basis grad) yet
    its field tower is still needed for the composite field concatenation, so the composite's B-spline
    child must not raise ``NotImplementedError``. The flat reducer publishes per-child
    blocks; this test independently compares the RAW grad-pytree radial leaves of every GTO/pGTO child
    (non-GTO children carry no radial params)."""
    if is_mixed_precision_enabled():
        _compare_native_api_exotic("composite-bspline-j3", quantity)
        return
    ham = _build_ham_j3_composite_bspline()
    aos = ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data.aos_data
    assert type(aos).__name__ == "AOs_composite_data"
    child_kinds = [type(c).__name__ for c in aos.children]
    assert any("pgto" in k for k in child_kinds) and any("bspline" in k for k in child_kinds)

    r_up, r_dn = _draw(ham)
    atol, rtol = get_tolerance_min(
        ("ao_eval", "jastrow_eval", "jastrow_grad_lap", "det_eval", "coulomb", "wf_kinetic"), "strict"
    )
    if quantity == "ln_Psi":
        a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags("c"), analytic=True)["c"]
        b = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags("c"), analytic=False)["c"]
    else:
        a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("c"), analytic=True)["c"]
        b = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("c"), analytic=False)["c"]
    ga = a.jastrow_data.jastrow_three_body_data.orb_data.aos_data  # grad pytree: MO wrapper -> composite
    gb = b.jastrow_data.jastrow_three_body_data.orb_data.aos_data

    compared = 0
    for i, (ca, cb) in enumerate(zip(ga.children, gb.children)):
        if "gto" not in type(ca).__name__:  # B-spline child: no radial params -> no basis grad emitted
            assert getattr(ca, "radial", None) is None and getattr(cb, "radial", None) is None
            continue
        np.testing.assert_allclose(
            np.asarray(ca.radial.exponents),
            np.asarray(cb.radial.exponents),
            atol=atol,
            rtol=rtol,
            err_msg=f"{quantity} composite-bspline J3-basis child{i} d/d exponents",
        )
        np.testing.assert_allclose(
            np.asarray(ca.radial.coefficients),
            np.asarray(cb.radial.coefficients),
            atol=atol,
            rtol=rtol,
            err_msg=f"{quantity} composite-bspline J3-basis child{i} d/d coefficients",
        )
        compared += 1
    assert compared > 0, "no GTO child compared -- the composite J3-basis check would be vacuous"
    jax.clear_caches()


@pytest.mark.light
@pytest.mark.parametrize("stem", ["H2_ae_ccpvdz_cart.h5"])
def test_dtn_force_vs_auto(stem):
    """DTN nuclear force: d ln|Psi|/dR (Pulay, dtn.grad_dtn_wrt_R) and d e_L/dR (adds the DTN
    kinetic force, u/chi/f channels). Molecular AE DTN-only, built at test time from the trexio
    structure. Mirrors test_dtn_e_L_dc_vs_auto but for the R axis. vs _auto.

    Fixture note: the R/e_L axis routes through the jitted compute_grads_and_laplacian_Jastrow_part
    (via grad_T_wrt_R_withJ). Jastrow_DTN_data carries numpy arrays as pytree_node=False *static*
    metadata (M_alpha/b_alpha/... constraint matrices); when two DTN instances of the SAME molecular
    structure flow through that jitted kernel in one process, their treedefs hash-collide and JAX
    raises 'arrays cannot be passed as metadata fields' on the aux-data equality check. The existing
    test_dtn_e_L_dc_vs_auto already exercises this kernel on the N_ae structure, so this force test
    uses the distinct H2_ae structure to stay green in the shared suite. (The N_ae variant is exact
    too -- e_L/R 5.3e-12 vs _auto -- but only when run in isolation from test_dtn_e_L_dc; the collision
    is a pre-existing source limitation, the documented JAX-state class, not a logic error.)"""
    if is_mixed_precision_enabled():
        _compare_native_api_next(stem, "dtn-force")
        return
    from jqmc.jastrow_factor import Jastrow_DTN_data

    (structure_data, _aos, _mu, _md, gem, cpd) = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    z_eff, pseudized = _dtn_species_policy(structure_data, cpd)
    dtn = Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=z_eff,
        pseudized_per_species=pseudized,
    )
    jastrow = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=dtn
    )
    wf = Wavefunction_data(jastrow_data=jastrow, geminal_data=gem)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=cpd, wavefunction_data=wf)
    r_up, r_dn = _draw(ham)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    for quantity in ("ln_Psi", "e_L"):
        if quantity == "e_L":
            a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("R"), analytic=True)["R"]
            b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="R", mask=None)
        else:
            a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags("R"), analytic=True)["R"]
            b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="R", mask=None)
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


def _build_pbc_dtn_ham(stem: str) -> Hamiltonian_data:
    """Real-Gamma periodic ECP Hamiltonian with a DTN Jastrow built from the trexio structure
    (same construction as test_dtn_stress_vs_auto), a committed JDTN PBC ECP system."""
    from jqmc.jastrow_factor import Jastrow_DTN_data

    (structure_data, _aos, _mu, _md, gem, cpd) = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    z_eff, pseudized = _dtn_species_policy(structure_data, cpd)
    dtn = Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=z_eff,
        pseudized_per_species=pseudized,
    )
    jastrow = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=dtn
    )
    wf = Wavefunction_data(jastrow_data=jastrow, geminal_data=gem)
    return Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=cpd, wavefunction_data=wf)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5"])
def test_dtn_kinetic_shared_inverse_no_svd(stem):
    """The DTN kinetic cotangent must reuse the running geminal inverse (no fresh pinv SVD). With the
    per-walker ``geminal_inv`` threaded into the production e_L/{r,R,eps} closures, the lowered
    StableHLO carries ZERO ``lapack_dgesdd_ffi`` for ALL of e_L/{r,R,eps}. On the eps axis
    ``one_move_mesh.mesh_stress_wf`` uses the closed-form periodic Pulay-stress delta
    (``sum_g W_g (stress_lnPsi_pbc(config) - stress_lnPsi_pbc(base))``) streamed with per-config
    Sherman-Morrison inverses off the base ``G^{-1}``, not autodiff of the strained log-ratio through a
    fresh strained-geminal pinv -- no fresh pinv on any config.
    Numerically, providing the inverse is inert: the r/R/eps DTN kinetic gradients agree with the
    SVD-pinv fallback to fp."""
    from jqmc.derivatives import bridge
    from jqmc.derivatives.fast import geminal_pinv
    from jqmc.derivatives.hoist import cartesianize_orbitals
    from jqmc.determinant import compute_geminal_all_elements

    ham = _build_pbc_dtn_ham(stem)
    assert ham.structure_data.pbc_flag and ham.wavefunction_data.jastrow_data.jastrow_dtn_data is not None
    gd = ham.wavefunction_data.geminal_data
    ru0, rd0 = _draw(ham, seed=0)
    ru1, rd1 = _draw(ham, seed=1)
    rub = jnp.stack([ru0, ru1])
    rdb = jnp.stack([rd0, rd1])
    RTb = jnp.tile(jnp.eye(3)[None], (rub.shape[0], 1, 1))
    gi_b = jax.jit(jax.vmap(lambda a, b: geminal_pinv(compute_geminal_all_elements(gd, a, b))))(rub, rdb)
    jax.block_until_ready(gi_b)

    hc = cartesianize_orbitals(ham)
    expected_svd = {"r": 0, "R": 0, "eps": 0}  # eps 1->0: mesh_stress_wf autodiff replaced by closed form
    level = "medium" if is_mixed_precision_enabled() else "strict"
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), level)
    for wrt in ("r", "R", "eps"):
        txt = bridge._jit_e_L_giv(wrt, False).lower(hc, rub, rdb, RTb, gi_b).as_text()
        assert txt.count("lapack_dgesdd_ffi") == expected_svd[wrt], (
            f"e_L/{wrt}: {txt.count('lapack_dgesdd_ffi')} SVD FFI calls, expected {expected_svd[wrt]}"
        )
        a = np.asarray(bridge.batched_grad_e_L(wrt=wrt, with_geminal_inv=False)(ham, rub, rdb, RTb))
        b = np.asarray(bridge.batched_grad_e_L(wrt=wrt, with_geminal_inv=True)(ham, rub, rdb, RTb, gi_b))
        np.testing.assert_allclose(b, a, atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5"])
@pytest.mark.parametrize("wrt", ["r", "R", "c"])
def test_pbc_dtn_ecp_e_L_closed_form_vs_auto(stem, wrt):
    """The PBC-DTN ECP-nonlocal ``e_L`` gradients are assembled by the ``one_move_mesh`` rank-1 fast path,
    whose DTN mesh kernels are CLOSED FORM (no jax.grad/value_and_grad in the analytic graphs):

      * e_L/r -> ``one_move_mesh.dtn_mesh`` streams the closed-form per-config DTN drift
                 (``compute_grads_and_laplacian_Jastrow_DTN``) instead of autodiffing the one-move delta;
      * e_L/R -> ``one_move_mesh.mesh_force`` streams ``dtn.grad_dtn_wrt_R`` (config - base) instead of grad;
      * e_L/c -> ``one_move_mesh.dV_ecp_dc_fast`` streams ``dtn.grad_dtn_wrt_param`` (config - base) instead of grad.

    Checked against the ``_auto`` AD oracle on the committed lithium.h5 PBC ECP fixture (real Gamma; DTN
    built from the trexio structure).  Parametrized per-axis so xdist keeps each a single light oracle
    (avoids the DTN full-``_auto`` XLA-accumulation flake of one multi-axis test).  The eps axis (the
    closed-form ``one_move_mesh.mesh_stress_wf`` periodic Pulay-stress delta) is covered by the sibling
    ``test_dtn_stress_vs_auto`` (d e_L/d eps routes through ``grad_V_ecp_wrt_strain -> mesh_stress_wf``)."""
    if is_mixed_precision_enabled():
        _compare_native_api_next(stem, "dtn-closed", wrt)
        return
    ham = _build_pbc_dtn_ham(stem)
    assert ham.structure_data.pbc_flag and ham.wavefunction_data.jastrow_data.jastrow_dtn_data is not None
    r_up, r_dn = _draw(ham, seed=0)
    RT = jnp.eye(3)
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    a = derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, _flags(wrt), analytic=True)[wrt]
    b = _auto.compute_grad_e_L(ham, r_up, r_dn, RT, wrt=wrt, mask=None)
    if wrt == "c":  # DTN indep params (dV_ecp_dc_fast DTN block)
        da, db = a.jastrow_data.jastrow_dtn_data, b.jastrow_data.jastrow_dtn_data
        for k in ("alpha_indep", "beta_indep", "gamma_indep"):
            np.testing.assert_allclose(np.asarray(getattr(da, k)), np.asarray(getattr(db, k)), atol=atol, rtol=rtol)
    else:
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["lithium.h5"])
def test_dtn_stress_vs_auto(stem):
    """DTN stress (periodic-only): d ln|Psi|/d eps (dtn.grad_dtn_wrt_strain) and d e_L/d eps (adds
    grad_T_dtn_wrt_strain), both wired into pbc_stress. Real-Gamma periodic (lithium) with DTN built
    from the trexio structure alongside the usual geminal. vs _auto."""
    if is_mixed_precision_enabled():
        _compare_native_api_next(stem, "dtn-stress")
        return
    from jqmc.derivatives import dtn as dtn_module
    from jqmc.jastrow_factor import Jastrow_DTN_data

    (structure_data, _aos, _mu, _md, gem, cpd) = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    z_eff, pseudized = _dtn_species_policy(structure_data, cpd)
    dtn = Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=z_eff,
        pseudized_per_species=pseudized,
    )
    jastrow = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=dtn
    )
    wf = Wavefunction_data(jastrow_data=jastrow, geminal_data=gem)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=cpd, wavefunction_data=wf)
    r_up, r_dn = _draw(ham)
    assert ham.structure_data.pbc_flag
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    # The AD oracle differentiates through ``Hamiltonian_data._apply_strain``, which follows the
    # comoving cutoff policy for periodic DTN factors (``Jastrow_DTN_data.cutoff_scale``), so both
    # d ln|Psi|/d eps and d e_L/d eps are compared STRICTLY against the analytic kernels -- the
    # isotropic policy term is present on both sides. Its magnitude is additionally pinned by
    # ``TestDTNComovingApplyStrain`` (fixed-cutoff AD + policy/3 I == comoving AD) and by the
    # Rust parity tests (test_gfmc_stress_ecp_pbc_dtn).
    for quantity in ("ln_Psi", "e_L"):
        if quantity == "e_L":
            a = derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), _flags("eps"), analytic=True)["eps"]
            b = _auto.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), wrt="eps", mask=_ref_mask(stem))
        else:
            a = derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, _flags("eps"), analytic=True)["eps"]
            b = _auto.compute_grad_ln_Psi(ham, r_up, r_dn, wrt="eps", mask=_ref_mask(stem))
            # Sanity of the convention: the comoving oracle differs from the fixed-cutoff one by a
            # non-trivial isotropic policy term for this fixture.
            policy = float(dtn_module.dtn_cutoff_policy_value(dtn, r_up, r_dn))
            assert abs(policy) > 1.0e-3
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol)


def _native_api_exp_case(stem):
    form = "exp"
    from jqmc.derivatives.assemble import grad_J1_wrt_param, grad_J2_wrt_param
    from jqmc.jastrow_factor import compute_Jastrow_one_body, compute_Jastrow_two_body

    structure_data, _aos, _mu, _md, gem, _cpd = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    n_atom = len(structure_data.positions)
    p1, p2 = (1.3, 0.7)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=p1, structure_data=structure_data, core_electrons=core_electrons_of(_cpd, n_atom), jastrow_1b_type=form
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=p2, jastrow_2b_type=form, structure_data=structure_data
    )
    rng = np.random.default_rng(7)
    r_up = jnp.asarray(rng.normal(scale=0.7, size=(int(gem.num_electron_up), 3)))
    r_dn = jnp.asarray(rng.normal(scale=0.7, size=(int(gem.num_electron_dn), 3)))
    return (j1, j2, r_up, r_dn)


def _native_api_exp_evaluate(case, entry_floor):
    from jqmc.derivatives.assemble import grad_J1_wrt_param, grad_J2_wrt_param
    from jqmc.jastrow_factor import compute_Jastrow_one_body, compute_Jastrow_two_body

    j1, j2, up, down = case
    ana1 = np.asarray(grad_J1_wrt_param(j1, up, down), dtype=np.float64)
    with _fp64_reference_policy():
        ref1 = np.asarray(
            jax.grad(lambda p: compute_Jastrow_one_body(dataclasses.replace(j1, jastrow_1b_param=p), up, down))(
                float(j1.jastrow_1b_param)
            ),
            dtype=np.float64,
        )
    # Retain the original independent mathematical-reference liveness predicate
    # at the stored parameter and the caller's original entry-mode floor.
    # This reference does not serve as the Mixed numerical comparator.
    assert abs(float(ref1)) > entry_floor
    ana2 = np.asarray(grad_J2_wrt_param(j2, up, down), dtype=np.float64)
    with _fp64_reference_policy():
        ref2 = np.asarray(
            jax.grad(lambda p: compute_Jastrow_two_body(dataclasses.replace(j2, jastrow_2b_param=p), up, down))(
                float(j2.jastrow_2b_param)
            ),
            dtype=np.float64,
        )
    assert abs(float(ref2)) > entry_floor
    return {"blocks": {"J1": ana1, "J2": ana2}, "up": up, "down": down}


def _assert_native_api_exp_fidelity(mixed_block, full_block, atol, rtol):
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max()


def _compare_native_api_exp_mixed(stem):
    entry_floor, _ = get_tolerance_min(("jastrow_eval",), "strict")
    mixed, full = mixed_vs_full(
        lambda: _native_api_exp_case(stem),
        lambda case: _native_api_exp_evaluate(case, entry_floor),
    )
    assert np.array_equal(mixed["up"], full["up"]) and np.array_equal(mixed["down"], full["down"])
    assert set(mixed["blocks"]) == set(full["blocks"]) == {"J1", "J2"}
    blocks = mixed_fidelity_blocks(mixed["blocks"], full["blocks"])
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    for mixed_block, full_block in blocks:
        _assert_native_api_exp_fidelity(mixed_block, full_block, atol, rtol)
    faulty = out_of_budget_mixed_block(*blocks[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _assert_native_api_exp_fidelity(faulty, blocks[0][1], atol, rtol)


@pytest.mark.light
@pytest.mark.parametrize("stem", ["Li_ae_ccpvdz_cart.h5", "lithium.h5"])
def test_j1_j2_exp_parameter_precision_partition(stem):
    """Expose exactly the Exp cases to the Mixed fidelity contract."""
    if is_mixed_precision_enabled():
        _compare_native_api_exp_mixed(stem)
    else:
        test_j1_j2_param_grad_functional_forms_vs_auto(stem, "exp")


@pytest.mark.light
@pytest.mark.parametrize("stem", ["Li_ae_ccpvdz_cart.h5", "lithium.h5"])
def test_j1_j2_pade_parameter_math_partition(stem):
    """Preserve the fixed-FP64 Pade mathematical derivatives in both modes."""
    test_j1_j2_param_grad_functional_forms_vs_auto(stem, "pade")


@pytest.mark.light
@pytest.mark.parametrize("stem", ["Li_ae_ccpvdz_cart.h5", "lithium.h5"])
@pytest.mark.parametrize("form", ["exp", "pade"])
def test_j1_j2_param_grad_functional_forms_vs_auto(stem, form):
    """``dJ1/dp`` and ``dJ2/dp`` vs ``jax.grad`` for BOTH functional forms.

    The hand-written parameter derivatives in ``derivatives.assemble`` branch on
    ``jastrow_1b_type`` / ``jastrow_2b_type``, but every numeric analytic-vs-AD
    fixture pinned ``"exp"``, so the ``"pade"`` arms were reachable in production
    (``jastrow_2b_type`` even DEFAULTS to ``"pade"``) while no oracle ever
    evaluated them: a factor-2 error in either Pade arm survived the whole
    registered set. Differentiating the Jastrow VALUE function is an oracle
    independent of the analytic branch under test.

    Precision: Pade parameter arithmetic is fp64 in both modes; Exp has an fp32
    parameter-only denominator in mixed precision and uses numerical fidelity there.
    For the retained mathematical comparisons the ``jax.grad`` oracle is evaluated with every precision zone at fp64
    (``_fp64_reference_policy``; a no-op in full mode) instead of in the mixed fp32
    ``jastrow_eval`` zone, and at the parameter value the analytic kernel actually reads
    (the stored leaf, float64 in both modes, i.e. exactly ``p1`` / ``p2``).
    """
    if is_mixed_precision_enabled() and form == "exp":
        _compare_native_api_exp_mixed(stem)
        return
    from jqmc.derivatives.assemble import grad_J1_wrt_param, grad_J2_wrt_param
    from jqmc.jastrow_factor import compute_Jastrow_one_body, compute_Jastrow_two_body

    (structure_data, _aos, _mu, _md, gem, _cpd) = read_trexio_file(os.path.join(_TREXIO, stem), store_tuple=True)
    n_atom = len(structure_data.positions)
    p1, p2 = 1.3, 0.7
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=p1,
        structure_data=structure_data,
        core_electrons=core_electrons_of(_cpd, n_atom),
        jastrow_1b_type=form,
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=p2, jastrow_2b_type=form, structure_data=structure_data
    )
    rng = np.random.default_rng(7)
    r_up = jnp.asarray(rng.normal(scale=0.7, size=(int(gem.num_electron_up), 3)))
    r_dn = jnp.asarray(rng.normal(scale=0.7, size=(int(gem.num_electron_dn), 3)))
    atol, rtol = get_tolerance_min(("jastrow_eval",), "strict")

    ana1 = np.asarray(grad_J1_wrt_param(j1, r_up, r_dn), dtype=np.float64)
    with _fp64_reference_policy():
        ref1 = np.asarray(
            jax.grad(lambda p: compute_Jastrow_one_body(dataclasses.replace(j1, jastrow_1b_param=p), r_up, r_dn))(
                float(j1.jastrow_1b_param)
            ),
            dtype=np.float64,
        )
    # A vanishing reference would make the comparison vacuous for any mutation
    # that scales the analytic result.
    assert abs(float(ref1)) > atol, f"J1 parameter derivative is degenerate for form={form}, stem={stem}"
    np.testing.assert_allclose(ana1, ref1, atol=atol, rtol=rtol)

    ana2 = np.asarray(grad_J2_wrt_param(j2, r_up, r_dn), dtype=np.float64)
    with _fp64_reference_policy():
        ref2 = np.asarray(
            jax.grad(lambda p: compute_Jastrow_two_body(dataclasses.replace(j2, jastrow_2b_param=p), r_up, r_dn))(
                float(j2.jastrow_2b_param)
            ),
            dtype=np.float64,
        )
    assert abs(float(ref2)) > atol, f"J2 parameter derivative is degenerate for form={form}, stem={stem}"
    np.testing.assert_allclose(ana2, ref2, atol=atol, rtol=rtol)


def _api_first17_ecp_fidelity(stem, axis):
    """Same physical input and configurations; independently rebuilt XLA policies."""
    import tempfile
    from pathlib import Path

    from jqmc._precision import configure
    from jqmc._setting import get_tolerance_mixed_fidelity
    from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

    entry_mode = "mixed" if is_mixed_precision_enabled() else "full"
    with tempfile.TemporaryDirectory(prefix="jqmc_api_ecp_source_") as temporary:
        source = Path(temporary) / "full_physical_source.h5"
        try:
            configure("full")
            full_source = _build_ham(stem)
            up, down = _draw(full_source)
            fixed_up, fixed_down = np.asarray(up).copy(), np.asarray(down).copy()
            full_source.save_to_hdf5(source)
        finally:
            configure(entry_mode)

        def factory():
            return Hamiltonian_data.load_from_hdf5(source)

        def evaluate(ham):
            assert ham.coulomb_potential_data.ecp_flag
            up, down = jnp.asarray(fixed_up), jnp.asarray(fixed_down)
            flags = _flags(axis)
            result = derivatives.compute_grad_e_L(ham, up, down, jnp.eye(3), flags, analytic=True)[axis]
            if axis == "c":
                return _active_param_grads(ham.wavefunction_data, result, flags)
            if axis == "r":
                return {"electron-up": result[0], "electron-down": result[1]}
            return {"strain": result}

        mixed, full = mixed_vs_full(factory, evaluate)
    assert set(mixed) == set(full)
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    compared = 0
    for field in full:
        assert np.asarray(mixed[field]).shape == np.asarray(full[field]).shape
        if np.asarray(full[field]).size == 0:
            continue
        for mixed_block, full_block in mixed_fidelity_blocks(mixed[field], full[field]):
            compared += 1
            assert np.max(np.abs(mixed_block - full_block)) <= atol + rtol * np.max(np.abs(full_block)), field
            changed = out_of_budget_mixed_block(mixed_block, full_block, atol=atol, rtol=rtol)
            with pytest.raises(AssertionError):
                assert np.max(np.abs(changed - full_block)) <= atol + rtol * np.max(np.abs(full_block))
    assert compared > 0
