"""Complex-twist DTN x ECP c-axis (param-grad) production closures.

The intersection ``{complex-twist orbitals} x {DTN jastrow} x {ECP} x {e_L/c axis}``
-- all analytic-implemented components -- must lower and stay complex. In the DTN
kinetic ``dT/dc`` kernel (``jqmc/derivatives/dtn.py`` ``grad_T_dtn_wrt_param``) a
complex-Bloch (twist) geminal drift ``GD`` makes the accumulated contributions
``complex128``, so three accumulators -- the f-channel (``gamma``, e-e-N) ``lax.scan``
carry and the ``beta`` / ``gamma`` species-scatter zeros -- are initialised with
``jnp.result_type(M.dtype, GD.dtype)`` (complex for twist; ``f64`` for real systems ->
bit-identical). A ``float64`` scan carry raises a ``scan carry type float64 vs
complex128`` ``TypeError``; ``float64`` scatter zeros drop Im silently in ``.at[].add``.
The accumulation stays complex and is Re-projected once at the top of the tree by
``assemble.compute_grad_e_L`` (L1005 ``tree_map(jnp.real)``), matching the sibling
``kinetic.grad_T_wrt_J1/J2/J3`` convention (they too return the complex, un-projected
``cot=gJ+gD`` contraction). Because the DTN feature grad/lap fields are REAL, a per-term
``Re`` is mathematically equivalent to the end-of-tree ``Re`` here
(``Re(GD.real) = Re(GD).real``), as the ``vs_auto`` oracle below confirms (u/chi/gamma
blocks match to ~1e-14).

The checks fire at TRACE / LOWER time (no MCMC step needed), so they are cheap.

Coverage elsewhere: the 8-set harness (test_analytic_grad_matches_autograd_curated)
pairs the complex-twist fixture (``lithium_k025.h5``) with j12 / J3 and with DTN (the
``Li_pbc_twist_JDTN`` row); the DTN c-axis tests (test_jqmc_derivatives
``test_dtn_e_L_dc_vs_auto`` / ``test_dtn_okc_pbc_vs_auto``) run real molecular /
real-Gamma PBC only, and no committed fixture is a twist + DTN Hamiltonian.

This module builds that fixture in-test (committed complex-twist ECP trexio
``lithium_k025.h5`` + a DTN Jastrow built at test time, u/chi/f channels) and drives
the SAME production bridge closures the optimizer uses (``bridge._jit_*_giv("c",
False)`` with the running Sherman-Morrison ``geminal_inv`` threaded), plus the eager
``compute_grad_e_L(analytic=True)`` vs the ``_auto`` autodiff oracle.
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

import os
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]

from jqmc import derivatives
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import get_tolerance_mixed_fidelity
from jqmc.derivatives import _auto, bridge
from jqmc.derivatives.fast import geminal_pinv
from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_DTN_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

_TREXIO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")

# dtype-agnostic SVD-pinv detector (matches the fp32/fp64/complex LAPACK SVD call sites the
# streamed param-grad path would lower if the running geminal_inv were NOT threaded).
_SVD = re.compile(r"lapack_[a-z]gesdd|lapack_[a-z]gesvd")


def _eager_inputs(gem, seed: int = 0):
    """Eager (un-batched) (r_up, r_dn) for the ``compute_grad_e_L`` / ``_auto`` oracle compare."""
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=0.7, size=(nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=0.7, size=(nd, 3)))
    return r_up, r_dn


def _build_twist_dtn_ham() -> tuple[Hamiltonian_data, object]:
    """Complex-twist (k=1/4) ccECP Li Hamiltonian (committed ``lithium_k025.h5`` trexio:
    ``AOs_sphe_pgto_complex_data`` geminal + ECP coulomb) wrapped with a DTN-only Jastrow
    (u/chi/f) built at test time -- NO derived h5 committed.  Small polynomial orders keep
    it light; ``use_f=True`` is REQUIRED to exercise the f-channel scan that carries the bug.
    Returns (ham, geminal_data)."""
    (structure_data, _aos, _mu, _md, gem, cpd) = read_trexio_file(os.path.join(_TREXIO, "lithium_k025.h5"), store_tuple=True)
    dtn = Jastrow_DTN_data.build(
        structure_data=structure_data,
        z_eff_per_species=(1.0,),  # Li ccECP: Z=3, two core electrons
        pseudized_per_species=(True,),
        use_u=True,
        use_chi=True,
        use_f=True,  # the f (e-e-N) channel scan is where the complex-carry crash fires
        cutoff_smoothness=4,
        n_order_u=2,
        n_order_chi=2,
        n_order_eN=2,
        n_order_ee=2,
        L_u=None,
        L_chi=None,
        L_f=None,
        random_init=True,
        seed=13,
    )
    jastrow = Jastrow_data(
        jastrow_one_body_data=None, jastrow_two_body_data=None, jastrow_three_body_data=None, jastrow_dtn_data=dtn
    )
    wf = Wavefunction_data(jastrow_data=jastrow, geminal_data=gem)
    ham = Hamiltonian_data(structure_data=structure_data, coulomb_potential_data=cpd, wavefunction_data=wf)
    return ham, gem


def _batched_inputs(gem, nw: int = 1, seed: int = 0):
    """One-walker batched (r_up, r_dn, RT, geminal_inv) matching the production bridge call
    convention: the running SM ``geminal_inv`` is threaded so NO fresh SVD-pinv lowers."""
    nu, nd = int(gem.num_electron_up), int(gem.num_electron_dn)
    rng = np.random.default_rng(seed)
    r_up = jnp.asarray(rng.normal(scale=0.7, size=(nw, nu, 3)))
    r_dn = jnp.asarray(rng.normal(scale=0.7, size=(nw, nd, 3)))
    RT = jnp.tile(jnp.eye(3)[None], (nw, 1, 1))
    gi = jax.jit(jax.vmap(lambda a, b: geminal_pinv(compute_geminal_all_elements(gem, a, b)), in_axes=(0, 0)))(r_up, r_dn)
    jax.block_until_ready(gi)
    return r_up, r_dn, RT, gi


@pytest.mark.light
def test_twist_dtn_ecp_fixture_is_the_intended_cell():
    """Precondition pin: the builder yields EXACTLY the twist x DTN x ECP intersection cell whose
    analytic e_L/c path this module regresses -- a periodic complex-Bloch (twist) geminal leaf, an
    active DTN jastrow, and an ECP potential. The behavioral analytic-path coverage for this cell is
    ``test_twist_dtn_e_L_c_lowers`` (lowers, no dtype crash) + the ``vs_auto`` oracle below."""
    ham, _ = _build_twist_dtn_ham()
    assert ham.structure_data.pbc_flag
    assert ham.coulomb_potential_data.ecp_flag
    assert ham.wavefunction_data.jastrow_data.jastrow_dtn_data is not None
    # the twist geminal leaf is the complex-Bloch sphe-pgto (via an MO wrapper)
    od = ham.wavefunction_data.geminal_data.orb_data_up_spin
    aos = od.aos_data if type(od).__name__ == "MOs_data" else od
    assert "complex" in type(aos).__name__.lower()


@pytest.mark.light
def test_twist_dtn_o_k_c_lowers_zero_svd():
    """CONTROL (working half): the O_k = d ln_Psi/dc production closure lowers cleanly and, with the
    running geminal_inv threaded, lowers 0 SVD-pinv. Proves the fixture is buildable and the streamed
    param-grad path has no LAPACK-SVD hole -- so the e_L/c failure below is specific to the kinetic
    dT/dc scan, not the fixture or the c-axis machinery in general."""
    ham, gem = _build_twist_dtn_ham()
    r_up, r_dn, _RT, gi = _batched_inputs(gem)
    jf = bridge._jit_ln_psi_giv("c", False)
    text = jf.lower(ham, r_up, r_dn, gi).as_text()  # must not raise
    n_svd = len(_SVD.findall(text))
    assert n_svd == 0, f"O_k(ln_Psi)/c streamed param-grad lowered {n_svd} SVD-pinv (geminal_inv not reused?)"


@pytest.mark.light
def test_twist_dtn_e_L_c_lowers():
    """The e_L/dc production closure LOWERS for the twist x DTN x ECP cell: the
    dtn.grad_T_dtn_wrt_param f-channel scan carry and the beta/gamma species-scatter zeros take
    ``jnp.result_type(M, GD)``, so the complex-Bloch geminal drift ``GD`` flows through (no ``TypeError``,
    no Im-drop).

    SVD: the lowered e_L/c graph carries ZERO ``lapack_[dz]gesdd``. The ECP non-local ``dV_nl/dc``
    STREAMED mesh path re-evaluates ``O_k`` at the mesh-DISPLACED configs
    (``one_move_mesh.fast_supported`` returns ``False`` for complex-twist orbitals, so
    ``assemble._dV_ecp_dc`` takes the streamed ``_mesh_wsum`` branch), and each mesh-config inverse is
    an EXACT rank-1 Sherman-Morrison update off the threaded base ``geminal_inv``
    (``one_move_mesh.sm_geminal_inv_configs``): an ECP move displaces ONE electron -> a rank-1 row/column update
    of the pairing matrix ``G`` -> its inverse is a pure-GEMM SM update of the base inverse, no fresh
    decomposition (dtype-agnostic, complex128 twist OK). MPI runs cannot register the LAPACK Host FFI,
    so no SVD may remain on the twist e_L/{c,r,eps,R} axes (the SM inverse agrees with the fresh-pinv
    result to ~1e-14)."""
    ham, gem = _build_twist_dtn_ham()
    r_up, r_dn, RT, gi = _batched_inputs(gem)
    jf = bridge._jit_e_L_giv("c", False)
    text = jf.lower(ham, r_up, r_dn, RT, gi).as_text()  # must NOT raise
    n_svd = len(_SVD.findall(text))
    assert n_svd == 0, (
        f"twist e_L/c giv expected 0 SVD-pinv (the ECP mesh-config inverse is the rank-1 "
        f"Sherman-Morrison update of the threaded base geminal_inv), lowered {n_svd}"
    )


@pytest.mark.light
def test_twist_dtn_e_L_c_vs_auto():
    """CORRECTNESS oracle: analytic ``dE_L/dc`` == ``_auto`` for ALL DTN param blocks (alpha/beta/gamma)
    on the twist x DTN x ECP cell.  ``_auto`` (jax.grad of the Re-projected complex-Bloch local energy)
    serves this cell for real DTN params, so it is a valid a==b twin.  The u/chi/gamma blocks match to
    the module ``strict`` tolerance: the DTN feature grad/lap fields are real, so a per-term
    ``Re(GD . real_field) = Re(GD) . real_field`` equals the end-of-tree ``Re`` (assemble L1005)."""
    if is_mixed_precision_enabled():
        _mixed_twist_dtn_fidelity()
        return
    ham, gem = _build_twist_dtn_ham()
    r_up, r_dn = _eager_inputs(gem)
    RT = jnp.eye(3)
    flags = {k: True for k in derivatives._PARAM_FLAG_KEYS}
    b = _auto.compute_grad_e_L(ham, r_up, r_dn, RT, wrt="c", mask=None)  # autodiff oracle (Re-projected)
    a = derivatives.compute_grad_e_L(ham, r_up, r_dn, RT, flags, analytic=True)["c"]
    atol, rtol = get_tolerance_min(("ao_eval", "jastrow_eval", "det_eval", "local_energy"), "strict")
    da = a.jastrow_data.jastrow_dtn_data
    db = b.jastrow_data.jastrow_dtn_data
    for k in ("alpha_indep", "beta_indep", "gamma_indep"):
        va = np.asarray(getattr(da, k))
        vb = np.asarray(getattr(db, k))
        assert not np.iscomplexobj(va), f"{k}: analytic e_L/c must be Re-projected (real), got {va.dtype}"
        np.testing.assert_allclose(va, vb, atol=atol, rtol=rtol, err_msg=f"twist DTN e_L/c block {k}")


def _mixed_twist_dtn_assert_bound(mixed_block, full_block, atol, rtol, label):
    """Apply the unchanged central budget to one real DTN source leaf."""
    assert np.abs(mixed_block - full_block).max() <= atol + rtol * np.abs(full_block).max(), label


def _mixed_twist_dtn_fidelity():
    """Independently rebuild the original eager twist/ECP analytical route."""
    from tests._precision_fidelity import mixed_fidelity_blocks, mixed_vs_full, out_of_budget_mixed_block

    names = ("alpha_indep", "beta_indep", "gamma_indep")

    def factory():
        bridge.cache_clear()
        derivatives._jitted_assemble.cache_clear()
        ham, gem = _build_twist_dtn_ham()
        r_up, r_down = _eager_inputs(gem)
        return ham, r_up, r_down

    def evaluate(case):
        ham, r_up, r_down = case
        flags = {key: True for key in derivatives._PARAM_FLAG_KEYS}
        gradient = derivatives.compute_grad_e_L(ham, r_up, r_down, jnp.eye(3), flags, analytic=True)["c"]
        parameters = ham.wavefunction_data.jastrow_data.jastrow_dtn_data
        source = gradient.jastrow_data.jastrow_dtn_data
        blocks = {}
        for name in names:
            value = np.asarray(getattr(source, name))
            assert not np.iscomplexobj(value), f"{name}: analytical e_L/c must remain real"
            assert value.shape == getattr(parameters, name).shape, f"DTN source shape changed: {name}"
            blocks[name] = value
        return {"blocks": blocks, "up": r_up, "down": r_down}

    mixed, full = mixed_vs_full(factory, evaluate)
    assert np.array_equal(mixed["up"], full["up"]) and np.array_equal(mixed["down"], full["down"])
    assert set(mixed["blocks"]) == set(full["blocks"]) == set(names)
    atol, rtol = get_tolerance_mixed_fidelity("parameter_derivative")
    nonempty = [name for name in names if full["blocks"][name].size]
    checked = []
    for name, (mixed_block, full_block) in zip(nonempty, mixed_fidelity_blocks(mixed["blocks"], full["blocks"]), strict=True):
        _mixed_twist_dtn_assert_bound(mixed_block, full_block, atol, rtol, f"twist DTN e_L/c {name}")
        checked.append((mixed_block, full_block))
    assert checked, "twist DTN numerical source evidence would be empty"
    faulty = out_of_budget_mixed_block(*checked[0], atol=atol, rtol=rtol)
    with pytest.raises(AssertionError):
        _mixed_twist_dtn_assert_bound(faulty, checked[0][1], atol, rtol, "twist DTN excess-error control")
