"""Shared harness for the complex-WF optimization ground-truth suite.

Normative conventions: ``doc/notes/complex_gradients.md``,
``doc/notes/derivatives.md``, and ``doc/notes/wavefunction_optimization.md``.
These tests are DETERMINISTIC (fixed electron
configurations, no MCMC dynamics, fixed seeds used only to synthesize generic
configurations/perturbations) and IMPLEMENTATION-INDEPENDENT: every reference
is either a finite-difference of a directly-evaluated energy functional or a
few transparent lines of dense numpy transcribing the pinned math -- never a
prod-vs-debug twin comparison.

The central exact identity (T1)
-------------------------------
Fix a sample set {R_i} and define the reweighted discrete energy functional

    E(c) = sum_i w_i(c) e_L,i(c) / sum_i w_i(c),   w_i(c) = |Psi_c(R_i)|^2

(complex-valued at finite sampling since e_L is complex at a twist). For a
variational parameter c on which Psi depends HOLOMORPHICALLY (untied complex
MO coefficients; the fixtures below untie the closed-shell time-reversal
pairing to get such blocks), e_L = H Psi / Psi is holomorphic in c, so
d e_L / d c* == 0 IDENTICALLY -- not merely in expectation. The quotient rule
then gives the finite-sample EXACT identity

    f_c := -2 dE/dc* = -(dE/dx + i dE/dy) = -2 <w conj(O_c - <O_c>) (e_L - <e_L>)>

with zero statistical residue: a central finite difference of E over Re(c) and
Im(c) pins the complex generalized force to FD truncation error. The term that
usually needs the Hermiticity/completeness argument (<de_L/dc>) enters only
dE/dc, never dE/dc*. For REAL parameters (Jastrow) the analogous exact
decomposition is  dE/dx = 2<dO (e-<e>)> + <de_L/dx>,  which T1 also verifies
(the covariance part is what the optimizer keeps; the <de_L/dx> part is the
finite-sample term that Hermiticity kills in expectation).

The oracles are split into one module per oracle type
(force / SR solver / linear method / gauge) so the CI matrix schedules them
as independent heavy jobs; this harness carries the shared fixtures and
dense references.
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

import jax
import numpy as np
import scipy.linalg

from jqmc._setting import (
    RTOL_FD_VS_ANALYTIC_TIGHT,
)
from jqmc.derivatives import bridge
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction

_TREXIO = os.path.join(os.path.dirname(__file__), "..", "trexio_example_files")

# Fixed sample-set size. Large enough that the jackknife product-term bias of
# get_gF (an exact, deterministic O(1/N) deviation from the plain covariance --
# see _N_JACKKNIFE_RTOL) sits far below the generic Im/Re force ratio; small
# enough to keep the light tier fast.
_N_SAMPLES = 240
# get_gF returns the mean over jackknife replicates; for the nonlinear
# <e><O> product term this deviates from the plain-covariance estimator by an
# exact O(1/M) term (M = number of leave-one-out replicates = _N_SAMPLES here,
# equal unit weights, one sample per bin). 24/N is a comfortable bound.
_N_JACKKNIFE_RTOL = 24.0 / _N_SAMPLES

_FD_H = 1e-4  # central-difference step; truncation O(h^2)
_FD_RTOL = RTOL_FD_VS_ANALYTIC_TIGHT  # FD-vs-analytic agreement (O(h^2) truncation)

_MASK_KWARGS = {
    "opt_J1_param": False,
    "opt_J2_param": True,
    "opt_J3_param": False,
    "opt_geminal": True,
}

_CASES = ("gamma", "twist")
_STEMS = {
    "gamma": "H2_pbc_ae_cart.h5",  # real Gamma control -- must stay GREEN
    "twist": "H2_pbc_ae_k025_cart.h5",  # complex Bloch twist
}


# ---------------------------------------------------------------------------
# Case construction: Hamiltonian at a GENERIC parameter point + fixed samples
# ---------------------------------------------------------------------------


def _build_ham(kind: str) -> Hamiltonian_data:
    """H2 AE periodic fixture with an UNTIED SD geminal at a generic point.

    * ``time_reversal=False`` so ``mo_coefficients_up`` / ``mo_coefficients_dn``
      are independent variational blocks on which Psi is HOLOMORPHIC (the
      closed-shell tie ``C_dn = conj(C_up)`` would make Psi depend on conj(c),
      breaking the exact T1 identity for that block).
    * A deterministic perturbation moves the MOs off the SCF point so the
      force is generic -- in particular Im(f) = O(Re(f)) at the twist, which
      is what makes the RED assertions meaningful (near a stationary/gauge-
      aligned point Im(f) can be accidentally small).
    * AE (no ECP) so e_L has no quadrature-mesh/RT dependence -- the FD
      reference needs e_L to be a plain deterministic function of (ham, R).
    """
    (structure_data, _aos, _mu, _md, gem, coulomb_potential_data) = read_trexio_file(
        os.path.join(_TREXIO, _STEMS[kind]), store_tuple=True
    )
    rng = np.random.default_rng(11)
    up = np.asarray(gem.orb_data_up_spin.mo_coefficients)
    dn = np.asarray(gem.orb_data_dn_spin.mo_coefficients)
    scale = 0.10 * np.max(np.abs(up))
    if kind == "twist":
        pert_up = scale * (rng.standard_normal(up.shape) + 1j * rng.standard_normal(up.shape))
        pert_dn = scale * (rng.standard_normal(dn.shape) + 1j * rng.standard_normal(dn.shape))
    else:
        pert_up = scale * rng.standard_normal(up.shape)
        pert_dn = scale * rng.standard_normal(dn.shape)
    gem = gem.replace(
        time_reversal=False,
        orb_data_up_spin=gem.orb_data_up_spin.replace(mo_coefficients=up + pert_up),
        orb_data_dn_spin=gem.orb_data_dn_spin.replace(mo_coefficients=dn + pert_dn),
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=0.8, jastrow_2b_type="exp", structure_data=structure_data
    )
    wf = Wavefunction_data(jastrow_data=Jastrow_data(jastrow_two_body_data=j2), geminal_data=gem)
    return Hamiltonian_data(
        structure_data=structure_data,
        wavefunction_data=wf,
        coulomb_potential_data=coulomb_potential_data,
    )


def _fixed_configs(ham: Hamiltonian_data, n: int):
    """Deterministic electron configurations, kept away from coalescence
    points (a conditioning measure only -- the identities hold for any set).
    """
    cell = np.asarray(ham.structure_data.cell)
    positions = np.asarray(ham.structure_data._positions_cart_np)
    n_up = int(ham.wavefunction_data.geminal_data.num_electron_up)
    n_dn = int(ham.wavefunction_data.geminal_data.num_electron_dn)
    rng = np.random.default_rng(2026)
    r_up, r_dn = [], []
    while len(r_up) < n:
        frac = rng.uniform(0.0, 1.0, size=(n_up + n_dn, 3))
        cart = frac @ cell
        d_nuc = np.linalg.norm(cart[:, None, :] - positions[None, :, :], axis=-1)
        d_ee = np.linalg.norm(cart[:, None, :] - cart[None, :, :], axis=-1)
        np.fill_diagonal(d_ee, np.inf)
        if d_nuc.min() < 0.6 or d_ee.min() < 0.6:
            continue
        r_up.append(cart[:n_up])
        r_dn.append(cart[n_up:])
    return np.asarray(r_up), np.asarray(r_dn)


_ln_batched = jax.jit(jax.vmap(evaluate_ln_wavefunction, in_axes=(None, 0, 0)))
_eL_batched = jax.jit(jax.vmap(compute_local_energy, in_axes=(None, 0, 0, None)))
_RT_IDENTITY = np.eye(3)


def _flatten_grad_map(grad_map: dict, blocks: list, n: int) -> np.ndarray:
    """Assemble the (N, K) design matrix in block order from a name->array dict."""
    cols = []
    for block in blocks:
        arr = np.asarray(grad_map[block.name]).reshape(n, block.size)
        cols.append(arr)
    out = np.concatenate(cols, axis=1)
    dtype = np.result_type(*[c.dtype for c in cols])
    return out.astype(dtype)


class _Case:
    """All deterministic per-case data shared by the tests."""

    def __init__(self, kind: str):
        self.kind = kind
        self.ham = _build_ham(kind)
        self.wf = self.ham.wavefunction_data
        self.blocks = self.wf.get_variational_blocks(state_token=("test", 0), **_MASK_KWARGS)
        self.block_names = [b.name for b in self.blocks]
        assert self.block_names == ["j2_param", "mo_coefficients_up", "mo_coefficients_dn"], self.block_names
        mo_field = "complex" if kind == "twist" else "real"
        mo_policy = "holomorphic" if kind == "twist" else "direct_real"
        assert [(b.field, b.derivative_policy, b.sync_scope, b.tangent_spec.kind) for b in self.blocks] == [
            ("real", "direct_real", "world", "euclidean"),
            (mo_field, mo_policy, "sampler", "s_stiefel_rows"),
            (mo_field, mo_policy, "sampler", "s_stiefel_rows"),
        ]
        assert all(b.state_token == ("test", 0) for b in self.blocks)
        self.offsets = {}
        start = 0
        for b in self.blocks:
            self.offsets[b.name] = (start, start + b.size)
            start += b.size
        self.K = start

        self.r_up, self.r_dn = _fixed_configs(self.ham, _N_SAMPLES)
        self.ln0 = np.asarray(_ln_batched(self.wf, self.r_up, self.r_dn))  # (N,) real
        self.e = np.asarray(_eL_batched(self.ham, self.r_up, self.r_dn, _RT_IDENTITY))  # (N,)
        self.w = np.ones(_N_SAMPLES)

        # Production O_k = d ln|Psi| / dc and dE_k = d e_L / dc kernels
        # (bridge; production-default `analytic` route), reduced to the flat
        # block dict exactly as run() stores them.
        o_fn = bridge.batched_grad_ln_Psi(wrt="c", mask=_MASK_KWARGS)
        o_raw = o_fn(self.ham, self.r_up, self.r_dn)
        self.O_map = bridge._reduce_c_param_grads(self.wf, o_raw, _N_SAMPLES, _MASK_KWARGS)
        self.O = _flatten_grad_map(self.O_map, self.blocks, _N_SAMPLES)  # (N, K)

        de_fn = bridge.batched_grad_e_L(wrt="c", mask=_MASK_KWARGS)
        rt_batch = np.broadcast_to(_RT_IDENTITY, (_N_SAMPLES, 3, 3)).copy()
        de_raw = de_fn(self.ham, self.r_up, self.r_dn, rt_batch)
        self.dE_map = bridge._reduce_c_param_grads(self.wf, de_raw, _N_SAMPLES, _MASK_KWARGS)
        self.dE = _flatten_grad_map(self.dE_map, self.blocks, _N_SAMPLES)  # (N, K)

        # Plain-covariance references (the pinned math).
        W = self.w.sum()
        self.e_bar = np.sum(self.w * self.e) / W  # complex mean; centering cancels exactly
        self.O_bar = self.w @ self.O / W
        self.dO = self.O - self.O_bar[None, :]
        self.de = self.e - self.e_bar
        # f_k = -2 <w conj(dO_k) de>   (COMPLEX -- the pinned generalized force)
        self.f_ref = -2.0 * (self.dO.conj().T @ (self.w * self.de)) / W
        self.H_0 = float(np.real(self.e_bar))

    # -- reweighted discrete energy functional ---------------------------------

    def energy_at(self, thetas: np.ndarray) -> complex:
        """E(c0 + thetas) on the fixed samples, reweighted by |Psi|^2 ratios."""
        wf_new = self.wf.apply_block_updates(blocks=self.blocks, thetas=thetas, learning_rate=1.0)
        ham_new = Hamiltonian_data(
            structure_data=self.ham.structure_data,
            wavefunction_data=wf_new,
            coulomb_potential_data=self.ham.coulomb_potential_data,
        )
        ln = np.asarray(_ln_batched(ham_new.wavefunction_data, self.r_up, self.r_dn))
        e = np.asarray(_eL_batched(ham_new, self.r_up, self.r_dn, _RT_IDENTITY))
        w = np.exp(2.0 * (ln - self.ln0))
        return complex(np.sum(w * e) / np.sum(w))

    def fd_dE_along(self, direction: np.ndarray, h: float = _FD_H) -> complex:
        """Central-difference dE/dgamma along c0 + gamma*direction (gamma real)."""
        e_plus = self.energy_at(h * direction)
        e_minus = self.energy_at(-h * direction)
        return (e_plus - e_minus) / (2.0 * h)

    def fd_wirtinger_force(self, flat_idx: int) -> complex:
        """f_FD = -(dE/dx + i dE/dy) at parameter component ``flat_idx``."""
        unit = np.zeros(self.K, dtype=np.complex128)
        unit[flat_idx] = 1.0
        dE_dx = self.fd_dE_along(unit)
        dE_dy = self.fd_dE_along(1j * unit)
        return -(dE_dx + 1j * dE_dy)

    # -- MCMC with injected deterministic buffers -------------------------------

    def injected_mcmc(self) -> MCMC:
        m = MCMC(
            hamiltonian_data=self.ham,
            num_walkers=1,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=True,
            comput_position_deriv=False,
            random_discretized_mesh=False,
        )
        n = _N_SAMPLES
        m._MCMC__stored_w_L = np.ones((n, 1))
        m._MCMC__stored_e_L = np.asarray(self.e).reshape(n, 1)
        m._MCMC__stored_log_WF_param_grads = {
            name: np.asarray(arr).reshape(n, 1, *np.shape(arr)[1:]) for name, arr in self.O_map.items()
        }
        m._MCMC__stored_e_L_param_grads = {
            name: np.asarray(arr).reshape(n, 1, *np.shape(arr)[1:]) for name, arr in self.dE_map.items()
        }
        m._MCMC__mcmc_counter = n
        return m


_case_cache: dict[str, _Case] = {}


def get_case(kind: str) -> _Case:
    """Build (once per process) and return the deterministic case data.

    The consuming test modules each define a module-scoped ``case`` fixture
    parametrized over ``_CASES`` that delegates here, so the expensive
    fixture build is shared across the classes of one module while the
    fixture NAME stays local to each module (no cross-module conftest).
    """
    if kind not in _case_cache:
        _case_cache[kind] = _Case(kind)
    return _case_cache[kind]


def _probe_indices(case: _Case) -> list[int]:
    """A few complex MO components (both spins) with non-negligible force."""
    idx = []
    for name in ("mo_coefficients_up", "mo_coefficients_dn"):
        s, e = case.offsets[name]
        block_f = np.abs(case.f_ref[s:e])
        top = np.argsort(block_f)[::-1][:2]
        idx.extend((s + t) for t in top)
    return idx


def _lm_reference(case, subset):
    """Dense reference of the pinned LM: H_bar/S_bar assembled by
    the pinned formulas from raw samples, generalized eigh, max-|v0|^2 selection,
    delta = v[1:]/v[0]. ~15 transparent lines; no dgelscut, no MPI, no code reuse.
    """
    W = case.w.sum()
    dO = case.dO[:, subset]
    ddE = (case.dE - (case.w @ case.dE)[None, :] / W)[:, subset]
    e = case.e
    w = case.w
    f = -2.0 * (dO.conj().T @ (w * (e - case.e_bar))) / W
    S = dO.conj().T @ (w[:, None] * dO) / W
    Km = dO.conj().T @ ((w * e)[:, None] * dO) / W
    B = (w[:, None] * ddE).conj().T @ dO / W
    H = Km + B
    H = 0.5 * (H + H.conj().T)

    p = len(subset)
    dt = np.result_type(S.dtype, f.dtype)
    H_bar = np.zeros((p + 1, p + 1), dtype=dt)
    H_bar[0, 0] = case.H_0
    H_bar[1:, 0] = -0.5 * f  # ket column <psi_k|(H-E)|psi_0> = -f_k/2
    H_bar[0, 1:] = -0.5 * np.conj(f)
    H_bar[1:, 1:] = H
    S_bar = np.zeros((p + 1, p + 1), dtype=dt)
    S_bar[0, 0] = 1.0
    S_bar[1:, 1:] = S

    eigvals, eigvecs = scipy.linalg.eigh(H_bar, S_bar)  # v^dagger S_bar v = 1
    v0_sq = np.abs(eigvecs[0, :]) ** 2
    best = int(np.argmax(v0_sq))
    v = eigvecs[:, best]
    delta = v[1:] / v[0]
    return delta, float(eigvals[best]), (H_bar, S_bar)


def _rayleigh_quotient(H_bar, S_bar, delta):
    v = np.concatenate(([1.0], delta))
    return float(np.real(np.conj(v) @ H_bar @ v) / np.real(np.conj(v) @ S_bar @ v))


def _lm_subset(case):
    """Small, well-conditioned parameter subset (keeps dgelscut trivially inactive)."""
    s_up, _ = case.offsets["mo_coefficients_up"]
    s_dn, _ = case.offsets["mo_coefficients_dn"]
    top_up = list(s_up + np.argsort(np.abs(case.f_ref[slice(*case.offsets["mo_coefficients_up"])]))[::-1][:3])
    top_dn = list(s_dn + np.argsort(np.abs(case.f_ref[slice(*case.offsets["mo_coefficients_dn"])]))[::-1][:3])
    return [case.offsets["j2_param"][0]] + top_up + top_dn
