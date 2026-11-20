"""Fixed-grid complex ECP sources and independent finite-sample optimizer references."""

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

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import (
    ATOL_FD_VS_ANALYTIC_TIGHT,
    RTOL_FD_VS_ANALYTIC_TIGHT,
    ATOL_PYTHON_FP64_LOOSE,
    RTOL_PYTHON_FP64_LOOSE,
    NN_default,
    Nv_default,
)
from jqmc.coulomb_potential import (
    compute_bare_coulomb_potential,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors_holomorphic,
)
from jqmc.derivatives import bridge
from jqmc.determinant import compute_geminal_all_elements
from jqmc.hamiltonians import Hamiltonian_data, compute_local_energy
from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
from jqmc.jqmc_optimizer import ParameterSelection
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, evaluate_wavefunction

from .complex_opt_ground_truth_harness import _FD_H, _fixed_configs

RT = np.eye(3)
SELECTION = ParameterSelection(opt_J1_param=False, opt_J2_param=True, opt_J3_param=False, opt_geminal=True)
MASK = SELECTION.variational_block_kwargs()
STEMS = {"gamma": "lithium_cart.h5", "twist": "lithium_k025_cart.h5"}
ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "det_eval",
    "det_grad_lap",
    "jastrow_eval",
    "jastrow_grad_lap",
    "coulomb",
    "local_energy",
)


def assert_same_mode(actual, reference, level, label):
    """Use the existing element-wise zone budget without a fitted scale or floor."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape, label
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)), label
    assert level == ("medium" if is_mixed_precision_enabled() else "strict"), label
    atol, rtol = get_tolerance_min(ZONES, "medium" if is_mixed_precision_enabled() else "strict")
    for part in (np.real, np.imag):
        np.testing.assert_allclose(part(actual), part(reference), atol=atol, rtol=rtol, equal_nan=False, err_msg=label)


def assert_fd(actual, reference, label):
    """Compare real and imaginary parts separately at the existing full-only FD budget."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape, label
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)), label
    for part in (np.real, np.imag):
        np.testing.assert_allclose(
            part(actual),
            part(reference),
            atol=ATOL_FD_VS_ANALYTIC_TIGHT,
            rtol=RTOL_FD_VS_ANALYTIC_TIGHT,
            equal_nan=False,
            err_msg=label,
        )


def assert_algebra(actual, reference, label):
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape, label
    assert np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)), label
    np.testing.assert_allclose(
        actual, reference, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_LOOSE, equal_nan=False, err_msg=label
    )


def build_ham(kind):
    """Committed lithium ECP Gamma/twist with untied, deterministically perturbed MOs and J2."""
    path = Path(__file__).parents[1] / "trexio_example_files" / STEMS[kind]
    structure, _ao, _up, _dn, gem, cp = read_trexio_file(str(path), store_tuple=True)
    rng = np.random.default_rng(11)
    up, dn = np.asarray(gem.orb_data_up_spin.mo_coefficients), np.asarray(gem.orb_data_dn_spin.mo_coefficients)
    scale = 0.10 * np.max(np.abs(up))
    if kind == "twist":
        perturb_up = scale * (rng.standard_normal(up.shape) + 1j * rng.standard_normal(up.shape))
        perturb_dn = scale * (rng.standard_normal(dn.shape) + 1j * rng.standard_normal(dn.shape))
    else:
        perturb_up, perturb_dn = scale * rng.standard_normal(up.shape), scale * rng.standard_normal(dn.shape)
    gem = gem.replace(
        time_reversal=False,
        orb_data_up_spin=gem.orb_data_up_spin.replace(mo_coefficients=up + perturb_up),
        orb_data_dn_spin=gem.orb_data_dn_spin.replace(mo_coefficients=dn + perturb_dn),
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
        jastrow_2b_param=0.8,
        jastrow_2b_type="exp",
        structure_data=structure,
    )
    wf = Wavefunction_data(jastrow_data=Jastrow_data(jastrow_two_body_data=j2), geminal_data=gem)
    return Hamiltonian_data(structure_data=structure, wavefunction_data=wf, coulomb_potential_data=cp)


def displace(wf, blocks, flat):
    """Write ambient parameter leaves independently of optimizer update/gradient routines."""
    jd, gd, offset = wf.jastrow_data, wf.geminal_data, 0
    assert not gd.time_reversal
    for block in blocks:
        delta = jnp.asarray(flat[offset : offset + block.size]).reshape(block.shape)
        offset += block.size
        if block.field == "real":
            delta = jnp.real(delta)
        if block.name == "j2_param":
            j2 = jd.jastrow_two_body_data
            jd = jd.replace(jastrow_two_body_data=j2.replace(jastrow_2b_param=j2.jastrow_2b_param + delta))
        elif block.name == "mo_coefficients_up":
            mo = gd.orb_data_up_spin
            gd = gd.replace(orb_data_up_spin=mo.replace(mo_coefficients=mo.mo_coefficients + delta))
        elif block.name == "mo_coefficients_dn":
            mo = gd.orb_data_dn_spin
            gd = gd.replace(orb_data_dn_spin=mo.replace(mo_coefficients=mo.mo_coefficients + delta))
        else:
            raise AssertionError(block.name)
    return wf.replace(jastrow_data=jd, geminal_data=gd)


@jax.jit
def reference_values(ham, up, dn):
    """Assemble the complex reference explicitly, without subtracting production ECP."""

    def one(u, d):
        wf, cp = ham.wavefunction_data, ham.coulomb_potential_data
        coordinates = jnp.concatenate((u.ravel(), d.ravel()))

        def psi_at(flat):
            return evaluate_wavefunction(wf, flat[: u.size].reshape(u.shape), flat[u.size :].reshape(d.shape))

        # Complex Psi is differentiated with respect to real Cartesian coordinates;
        # differentiating ln|Psi| would silently lose the Bloch phase kinetic term.
        kinetic = -0.5 * jnp.trace(jax.jacfwd(jax.jacfwd(psi_at))(coordinates)) / psi_at(coordinates)
        bare = compute_bare_coulomb_potential(cp, u, d)
        local = compute_ecp_local_parts_all_pairs(cp, u, d)
        nonlocal_ecp = compute_ecp_non_local_parts_nearest_neighbors_holomorphic(
            cp, wf, u, d, RT, NN=NN_default, Nv=Nv_default
        )[3]
        return (
            kinetic + bare + local + nonlocal_ecp,
            evaluate_wavefunction(wf, u, d),
            nonlocal_ecp,
        )

    return jax.vmap(one)(up, dn)


@jax.jit
def values(ham, up, dn):
    production = jax.vmap(lambda u, d: compute_local_energy(ham, u, d, RT))(up, dn)
    return (production, *reference_values(ham, up, dn))


def source_columns(ham, blocks, up, dn, quantity):
    fn = bridge._batched_c_source(quantity, mask=MASK)
    args = (ham, up, dn) if quantity == "ln_Psi" else (ham, up, dn, np.broadcast_to(RT, (len(up), 3, 3)).copy())
    raw = fn(*args)
    sources = bridge._collect_c_param_derivative_sources(ham.wavefunction_data, raw, blocks, len(up), MASK)
    columns = []
    for block, source in zip(blocks, sources, strict=True):
        assert block.derivative_policy == ("direct_real" if block.field == "real" else "holomorphic")
        columns.append(np.asarray(source.direct if block.field == "real" else source.d_p).reshape(len(up), block.size))
    return np.concatenate(columns, axis=1)


@dataclass
class Case:
    ham: object
    blocks: object
    up: np.ndarray
    dn: np.ndarray
    O: np.ndarray
    D: np.ndarray
    base: tuple

    @property
    def size(self):
        return sum(block.size for block in self.blocks)

    def shifted_values(self, direction, step):
        wf = displace(self.ham.wavefunction_data, self.blocks, step * direction)
        return values(self.ham.replace(wavefunction_data=wf), self.up, self.dn)

    def autodiff(self, direction):
        def independent_values(step):
            wf = displace(self.ham.wavefunction_data, self.blocks, step * direction)
            return reference_values(self.ham.replace(wavefunction_data=wf), self.up, self.dn)[:2]

        _, derivatives = jax.jvp(independent_values, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
        return np.asarray(derivatives[0]), np.asarray(derivatives[1]) / self.base[2]

    def finite_difference(self, direction):
        plus, minus = self.shifted_values(direction, _FD_H), self.shifted_values(direction, -_FD_H)
        half_plus, half_minus = self.shifted_values(direction, _FD_H / 2), self.shifted_values(direction, -_FD_H / 2)
        derivatives = []
        for p, m, hp, hm in zip(plus, minus, half_plus, half_minus, strict=True):
            coarse, fine = (np.asarray(p) - np.asarray(m)) / (2 * _FD_H), (np.asarray(hp) - np.asarray(hm)) / _FD_H
            derivatives.append((4 * fine - coarse) / 3)
        return derivatives[0], derivatives[1], derivatives[2] / self.base[2]

    def probes(self):
        offset = 0
        for block in self.blocks:
            unit = np.zeros(self.size, complex)
            unit[offset] = 1
            yield block.name + "/index0/x", unit, self.D[:, offset], self.O[:, offset]
            if block.field == "complex":
                yield block.name + "/index0/y", 1j * unit, 1j * self.D[:, offset], 1j * self.O[:, offset]
            offset += block.size


def make_case(ham, blocks, up=None, dn=None):
    if up is None:
        up, dn = _fixed_configs(ham, 8)
    base = tuple(np.asarray(v) for v in values(ham, up, dn))
    matrices = jax.vmap(lambda u, d: compute_geminal_all_elements(ham.wavefunction_data.geminal_data, u, d))(up, dn)
    conditions = np.linalg.cond(np.asarray(matrices))
    assert np.all(np.isfinite(conditions)) and np.all(np.abs(base[2]) > 0)
    return Case(
        ham,
        blocks,
        np.asarray(up),
        np.asarray(dn),
        source_columns(ham, blocks, up, dn, "ln_Psi"),
        source_columns(ham, blocks, up, dn, "e_L"),
        base,
    )
