"""Closed-form physical energy checks on the actual XLA MCMC measurement route."""

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

import ast
import inspect
import textwrap

import jax
import numpy as np
import pytest

from jqmc import jqmc_mcmc as mcmc
from jqmc import hamiltonians
from jqmc._precision import get_dtype_np, get_tolerance_min
from jqmc.atomic_orbital import AOs_cart_gto_data
from jqmc.molecular_orbital import MOs_data
from jqmc.determinant import SD_data
from jqmc.structure import Structure_data
from jqmc.coulomb_potential import Coulomb_potential_data
from jqmc.wavefunction import Wavefunction_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.hamiltonians import Hamiltonian_data
from tests._fixed_sample_statistics import equal_bin_statistics

pytestmark = [pytest.mark.light, pytest.mark.with_chdir]


def _gaussian_hamiltonian(charge, exponent):
    symbol = "H" if charge == 1 else "He"
    structure = Structure_data.build(
        positions=np.zeros((1, 3)),
        pbc_flag=False,
        atomic_numbers=(charge,),
        element_symbols=(symbol,),
        atomic_labels=(symbol + "1",),
    )
    aos = AOs_cart_gto_data.build(
        structure_data=structure,
        nucleus_index=(0,),
        num_ao=1,
        num_ao_prim=1,
        orbital_indices=(0,),
        exponents=(exponent,),
        coefficients=(1.0,),
        angular_momentums=(0,),
        polynominal_order_x=(0,),
        polynominal_order_y=(0,),
        polynominal_order_z=(0,),
    )
    up = MOs_data(num_mo=1, aos_data=aos, mo_coefficients=np.ones((1, 1)))
    down = MOs_data(num_mo=charge - 1, aos_data=aos, mo_coefficients=np.ones((charge - 1, 1)))
    geminal = SD_data(num_electron_up=1, num_electron_dn=charge - 1, orb_data_up_spin=up, orb_data_dn_spin=down)
    return Hamiltonian_data(
        structure_data=structure,
        wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
        coulomb_potential_data=Coulomb_potential_data(structure_data=structure),
    )


def _zone_cast(values, zone):
    """Cast fp64-reconstructed geometry or an upstream zone output to the NumPy dtype of ``zone``.

    Precision: a mixed-precision production value is judged only against a reference
    computed in the same precision regime. Geometry (r - R, r_i - r_j) is formed
    in fp64 first and cast once to the zone that consumes it, as production does.
    """
    return np.asarray(values, dtype=np.float64).astype(get_dtype_np(zone))


def _gaussian_orbitals(diff, exponent, zone):
    """Normalized centred Cartesian s and p_x GTOs with gradients/Laplacians, all in the ``zone`` dtype.

    ``diff`` is the fp64 offset r - R. Callers take VALUES from the ``ao_eval``
    zone and gradients/Laplacians from the ``ao_grad_lap`` zone, which is
    the production split; nothing here calls production AO code.
    """
    dtype = get_dtype_np(zone)
    d = _zone_cast(diff, zone)
    a = dtype(exponent)
    r2 = np.sum(d * d, axis=-1)
    gauss = np.exp(-a * r2)
    s_norm = (dtype(2) * a / dtype(np.pi)) ** dtype(0.75)
    p_norm = s_norm * dtype(2) * np.sqrt(a)
    x = d[..., 0]
    s = s_norm * gauss
    px = p_norm * x * gauss
    unit_x = np.zeros(d.shape, dtype=dtype)
    unit_x[..., 0] = dtype(1)
    return {
        "s": s,
        "px": px,
        "s_grad": dtype(-2) * a * d * s[..., None],
        "px_grad": p_norm * gauss[..., None] * (unit_x - dtype(2) * a * x[..., None] * d),
        "s_lap": (dtype(4) * a * a * r2 - dtype(6) * a) * s,
        "px_lap": (dtype(4) * a * a * r2 - dtype(10) * a) * px,
    }


def _s_orbital_log_derivatives(points, exponent):
    """Per-electron grad ln(phi) and lap ln(phi) of the centred s GTO with the production zone split.

    The orbital value (the denominator) is evaluated in the ``ao_eval`` dtype and
    its gradient/Laplacian in the ``ao_grad_lap`` dtype; the quotients are
    formed in ``det_grad_lap``. In full mode every zone is fp64.
    """
    phi = _zone_cast(_gaussian_orbitals(points, exponent, "ao_eval")["s"], "det_grad_lap")
    derivatives = _gaussian_orbitals(points, exponent, "ao_grad_lap")
    grad = _zone_cast(derivatives["s_grad"], "det_grad_lap") / phi[..., None]
    lap = _zone_cast(derivatives["s_lap"], "det_grad_lap") / phi - np.sum(grad**2, axis=-1)
    return grad, lap


def _kinetic_elements(grad_ln_psi, lap_ln_psi):
    """Per-electron -(lap ln Psi + |grad ln Psi|^2)/2 assembled in the ``wf_kinetic`` dtype."""
    grad = _zone_cast(grad_ln_psi, "wf_kinetic")
    lap = _zone_cast(lap_ln_psi, "wf_kinetic")
    return -0.5 * (lap + np.sum(grad**2, axis=-1))


def _local_energy(up, down, charge, exponent):
    """Gaussian kinetic plus Coulomb local energy with every piece in its production zone dtype.

    Kinetic: -lap(phi)/(2 phi) per electron with phi in ``ao_eval`` and its
    derivatives in ``ao_grad_lap`` (not the simplified 3a-2a^2 r^2); Coulomb
    terms in ``coulomb``; the sum in ``local_energy``. Mathematically this is
    sum(3a-2a^2 r^2-Z/r) plus the He electron repulsion.
    """
    points = np.concatenate([up, down], axis=1)
    kinetic = np.sum(_kinetic_elements(*_s_orbital_log_derivatives(points, exponent)), axis=1)
    radii = np.linalg.norm(_zone_cast(points, "coulomb"), axis=-1)
    potential = -charge * np.sum(1 / radii, axis=1)
    if charge == 2:
        potential += 1 / np.linalg.norm(_zone_cast(up[:, 0] - down[:, 0], "coulomb"), axis=1)
    return _zone_cast(kinetic, "local_energy") + _zone_cast(potential, "local_energy")


def _run_and_compare(charge, exponent):
    ham = _gaussian_hamiltonian(charge, exponent)
    driver = mcmc.MCMC(
        hamiltonian_data=ham,
        num_walkers=3,
        mcmc_seed=34,
        num_mcmc_per_measurement=2,
        Dt=0.7,
        epsilon_AS=0.0,
        random_discretized_mesh=False,
    )
    expected_energy = []
    for _ in range(4):
        driver.run(num_mcmc_steps=1)
        up = np.asarray(driver._MCMC__latest_r_up_carts)
        down = np.asarray(driver._MCMC__latest_r_dn_carts)
        expected_energy.append(_local_energy(up, down, charge, exponent))
    expected_energy = np.asarray(expected_energy)
    actual_mean, actual_error, _, _ = driver.get_E(num_mcmc_warmup_steps=0, num_mcmc_bin_blocks=2)
    expected_mean, expected_error = equal_bin_statistics(expected_energy, 2)
    atol, rtol = get_tolerance_min(("ao_eval", "wf_kinetic", "coulomb", "local_energy"), "strict")
    for quantity, actual, expected in (
        ("local-energy", np.asarray(driver.e_L), expected_energy),
        ("mean", actual_mean, expected_mean),
        ("mean-error", actual_error, expected_error),
    ):
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity)
    return expected_energy


@pytest.mark.parametrize("charge", [1, 2])
@pytest.mark.parametrize("exponent", [0.5, 1.25])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_sampled_local_energies_match_gaussian_coulomb_formula(monkeypatch, charge, exponent):
    """Actual MCMC measurements equal a Gaussian/Coulomb formula on every sampled input.

    Synthetic open H (1up/0down) and He (1up/1down) use one Cartesian s GTO,
    SD/MO, no Jastrow/ECP, and exponent 0.5/1.25 at a single origin nucleus.
    Three walkers use the driver's seed-34 initialization and proposals with
    Dt=0.7, AS=0, two moves/measurement and four successive one-step run calls.
    No cond(G) screen or reselection. Every saved local energy is compared at
    its actual endpoint to sum(3a-2a^2 r^2-Z/r) plus the He electron repulsion,
    evaluated in the production zone dtypes: -lap(phi)/(2 phi) with the
    orbital value in ao_eval and its derivatives in ao_grad_lap, Coulomb in
    coulomb, so mixed mode is never judged against an fp64 reference.
    Pairwise bin-mean differences independently give the four-step/two-bin
    mean and reported error, not a sigma acceptance band. Use same-mode
    central strict tolerances. A GTO is not the exact hydrogen 1s eigenstate;
    neither -0.5 Ha nor zero variance is claimed.
    """
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    _run_and_compare(charge, exponent)


@pytest.mark.parametrize("kinetic_scale", [1.001])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_sampled_energy_reference_rejects_shared_kinetic_scale_fault(monkeypatch, kinetic_scale):
    """Scaling the actual fast local-energy kinetic term must fail the physical reference."""
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    _run_and_compare(2, 0.5)
    source = textwrap.dedent(inspect.getsource(hamiltonians.compute_local_energy_fast))
    original = "T.astype(dtype_jnp) + V.astype(dtype_jnp)"
    assert source.count(original) == 1
    tree = ast.parse(source.replace(original, f"{kinetic_scale!r} * T.astype(dtype_jnp) + V.astype(dtype_jnp)"))
    namespace = dict(vars(hamiltonians))
    exec(compile(tree, "<fault-injected-fast-local-energy>", "exec"), namespace)
    faulty = jax.jit(jax.vmap(namespace["compute_local_energy_fast"], in_axes=(None, 0, 0, 0, 0, 0)))
    monkeypatch.setattr(mcmc, "_jit_vmap_e_L_fast", faulty)
    with pytest.raises(AssertionError, match="local-energy"):
        _run_and_compare(2, 0.5)
