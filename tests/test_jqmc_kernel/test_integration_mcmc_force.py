"""collections of unit tests."""

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

import os
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._precision import get_tolerance_min
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    _phase_free_j3_orbitals,
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.jqmc_mcmc import MCMC
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests._test_data_cache import qe_example_files_root
from tests.conftest import check_symmetric_force, force_sum_rule_observation
from tests._ecp_core_electrons import core_electrons_of

# JAX float64
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

# --------------------------------------------------------------------------- #
# Complex-twist ALL-ELECTRON (twist-AE) force fixture. The committed
# lithium_k025.h5 twist force row ("pbc-lithium-k025-ewald") is ECP; it never
# exercises the AE electron-ion force branch (bare 1/r e-i + Ewald, NO ECP
# local/non-local) under a complex Bloch orbital. The genuine twist-AE fixture is
# the committed QE save H-ae_k025 (hydrogen all-electron, k=(1/4,1/4,1/4)), built
# AT TEST TIME via the QE converter (temporary, discarded -- NO derived h5). Plain
# pGTO projection -> AOs_cart_pgto_complex_data leaf; ecp_by_element=None -> AE. #
# --------------------------------------------------------------------------- #
import functools  # noqa: E402

_H_AE_SAVE = os.path.join(qe_example_files_root(), "H-ae_k025", "out", "h_k025.save")
_H_AE_UPF = os.path.join(_H_AE_SAVE, "H.coulomb-ae.UPF")
_H_AE_AVAILABLE = all(
    os.path.isfile(os.path.join(_H_AE_SAVE, f)) for f in ("data-file-schema.xml", "wfc1.hdf5", "H.coulomb-ae.UPF")
)
requires_h_ae = pytest.mark.skipif(not _H_AE_AVAILABLE, reason="H-ae_k025 QE all-electron twist fixture not present")


@functools.lru_cache(maxsize=1)
def _build_twist_ae_ham() -> Hamiltonian_data:
    """Complex-twist ALL-ELECTRON Hamiltonian (Jastrow off, MO-geminal) from the committed QE
    H-ae_k025 output, built once per process. Mirrors test_integration_qe_non_gamma.py: plain
    pGTO projection at the non-Gamma twist (complex cart-pGTO leaf) wrapped with an all-electron
    Coulomb potential (ecp_by_element=None -> bare 1/r e-i + Ewald, no ECP). Jastrow off keeps the
    position-deriv AD tape small (mirrors the lithium_k025-ewald row)."""
    from jqmc.pw_pgto_projection import project_pw_to_pgto
    from jqmc.qe_to_jqmc import build_hamiltonian_from_wavefunction, build_wavefunction_from_projection
    from jqmc.qe_wrapper import load_qe_reference
    from jqmc.upf_wrapper import UPFExponentInit

    ref = load_qe_reference(_H_AE_SAVE)
    g_max = float(np.sqrt(2.0 * ref.ecutwfc))
    strat = UPFExponentInit({"H": _H_AE_UPF}, g_target=g_max)
    shells = [strat.shells_for_element(lbl) for lbl in ref.atom_labels]
    proj = project_pw_to_pgto(ref, shells)
    sd, aos, gem = build_wavefunction_from_projection(ref, proj, kind="pgto")
    return build_hamiltonian_from_wavefunction(sd, aos, gem, j1_param=None, j2_param=None, ecp_by_element=None)


test_mcmc_force_params = [
    pytest.param(
        "H2_ecp_ccpvtz_cart.h5",
        {
            "jastrow_1b_param": True,
            "jastrow_2b_param": True,
            "jastrow_3b_param": True,
            "jastrow_nn_param": False,
            "core_electrons": (0, 0),
        },
        id="ecp",
    ),
    pytest.param(
        "H2_ae_ccpvtz_cart.h5",
        {
            "jastrow_1b_param": True,
            "jastrow_2b_param": True,
            "jastrow_3b_param": False,
            "jastrow_nn_param": False,
            "core_electrons": (0, 0),
        },
        id="ae",
    ),
]

# Multi-atom (periodic) systems validated via the translational force sum rule.
# lithium.h5 (2-atom Li ccECP) is used here instead of silicon.h5 (8-atom Si
# ccECP-ccpvtz): the silicon force-grad path hits ~67 GB AD-tape peak per walker on a
# 256 GB host and runs out of memory. lithium keeps the PBC + ECP code-path coverage
# at a fraction of the memory.
# Rows marked ``heavy`` exceed the local 60s/test budget under
# JIT warm-up (PBC ECP + full Jastrow / complex Bloch JIT compile).
_heavy = pytest.mark.heavy
# (trexio_file, jastrow_parameters)
test_mcmc_force_sum_rule_params = [
    pytest.param(
        "lithium.h5",
        {
            "jastrow_1b_param": True,
            "jastrow_2b_param": True,
            "jastrow_3b_param": True,
            "jastrow_nn_param": False,
            "core_electrons": (2,) * 2,  # Li ccECP: 1 valence / 2 core per atom
        },
        id="pbc-lithium-ewald",
        marks=_heavy,
    ),
    # Complex-Bloch fixture (twist at k_frac = (0.25, 0, 0)): exercises the
    # complex force path through the Re-inside-grad pattern. Jastrow off keeps the AD-tape memory below OOM.
    pytest.param(
        "lithium_k025.h5",
        {
            "jastrow_1b_param": False,
            "jastrow_2b_param": False,
            "jastrow_3b_param": False,
            "jastrow_nn_param": False,
        },
        id="pbc-lithium-k025-ewald",
        marks=_heavy,
    ),
]


def _compute_mcmc_force(trexio_file: str, jastrow_parameters: dict):
    """Build the WF/Hamiltonian, run VMC, and return ``(force_mean, force_std)``."""
    (
        structure_data,
        aos_data,
        _,
        _,
        geminal_mo_data,
        coulomb_potential_data,
    ) = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
        store_tuple=True,
    )

    # Jastrow setup from parameters.
    # ``jastrow_dtn`` (a ``_DTN_PRESET_MAP`` key, e.g. "u_chi_f") selects the
    # Drummond-Towler-Needs J_DTN Jastrow and is mutually exclusive with the Pade
    # J1/J2/J3 family. It exercises the DTN f-channel position-derivative path that
    # atomic forces differentiate (e_L w.r.t. R_atoms via
    # ``compute_grads_and_laplacian_Jastrow_DTN``) -- a path no other force fixture
    # covers (the Pade fixtures carry no ``Jastrow_DTN_data``).
    jastrow_dtn_preset = jastrow_parameters.get("jastrow_dtn", None)
    if jastrow_dtn_preset is not None:
        use_u, use_chi, use_f = {
            "u": (True, False, False),
            "u_chi": (True, True, False),
            "u_f": (True, False, True),
            "u_chi_f": (True, True, True),
        }[jastrow_dtn_preset]
        atomic_numbers = np.asarray(structure_data.atomic_numbers, dtype=np.float64)
        core_electrons = np.asarray(
            jastrow_parameters.get("core_electrons", (0.0,) * len(atomic_numbers)),
            dtype=np.float64,
        )
        z_eff_per_species = tuple(
            float(np.mean((atomic_numbers - core_electrons)[atomic_numbers == Z])) for Z in np.unique(atomic_numbers)
        )
        pseudized_per_species = tuple(
            bool(np.all(coulomb_potential_data.pseudized_per_atom[atomic_numbers == Z])) for Z in np.unique(atomic_numbers)
        )
        r_ins = structure_data._wigner_seitz_inscribed_radius()
        jastrow_dtn_data = Jastrow_DTN_data.build(
            structure_data=structure_data,
            z_eff_per_species=z_eff_per_species,
            pseudized_per_species=pseudized_per_species,
            use_u=use_u,
            use_chi=use_chi,
            use_f=use_f,
            cutoff_smoothness=4,
            n_order_u=4,
            n_order_chi=4,
            n_order_eN=2,
            n_order_ee=2,
            cutoff_ratio_u=2.0 / r_ins,
            cutoff_ratio_chi=2.0 / r_ins,
            cutoff_ratio_f=1.0 / (0.5 * r_ins),
            random_init=True,
            seed=42,
        )
        jastrow_data = Jastrow_data(jastrow_dtn_data=jastrow_dtn_data)
        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )
        return _run_mcmc_force(hamiltonian_data)

    jastrow_1b_param = jastrow_parameters.get("jastrow_1b_param", False)
    jastrow_2b_param = jastrow_parameters.get("jastrow_2b_param", False)
    jastrow_3b_param = jastrow_parameters.get("jastrow_3b_param", False)

    if jastrow_1b_param:
        core_electrons = jastrow_parameters.get("core_electrons", (0, 0))
        jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=0.5, structure_data=structure_data, core_electrons=tuple(core_electrons), jastrow_1b_type="pade"
        )
    else:
        jastrow_onebody_data = None

    if jastrow_2b_param:
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=0.5, jastrow_2b_type="exp", structure_data=structure_data
        )
    else:
        jastrow_twobody_data = None

    if jastrow_3b_param:
        jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=_phase_free_j3_orbitals(aos_data), random_init=True, random_scale=1.0e-3
        )
    else:
        jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=_phase_free_j3_orbitals(aos_data)
        )

    # define data
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=jastrow_onebody_data,
        jastrow_two_body_data=jastrow_twobody_data,
        jastrow_three_body_data=jastrow_threebody_data,
    )

    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)

    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )

    return _run_mcmc_force(hamiltonian_data)


def _run_mcmc_force(hamiltonian_data):
    """Run VMC with ``comput_position_deriv=True`` and return ``(force_mean, force_std)``."""
    num_mcmc_warmup_steps = 5
    num_mcmc_bin_blocks = 5
    mcmc_seed = 34356

    mcmc = MCMC(
        hamiltonian_data=hamiltonian_data,
        Dt=2.0,
        mcmc_seed=mcmc_seed,
        num_walkers=2,
        comput_position_deriv=True,
        comput_log_WF_param_deriv=False,
        comput_e_L_param_deriv=False,
        epsilon_AS=1.0e-2,
    )
    mcmc.run(num_mcmc_steps=20)
    mcmc.get_E(
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )
    force_mean, force_std = mcmc.get_aF(
        num_mcmc_warmup_steps=num_mcmc_warmup_steps,
        num_mcmc_bin_blocks=num_mcmc_bin_blocks,
    )
    return force_mean, force_std


class TestMCMCForce:
    @pytest.mark.heavy
    @pytest.mark.parametrize("trexio_file, jastrow_parameters", test_mcmc_force_params)
    def test_mcmc_force_with_SWCT(self, trexio_file: str, jastrow_parameters: dict):
        """Two-atom systems: forces obey Newton's third law under SWCT."""
        force_mean, force_std = _compute_mcmc_force(trexio_file, jastrow_parameters)
        # Force crosses ao_eval/jastrow_eval/det_eval/coulomb/wf_kinetic; bound by weakest zone (fp32 in mixed).
        # See [J. Chem. Phys. 156, 034101 (2022)]
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
            "strict",
        )
        check_symmetric_force(force_mean, force_std, atol, rtol)

    @pytest.mark.heavy
    @pytest.mark.parametrize(
        "trexio_file, jastrow_parameters",
        test_mcmc_force_sum_rule_params,
    )
    def test_mcmc_force_sum_rule_with_SWCT(self, trexio_file: str, jastrow_parameters: dict):
        """Multi-atom periodic systems: forces obey the translational sum rule under SWCT.

        The Ewald baseline (regression guard): ``sum_I F_I == 0`` to the strict tolerance,
        since Ewald is translationally invariant in the ion positions.
        """
        force_mean, _force_std = _compute_mcmc_force(trexio_file, jastrow_parameters)
        # Force crosses ao_eval/jastrow_eval/det_eval/coulomb/wf_kinetic; bound by weakest zone (fp32 in mixed).
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
            "strict",
        )
        force_values, force_sum, zero, atol_eff = force_sum_rule_observation(force_mean, atol)
        assert np.all(np.isfinite(force_values))
        np.testing.assert_allclose(force_sum, zero, atol=atol_eff, rtol=rtol)

    @pytest.mark.heavy
    def test_mcmc_force_sum_rule_with_DTN(self):
        """J_DTN (Drummond-Towler-Needs) atomic forces obey the translational sum rule.

        Closes the coverage gap documented in ``test_integration_mcmc_stress.py``
        (``DTN + position-deriv``): every other force/stress fixture carries NO
        ``Jastrow_DTN_data``, so the DTN f-channel POSITION-derivative path -- which
        atomic forces differentiate (``d e_L / d R_atom`` through
        ``compute_grads_and_laplacian_Jastrow_DTN``) -- was uncovered for force VALUES.

        This is the regression guard for the CPU DTN f-channel FFI: if its force path
        is broken -- either the custom_jvp closes over trace-level position arrays
        (R_carts / cell), which crashes the linearize lowering ("No constant handler"),
        OR the positions are frozen to a numpy constant, which silently zeroes
        ``d/d R_carts`` and breaks the forces -- this test catches it. Correct forces
        satisfy ``sum_I F_I == 0`` (SWCT partition of unity); a broken DTN force does
        not. lithium.h5 = 2-atom Li ccECP (ECP -> no AE cusp).
        """
        force_mean, _force_std = _compute_mcmc_force("lithium.h5", {"jastrow_dtn": "u_chi_f"})
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
            "strict",
        )
        force_values, force_sum, zero, atol_eff = force_sum_rule_observation(force_mean, atol)
        assert np.all(np.isfinite(force_values))
        np.testing.assert_allclose(force_sum, zero, atol=atol_eff, rtol=rtol)

    @pytest.mark.heavy
    @requires_h_ae
    def test_mcmc_force_sum_rule_twist_ae(self):
        """Complex-twist ALL-ELECTRON atomic forces obey the translational sum rule under SWCT.

        The AE companion of the ECP ``pbc-lithium-k025-ewald`` row: the QE H-ae_k025 fixture
        (hydrogen all-electron at k=(1/4,1/4,1/4), built at test time) drives the AE electron-ion
        force branch (bare 1/r e-i + Ewald, NO ECP local/non-local) under complex Bloch orbitals --
        a code path no committed twist force fixture (all ECP) covers. Jastrow off (as the k025-ewald
        row) keeps the position-deriv AD tape small. Ewald and the AE 1/r e-i are translationally
        invariant in the ion positions, so ``sum_I F_I == 0`` must hold to the strict tolerance."""
        force_mean, _force_std = _run_mcmc_force(_build_twist_ae_ham())
        atol, rtol = get_tolerance_min(
            ("ao_eval", "jastrow_eval", "det_eval", "coulomb", "wf_kinetic"),
            "strict",
        )
        force_values, force_sum, zero, atol_eff = force_sum_rule_observation(force_mean, atol)
        assert np.all(np.isfinite(force_values))
        np.testing.assert_allclose(force_sum, zero, atol=atol_eff, rtol=rtol)

    @pytest.mark.heavy
    def test_mcmc_force_without_SWCT(self):
        """Test MCMC force without SWCT (use_swct=False)."""
        trexio_file = "H2_ecp_ccpvtz_cart.h5"
        (
            structure_data,
            aos_data,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=0.5, structure_data=structure_data, core_electrons=(0, 0), jastrow_1b_type="pade"
        )
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp")
        jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=_phase_free_j3_orbitals(aos_data)
        )
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=jastrow_onebody_data,
            jastrow_two_body_data=jastrow_twobody_data,
            jastrow_three_body_data=jastrow_threebody_data,
        )
        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )

        mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=2.0,
            mcmc_seed=34356,
            num_walkers=2,
            comput_position_deriv=True,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            epsilon_AS=1.0e-2,
            use_swct=False,
        )
        mcmc.run(num_mcmc_steps=20)
        mcmc.get_E(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5)
        force_mean, force_std = mcmc.get_aF(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5)

        # Forces should be finite (no NaN/Inf)
        assert not np.any(np.isnan(np.array(force_mean))), "NaN detected in force_mean"
        assert not np.any(np.isnan(np.array(force_std))), "NaN detected in force_std"
        assert np.all(np.isfinite(np.array(force_mean))), "Inf detected in force_mean"

    @pytest.mark.parametrize(
        "with_nn",
        [pytest.param(False, id="open-shell"), pytest.param(True, id="open-shell-nn")],
    )
    @pytest.mark.heavy
    def test_mcmc_force_open_shell_finite(self, with_nn: bool):
        """Force evaluation must stay finite for open-shell systems with J2 active.

        Regression guard: dense (N,N) up-up/dn-dn pair sums in J2 trigger
        ``0 * inf = NaN`` on the i==j diagonal under second-order AD when num_up>1
        (or num_dn>1). H2 (n_up=n_dn=1) does not exercise this path; Li
        (n_up=2, n_dn=1) does.
        """
        trexio_file = "Li_ae_ccpvdz_cart.h5"
        (
            structure_data,
            aos_data,
            _,
            _,
            geminal_mo_data,
            coulomb_potential_data,
        ) = read_trexio_file(
            trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", trexio_file),
            store_tuple=True,
        )

        jastrow_onebody_data = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=0.5,
            structure_data=structure_data,
            core_electrons=core_electrons_of(coulomb_potential_data, len(structure_data.atomic_numbers)),
            jastrow_1b_type="exp",
        )
        jastrow_twobody_data = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp")
        jastrow_threebody_data = Jastrow_three_body_data.init_jastrow_three_body_data(
            orb_data=_phase_free_j3_orbitals(aos_data), random_init=True, random_scale=1.0e-3
        )
        jastrow_data = Jastrow_data(
            jastrow_one_body_data=jastrow_onebody_data,
            jastrow_two_body_data=jastrow_twobody_data,
            jastrow_three_body_data=jastrow_threebody_data,
        )
        wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            coulomb_potential_data=coulomb_potential_data,
            wavefunction_data=wavefunction_data,
        )

        mcmc = MCMC(
            hamiltonian_data=hamiltonian_data,
            Dt=2.0,
            mcmc_seed=34356,
            num_walkers=2,
            comput_position_deriv=True,
            comput_log_WF_param_deriv=False,
            comput_e_L_param_deriv=False,
            epsilon_AS=1.0e-2,
        )
        mcmc.run(num_mcmc_steps=20)
        mcmc.get_E(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5)
        force_mean, force_std = mcmc.get_aF(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=5)

        assert not np.any(np.isnan(np.array(force_mean))), "NaN detected in force_mean"
        assert not np.any(np.isnan(np.array(force_std))), "NaN detected in force_std"
        assert np.all(np.isfinite(np.array(force_mean))), "Inf detected in force_mean"


# ---------------------------------------------------------------------------
# DTN + position-deriv regression (PBC force path).
#
# Sibling of ``test_integration_mcmc_stress.py::test_stress_HF_PP_pgto_with_DTN_finite``.
# A NaN on both forces and stress on a PBC DTN + pGTO wavefunction with
# ``comput_position_deriv=True`` comes from a single ``sqrt(0)`` backward in ``_smooth_periodic_distance``
# reached through the inner ``jax.grad`` / ``jax.hessian`` taken inside
# ``_dtn_scalar_dist_with_derivs`` at the same-spin u-term self-pair
# (``u = r_i - r_i = 0``).
#
# This NaN was invisible to the existing force suite because every PBC
# force fixture (``lithium.h5``) carried only ``Jastrow_three_body_data``
# -- never ``Jastrow_DTN_data`` -- and every
# DTN-bearing suite (``test_jastrow_dtn.py``,
# ``test_param_grad_mask_production_path.py``) ran with
# ``comput_position_deriv=False``. The cross-product was silently
# uncovered.
#
# Implementation note: this test exercises the PRODUCTION bridge r/R
# position-grad closures ``bridge.batched_grad_e_L`` /
# ``bridge.batched_grad_ln_Psi`` (``wrt=("r", "R")``,
# ``with_geminal_inv=True``, ``analytic=_ANALYTIC_GRAD_ENABLED``) -- the
# SAME closures ``MCMC.run`` builds via ``_cached_grad_e_L_rR`` /
# ``_cached_grad_ln_psi_rR`` per step when ``comput_position_deriv=True``.
# The fused ``("r", "R")`` closures emit every force leaf (drift dr_up/dr_dn,
# HF/Pulay dR) whose finiteness the NaN regression guards. It does NOT call ``MCMC.run``
# because the kernel-level check repro-s the NaN bug directly (no MCMC
# dynamics required) and finishes in a few tens of seconds. The full
# ``MCMC.run`` numerical-accuracy check on these channels stays under the
# coverage of ``test_mcmc_force_with_SWCT`` /
# ``test_mcmc_force_sum_rule_with_SWCT`` on the molecular / lithium
# fixtures.
# ---------------------------------------------------------------------------


class TestMCMCForceRegression:
    @pytest.mark.heavy
    def test_force_kernels_pgto_with_DTN_finite(self):
        """Production position-grad closures must not NaN on a DTN-bearing PBC WF.

        Calls the production bridge r/R closures ``bridge.batched_grad_e_L`` /
        ``bridge.batched_grad_ln_Psi`` (``wrt=("r", "R")``,
        ``with_geminal_inv=True``) on a DTN + pGTO 4-H PBC fixture and asserts
        every returned gradient leaf is finite. A gradient that flows through
        the inner ``_smooth_periodic_distance`` AD ladder would NaN at the
        u-term self-pair (``sqrt(0)`` backward); the saddle-routed
        ``jnp.where`` keeps the cotangent finite at ``c = 0``.

        HF + Pulay + de_L/dr are exercised together in a single test to dodge
        the pre-existing ``Jastrow_DTN_data`` pytree-metadata-equality
        JIT-cache crash when a second DTN-bearing wavefunction is built in
        the same pytest session (see the corresponding NOTE in
        ``test_param_grad_mask_production_path.py`` and the matching stress
        test ``test_stress_HF_PP_pgto_with_DTN_finite``).
        """
        import jax.numpy as jnp

        from jqmc.derivatives import bridge
        from jqmc.jqmc_mcmc import _ANALYTIC_GRAD_ENABLED, _geminal_inv_batched

        here = os.path.dirname(__file__)
        if here not in sys.path:
            sys.path.insert(0, here)
        from test_param_grad_mask_production_path import _build_pbc_ham, _pick_electron_positions

        hamiltonian_data = _build_pbc_ham("pgto", include_dtn=True)
        r_up, r_dn = _pick_electron_positions(hamiltonian_data, "pgto", seed=11)
        r_up_b = r_up[None, ...]
        r_dn_b = r_dn[None, ...]
        RT_b = jnp.eye(3, dtype=jnp.float64)[None, ...]

        # Running-SM geminal inverse batch threaded into the with_geminal_inv=True
        # bridge closures (production shape -- the driver reuses the running inverse).
        ginv = _geminal_inv_batched(hamiltonian_data.wavefunction_data.geminal_data, r_up_b, r_dn_b)[1]

        # de_L r/R: fused drift (dr_up, dr_dn) + HF/Pulay force (dR) in ONE trace,
        # via the production bridge closure (returns pure position arrays).
        grad_e_L_r_up, grad_e_L_r_dn, grad_e_L_R = bridge.batched_grad_e_L(
            wrt=("r", "R"), with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED
        )(hamiltonian_data, r_up_b, r_dn_b, RT_b, ginv)
        assert np.isfinite(np.asarray(grad_e_L_R)).all(), f"de_L/dR has non-finite entries: {np.asarray(grad_e_L_R)}"
        assert np.isfinite(np.asarray(grad_e_L_r_up)).all(), f"de_L/dr_up has non-finite entries: {np.asarray(grad_e_L_r_up)}"
        assert np.isfinite(np.asarray(grad_e_L_r_dn)).all(), f"de_L/dr_dn has non-finite entries: {np.asarray(grad_e_L_r_dn)}"

        # Pulay ln|Psi| r/R: fused dR (Pulay force) + dr_up + dr_dn via the
        # production bridge closure.
        grad_ln_Psi_R, grad_ln_Psi_r_up, grad_ln_Psi_r_dn = bridge.batched_grad_ln_Psi(
            wrt=("r", "R"), with_geminal_inv=True, analytic=_ANALYTIC_GRAD_ENABLED
        )(hamiltonian_data, r_up_b, r_dn_b, ginv)
        assert np.isfinite(np.asarray(grad_ln_Psi_R)).all(), f"dln_Psi/dR has non-finite entries: {np.asarray(grad_ln_Psi_R)}"
        assert np.isfinite(np.asarray(grad_ln_Psi_r_up)).all(), (
            f"dln_Psi/dr_up has non-finite entries: {np.asarray(grad_ln_Psi_r_up)}"
        )
        assert np.isfinite(np.asarray(grad_ln_Psi_r_dn)).all(), (
            f"dln_Psi/dr_dn has non-finite entries: {np.asarray(grad_ln_Psi_r_dn)}"
        )


if __name__ == "__main__":
    from logging import Formatter, StreamHandler, getLogger

    logger = getLogger("jqmc")
    logger.setLevel("INFO")
    stream_handler = StreamHandler()
    stream_handler.setLevel("INFO")
    handler_format = Formatter("%(name)s - %(levelname)s - %(lineno)d - %(message)s")
    stream_handler.setFormatter(handler_format)
    logger.addHandler(stream_handler)
