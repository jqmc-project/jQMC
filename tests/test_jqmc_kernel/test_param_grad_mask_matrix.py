"""Parameter-gradient masking matrix.

For individually exercised flags and explicitly selected combinations, ``Wavefunction_data.with_param_grad_mask``
must let gradients flow ONLY to the parameters being optimized and stop gradients
to every other leaf. This is what makes ``grad(compute_local_energy, argnums=0)``
(the de_L/dc parameter gradient over the whole Hamiltonian) compute only the wanted
derivatives -- avoiding wasted compute and, for the large B-spline control-point
tensor, OOM.

Method
------
We differentiate ``ln|Psi|`` w.r.t. a *dict of the candidate parameter arrays*
(the grad primal is plain arrays, so DTN's ``pytree_node=False`` numpy metadata is
never in the differentiated pytree). Inside the function we substitute the arrays
back into the wavefunction, THEN apply ``with_param_grad_mask`` -- so the mask sees
the substituted arrays and stop-gradients exactly the frozen ones. A zero gradient
means "frozen", a nonzero gradient means "flows".

Key invariants asserted
-----------------------
* For ``AGP_data`` (the fixture used by this file) ``mo_coefficients`` is NEVER a
  variational block: the lambda matrix is in AO basis and ``run_optimize``'s
  rank-truncated MO opt goes through ``fix_geminal_rank`` (auto-convert
  to ``AGPn_data`` at entry). The direct
  MO-coefficient gradient flow on ``AGPn_data`` / ``SD_data`` is covered
  by ``test_jqmc_mcmc.py::test_run_optimize_with_SD_data_E2E`` and
  ``::test_auto_convert_AGP_to_AGPn``.
* With no AO-basis flag set (default; e.g. Jastrow-only / DTN-only), the geminal /
  J3 orbital basis (mo_coefficients, GTO exp/coeff, ...) carries no gradient.
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

from jqmc._setting import RTOL_NONVACUITY_DISTINCTION
import os
import sys
from pathlib import Path

import jax
import numpy as np
import numpy.testing as npt
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._setting import ATOL_PYTHON_FP64_LOOSE
from jqmc._precision import get_tolerance_min
from jqmc.determinant import Geminal_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data, evaluate_ln_wavefunction
from tests._ecp_core_electrons import core_electrons_of

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

TREXIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files")


def _load(name="H2_ae_ccpvdz_cart.h5"):
    return read_trexio_file(trexio_file=os.path.join(TREXIO_DIR, name), store_tuple=True)


def _build_wf():
    """GTO-MO geminal + J1 (exp) + J2 (pade) + J3 (AO GTO) + DTN (u_chi_f)."""
    from jqmc.determinant import to_AGP

    sd, aos, _mu, _md, gem_mo, _cpd = _load()
    # The TREXIO load-time dispatch routes a closed-shell
    # ``[1]*N_occ + [0]*N_virt`` diagonal MO geminal to ``SD_data``; this
    # test pre-dates the split and exercises the ``AGP_data`` dense
    # ``lambda_matrix`` parameter-grad mask surface, so promote back to
    # ``AGP_data`` for the legacy test wiring.
    gem_mo = to_AGP(gem_mo)
    j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
        jastrow_1b_param=1.0,
        structure_data=sd,
        core_electrons=core_electrons_of(_cpd, len(sd.atomic_numbers)),
    )
    j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0)
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos, random_init=True, random_scale=0.05, seed=1)
    # Large cutoffs so the DTN terms are nonzero at the sampled electron
    # positions (otherwise the DTN gradient is trivially zero).
    dtn = Jastrow_DTN_data.build(
        structure_data=sd,
        z_eff_per_species=(1.0,),  # all-electron H2 fixture
        pseudized_per_species=(False,),
        use_u=True,
        use_chi=True,
        use_f=True,
        cutoff_smoothness=4,
        n_order_u=4,
        n_order_chi=4,
        n_order_eN=2,
        n_order_ee=2,
        L_u=5.0,
        L_chi=5.0,
        L_f=5.0,
        random_init=True,
        seed=2,
    )
    jd = Jastrow_data(
        jastrow_one_body_data=j1,
        jastrow_two_body_data=j2,
        jastrow_three_body_data=j3,
        jastrow_dtn_data=dtn,
    )
    wf = Wavefunction_data(jastrow_data=jd, geminal_data=gem_mo)

    rng = np.random.RandomState(3)
    n_up = gem_mo.num_electron_up
    n_dn = gem_mo.num_electron_dn
    centers = sd._positions_cart_np

    def pick(n):
        base = centers[rng.randint(0, centers.shape[0], size=n)]
        return jax.numpy.asarray(base + 0.5 * rng.standard_normal((n, 3)))

    return wf, pick(n_up), pick(n_dn)


def _targets(wf):
    """Plain-array dict of the candidate parameters we test gradients of."""
    jd = wf.jastrow_data
    gem = wf.geminal_data
    orb = gem.orb_data_up_spin
    is_mo = hasattr(orb, "mo_coefficients")
    gem_basis = orb.aos_data if is_mo else orb
    return {
        "j1": np.asarray(jd.jastrow_one_body_data.jastrow_1b_param),
        "j2": np.asarray(jd.jastrow_two_body_data.jastrow_2b_param),
        "j3_matrix": np.asarray(jd.jastrow_three_body_data.j_matrix),
        # The variational DOF is the per-shell radial leaf
        # (``radial.exponents`` / ``radial.coefficients``), not the derived
        # full per-AO-primitive ``exponents`` / ``coefficients`` properties.
        "j3_exp": np.asarray(jd.jastrow_three_body_data.orb_data.radial.exponents),
        "j3_coeff": np.asarray(jd.jastrow_three_body_data.orb_data.radial.coefficients),
        "dtn_alpha": np.asarray(jd.jastrow_dtn_data.alpha_indep),
        "dtn_beta": np.asarray(jd.jastrow_dtn_data.beta_indep),
        "dtn_gamma": np.asarray(jd.jastrow_dtn_data.gamma_indep),
        "lambda": np.asarray(gem.lambda_matrix),
        # Dense AO AGP has no MO leaf. Keep a deliberately unused sentinel so
        # this generic mask table still proves that no MO coordinate flows.
        "mo_coeff": np.asarray(orb.mo_coefficients) if is_mo else np.zeros((1,), dtype=np.float64),
        "gem_exp": np.asarray(gem_basis.radial.exponents),
        "gem_coeff": np.asarray(gem_basis.radial.coefficients),
    }


def _rebuild(wf, t):
    """Substitute the dict ``t`` of arrays back into ``wf`` (returns new wf)."""
    jd = wf.jastrow_data
    gem = wf.geminal_data
    orb = gem.orb_data_up_spin
    new_j1 = jd.jastrow_one_body_data.replace(jastrow_1b_param=t["j1"])
    new_j2 = jd.jastrow_two_body_data.replace(jastrow_2b_param=t["j2"])
    j3 = jd.jastrow_three_body_data
    new_j3_orb = j3.orb_data.replace(radial=j3.orb_data.radial.replace(exponents=t["j3_exp"], coefficients=t["j3_coeff"]))
    new_j3 = j3.replace(j_matrix=t["j3_matrix"], orb_data=new_j3_orb)
    dtn = jd.jastrow_dtn_data
    new_dtn = dtn.replace(alpha_indep=t["dtn_alpha"], beta_indep=t["dtn_beta"], gamma_indep=t["dtn_gamma"])
    new_jd = jd.replace(
        jastrow_one_body_data=new_j1,
        jastrow_two_body_data=new_j2,
        jastrow_three_body_data=new_j3,
        jastrow_dtn_data=new_dtn,
    )
    orb_dn = gem.orb_data_dn_spin
    if hasattr(orb, "mo_coefficients"):
        new_aos_up = orb.aos_data.replace(
            radial=orb.aos_data.radial.replace(exponents=t["gem_exp"], coefficients=t["gem_coeff"])
        )
        new_orb_up = orb.replace(aos_data=new_aos_up, mo_coefficients=t["mo_coeff"])
        new_aos_dn = orb_dn.aos_data.replace(
            radial=orb_dn.aos_data.radial.replace(exponents=t["gem_exp"], coefficients=t["gem_coeff"])
        )
        new_orb_dn = orb_dn.replace(aos_data=new_aos_dn, mo_coefficients=t["mo_coeff"])
    else:
        new_orb_up = orb.replace(radial=orb.radial.replace(exponents=t["gem_exp"], coefficients=t["gem_coeff"]))
        new_orb_dn = orb_dn.replace(radial=orb_dn.radial.replace(exponents=t["gem_exp"], coefficients=t["gem_coeff"]))
    new_gem = gem.replace(lambda_matrix=t["lambda"], orb_data_up_spin=new_orb_up, orb_data_dn_spin=new_orb_dn)
    return wf.replace(jastrow_data=new_jd, geminal_data=new_gem)


def _flows(wf, r_up, r_dn, **flags):
    """Return {name: True if gradient flows (nonzero) else False} per target."""
    t0 = {k: jax.numpy.asarray(v) for k, v in _targets(wf).items()}

    def f(t):
        wf2 = _rebuild(wf, t)
        masked = wf2.with_param_grad_mask(**flags)
        return evaluate_ln_wavefunction(masked, r_up, r_dn)

    g = jax.grad(f)(t0)
    return {k: float(np.max(np.abs(np.asarray(v)))) > 0.0 for k, v in g.items()}


_BASE = dict(
    opt_J1_param=False,
    opt_J2_param=False,
    opt_J3_param=False,
    opt_geminal=False,
    opt_J3_basis_exp=False,
    opt_J3_basis_coeff=False,
    opt_lambda_basis_exp=False,
    opt_lambda_basis_coeff=False,
    opt_JDTN_u_param=False,
    opt_JDTN_chi_param=False,
    opt_JDTN_f_param=False,
)


@pytest.fixture(scope="module")
def wf_fixture():
    return _build_wf()


_PARAM_ORDER = [
    "j1",
    "j2",
    "j3_matrix",
    "j3_exp",
    "j3_coeff",
    "dtn_alpha",
    "dtn_beta",
    "dtn_gamma",
    "lambda",
    "mo_coeff",
    "gem_exp",
    "gem_coeff",
]

# Named optimization-flag scenarios to display, one row each.
_SCENARIOS = {
    "J1 only": {"opt_J1_param": True},
    "J2 only": {"opt_J2_param": True},
    "J3 matrix only": {"opt_J3_param": True},
    "J3 + basis exp/coeff": {"opt_J3_param": True, "opt_J3_basis_exp": True, "opt_J3_basis_coeff": True},
    "lambda only": {"opt_geminal": True},
    "lambda + basis exp/coeff": {"opt_geminal": True, "opt_lambda_basis_exp": True, "opt_lambda_basis_coeff": True},
    "DTN u only": {"opt_JDTN_u_param": True},
    "DTN chi only": {"opt_JDTN_chi_param": True},
    "DTN f only": {"opt_JDTN_f_param": True},
    "DTN u_chi_f": {"opt_JDTN_u_param": True, "opt_JDTN_chi_param": True, "opt_JDTN_f_param": True},
    "all Jastrow (J1+J2+J3+DTN)": {
        "opt_J1_param": True,
        "opt_J2_param": True,
        "opt_J3_param": True,
        "opt_JDTN_u_param": True,
        "opt_JDTN_chi_param": True,
        "opt_JDTN_f_param": True,
    },
    "everything on": {
        "opt_J1_param": True,
        "opt_J2_param": True,
        "opt_J3_param": True,
        "opt_geminal": True,
        "opt_J3_basis_exp": True,
        "opt_J3_basis_coeff": True,
        "opt_lambda_basis_exp": True,
        "opt_lambda_basis_coeff": True,
        "opt_JDTN_u_param": True,
        "opt_JDTN_chi_param": True,
        "opt_JDTN_f_param": True,
    },
    "nothing on": {},
}


def _j3_ao_only_wf(name="H2_ae_ccpvdz_cart.h5"):
    """J3 (AO GTO) Jastrow over an AO geminal; electrons near nuclei."""
    sd, aos, _mu, _md, gem_mo, _cpd = _load(name)
    gem_ao = Geminal_data.convert_from_MOs_to_AOs(gem_mo)
    j3 = Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos, random_init=True, random_scale=0.01, seed=42)
    jd = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=j3,
    )
    wf = Wavefunction_data(jastrow_data=jd, geminal_data=gem_ao)

    rng = np.random.RandomState(42)
    centers = sd._positions_cart_np

    def pick(n):
        base = centers[rng.randint(0, centers.shape[0], size=n)]
        return jax.numpy.asarray(base + 0.3 * rng.standard_normal((n, 3)))

    return wf, pick(gem_ao.num_electron_up), pick(gem_ao.num_electron_dn)


def _build_sd_wf(trexio_file: str):
    """SD_data wavefunction + sampled coordinates from a TREXIO loader."""
    from jqmc.jastrow_factor import Jastrow_two_body_data

    sd, _, _, _, geminal_data, _ = read_trexio_file(
        trexio_file=os.path.join(TREXIO_DIR, trexio_file),
        store_tuple=True,
    )
    jd = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp"),
        jastrow_three_body_data=None,
    )
    wf = Wavefunction_data(jastrow_data=jd, geminal_data=geminal_data)

    rng = np.random.RandomState(11)
    centers = sd._positions_cart_np

    def pick(n):
        base = centers[rng.randint(0, centers.shape[0], size=n)]
        return jax.numpy.asarray(base + 0.3 * rng.standard_normal((n, 3)))

    return wf, pick(geminal_data.num_electron_up), pick(geminal_data.num_electron_dn)


def _grad_norms_per_channel(g):
    """Return Frobenius norms of mo_coefficients up / dn grads on a Wavefunction_data grad pytree."""
    up = np.asarray(g.geminal_data.orb_data_up_spin.mo_coefficients)
    dn = np.asarray(g.geminal_data.orb_data_dn_spin.mo_coefficients)
    return float(np.linalg.norm(up)), float(np.linalg.norm(dn))


def _build_agpn_wf(trexio_file: str):
    """AGPn wavefunction with explicitly constructed common-rank MOs."""
    from jqmc.jastrow_factor import Jastrow_two_body_data
    from tests.test_jqmc_kernel.optimizer_metadata_helpers import build_explicit_common_rank_agpn_from_sd

    sd, _, _, _, geminal_data, _ = read_trexio_file(
        trexio_file=os.path.join(TREXIO_DIR, trexio_file),
        store_tuple=True,
    )
    agpn = build_explicit_common_rank_agpn_from_sd(geminal_data)
    jd = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=0.5, jastrow_2b_type="exp"),
        jastrow_three_body_data=None,
    )
    wf = Wavefunction_data(jastrow_data=jd, geminal_data=agpn)

    rng = np.random.RandomState(13)
    centers = sd._positions_cart_np

    def pick(n):
        base = centers[rng.randint(0, centers.shape[0], size=n)]
        return jax.numpy.asarray(base + 0.3 * rng.standard_normal((n, 3)))

    return wf, pick(agpn.num_electron_up), pick(agpn.num_electron_dn)


def _lambda_grad_norms(g):
    """Return Frobenius norms of (lambda_diag, lambda_unpaired) on an AGPn grad pytree."""
    diag = np.asarray(g.geminal_data.lambda_diag)
    unp = np.asarray(g.geminal_data.lambda_unpaired)
    return float(np.linalg.norm(diag)), float(np.linalg.norm(unp))


class TestGeneric:
    @pytest.mark.heavy
    @pytest.mark.parametrize("kind", ["agp-all-channels", "sd-open", "agpn-open"])
    def test_parameter_masks_preserve_coordinate_gradients(self, kind):
        """Freezing parameters preserves every nuclear cotangent and physical translation closure.

        The all-channel AGP fixture exercises J1/J2/J3/DTN; the open-shell
        SD/AGPn fixtures exercise whole-instance geminal masking. References
        never apply the parameter mask. Translation invariance supplies an
        independent physical anchor through the electron-coordinate gradient,
        avoiding a finite-difference of an fp32 staircase in Mixed.
        """
        if kind == "agp-all-channels":
            wf, r_up, r_dn = _build_wf()
        elif kind == "sd-open":
            wf, r_up, r_dn = _build_sd_wf("N_ae_ccpvdz_cart.h5")
        else:
            wf, r_up, r_dn = _build_agpn_wf("N_ae_ccpvdz_cart.h5")

        reference_tree = jax.grad(lambda w: evaluate_ln_wavefunction(w, r_up, r_dn))(wf)
        reference = np.asarray(wf.accumulate_position_grad(reference_tree))
        assert reference.ndim == 2 and reference.shape[1] == 3
        assert np.isfinite(reference).all() and np.any(reference != 0)
        electron_up, electron_dn = jax.grad(lambda up, dn: evaluate_ln_wavefunction(wf, up, dn), argnums=(0, 1))(r_up, r_dn)
        electron_reference = -(np.sum(np.asarray(electron_up), axis=0) + np.sum(np.asarray(electron_dn), axis=0))
        atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "jastrow_eval", "det_eval"), "strict")
        npt.assert_allclose(np.sum(reference, axis=0), electron_reference, atol=atol, rtol=rtol)

        def coordinates(tree):
            return {
                jax.tree_util.keystr(path): np.asarray(value)
                for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]
                if jax.tree_util.keystr(path).endswith(".positions")
                and ".structure_data." in jax.tree_util.keystr(path)
                and getattr(value, "ndim", 0) == 2
                and value.shape[1] == 3
            }

        expected_leaves = coordinates(reference_tree)
        assert expected_leaves
        for enabled in (False, True):
            flags = {name: enabled for name in _BASE}
            masked_tree = jax.grad(lambda w: evaluate_ln_wavefunction(w.with_param_grad_mask(**flags), r_up, r_dn))(wf)
            actual_leaves = coordinates(masked_tree)
            assert actual_leaves.keys() == expected_leaves.keys()
            for path, expected in expected_leaves.items():
                npt.assert_allclose(actual_leaves[path], expected, atol=atol, rtol=rtol, err_msg=path)
            actual = np.asarray(wf.accumulate_position_grad(masked_tree))
            assert np.isfinite(actual).all()
            npt.assert_allclose(actual, reference, atol=atol, rtol=rtol)

    @pytest.mark.light
    def test_dtn_only_freezes_everything_else(self, wf_fixture):
        """DTN only freezes everything else."""
        wf, r_up, r_dn = wf_fixture
        fl = _flows(
            wf,
            r_up,
            r_dn,
            **{**_BASE, "opt_JDTN_u_param": True, "opt_JDTN_chi_param": True, "opt_JDTN_f_param": True},
        )
        assert fl["dtn_alpha"] and fl["dtn_beta"] and fl["dtn_gamma"]
        for k in ("j1", "j2", "j3_matrix", "j3_exp", "j3_coeff", "lambda", "mo_coeff", "gem_exp", "gem_coeff"):
            assert not fl[k], f"{k} leaked a gradient under DTN-only opt"

    @pytest.mark.light
    def test_mo_coefficients_never_flow(self, wf_fixture):
        """MO coefficients never flow."""
        wf, r_up, r_dn = wf_fixture
        cases = [
            {"opt_J1_param": True, "opt_J2_param": True, "opt_J3_param": True},
            {"opt_geminal": True},
            {"opt_geminal": True, "opt_lambda_basis_exp": True, "opt_lambda_basis_coeff": True},
            {"opt_J3_param": True, "opt_J3_basis_exp": True, "opt_J3_basis_coeff": True},
            {"opt_JDTN_u_param": True, "opt_JDTN_chi_param": True, "opt_JDTN_f_param": True},
        ]
        for c in cases:
            fl = _flows(wf, r_up, r_dn, **{**_BASE, **c})
            assert not fl["mo_coeff"], f"mo_coefficients leaked a gradient with flags {c}"

    @pytest.mark.light
    def test_lambda_only_freezes_geminal_basis(self, wf_fixture):
        """Lambda only freezes geminal basis."""
        wf, r_up, r_dn = wf_fixture
        fl = _flows(wf, r_up, r_dn, **{**_BASE, "opt_geminal": True})
        assert fl["lambda"]
        assert not fl["mo_coeff"] and not fl["gem_exp"] and not fl["gem_coeff"]
        assert not fl["j1"] and not fl["dtn_alpha"]

    @pytest.mark.light
    def test_lambda_basis_opt_flows_geminal_exp_coeff_but_not_mo(self, wf_fixture):
        """Lambda basis opt flows geminal exp coeff but not MO."""
        wf, r_up, r_dn = wf_fixture
        fl = _flows(
            wf,
            r_up,
            r_dn,
            **{**_BASE, "opt_geminal": True, "opt_lambda_basis_exp": True, "opt_lambda_basis_coeff": True},
        )
        assert fl["gem_exp"] and fl["gem_coeff"]
        assert not fl["mo_coeff"]

    @pytest.mark.light
    def test_j3_basis_opt_flows_j3_exp_coeff_but_freezes_geminal(self, wf_fixture):
        """J3 basis opt flows j3 exp coeff but freezes geminal."""
        wf, r_up, r_dn = wf_fixture
        fl = _flows(
            wf,
            r_up,
            r_dn,
            **{**_BASE, "opt_J3_param": True, "opt_J3_basis_exp": True, "opt_J3_basis_coeff": True},
        )
        assert fl["j3_matrix"] and fl["j3_exp"] and fl["j3_coeff"]
        assert not fl["mo_coeff"] and not fl["gem_exp"] and not fl["gem_coeff"]

    @pytest.mark.light
    def test_each_channel_flows_only_when_enabled(self, wf_fixture):
        """Each channel flows only when enabled."""
        wf, r_up, r_dn = wf_fixture
        fl = _flows(wf, r_up, r_dn, **{**_BASE, "opt_J1_param": True})
        assert fl["j1"] and not fl["j2"] and not fl["j3_matrix"] and not fl["dtn_alpha"]
        fl = _flows(wf, r_up, r_dn, **{**_BASE, "opt_J2_param": True})
        assert fl["j2"] and not fl["j1"] and not fl["j3_matrix"]
        fl = _flows(wf, r_up, r_dn, **{**_BASE, "opt_J3_param": True})
        assert fl["j3_matrix"] and not fl["j3_exp"] and not fl["j3_coeff"]
        fl = _flows(wf, r_up, r_dn, **{**_BASE, "opt_JDTN_u_param": True})
        assert fl["dtn_alpha"] and not fl["dtn_beta"] and not fl["dtn_gamma"]

    # Order of the parameter columns in the printed matrix.

    @pytest.mark.light
    def test_print_grad_matrix(self, wf_fixture, capsys):
        """Print a human-readable flag-pattern x parameter gradient matrix.

        Run with ``pytest -s tests/test_jqmc_kernel/test_param_grad_mask_matrix.py -k print_grad_matrix``
        to eyeball which parameters receive a (non-zero) gradient for each
        optimization-flag scenario. ``Y`` = gradient flows, ``.`` = frozen
        (stop_gradient / not a variational parameter).
        """
        wf, r_up, r_dn = wf_fixture

        rows = []
        for name, scn in _SCENARIOS.items():
            fl = _flows(wf, r_up, r_dn, **{**_BASE, **scn})
            rows.append((name, {k: fl[k] for k in _PARAM_ORDER}))

        # Build a fixed-width table.
        col_w = max(len(p) for p in _PARAM_ORDER)
        name_w = max(len(n) for n in _SCENARIOS)
        header = " " * name_w + " | " + " ".join(f"{p:>{col_w}}" for p in _PARAM_ORDER)

        with capsys.disabled():
            print("\n\n=== parameter-gradient matrix (Y = grad flows, . = frozen) ===")
            print(header)
            print("-" * len(header))
            for name, flags_row in rows:
                cells = " ".join(f"{('Y' if flags_row[p] else '.'):>{col_w}}" for p in _PARAM_ORDER)
                print(f"{name:>{name_w}} | {cells}")
            print()

        # Light invariant: mo_coeff must be frozen in every scenario (never a block).
        for name, flags_row in rows:
            assert not flags_row["mo_coeff"], f"mo_coefficients leaked a gradient in scenario '{name}'"

    # ---------------------------------------------------------------------------
    # Direct stop_gradient checks for the J3 AO-basis mask (moved here from
    # test_ao_basis_optimization.py: these exercise with_param_grad_mask, not the
    # AO-basis optimization machinery itself).
    # ---------------------------------------------------------------------------

    @pytest.mark.light
    def test_with_param_grad_mask_stops_basis_grads(self):
        """opt_J3_basis_exp/coeff=False: J3 exp/coeff grads zero, j_matrix nonzero.

        ``jax.lax.stop_gradient`` only takes effect inside a traced (grad) context,
        so we apply the mask INSIDE the differentiated function -- exactly how the
        production driver wires it -- to verify the mechanism.
        """
        wf, r_up, r_dn = _j3_ao_only_wf()

        def ln_psi(w):
            masked = w.with_param_grad_mask(opt_J3_param=True, opt_J3_basis_exp=False, opt_J3_basis_coeff=False)
            return evaluate_ln_wavefunction(masked, r_up, r_dn)

        g = jax.grad(ln_psi)(wf)
        j3g = g.jastrow_data.jastrow_three_body_data
        # The differentiable DOF lives on the per-shell radial leaf.
        npt.assert_array_equal(np.array(j3g.orb_data.radial.exponents), 0.0)
        npt.assert_array_equal(np.array(j3g.orb_data.radial.coefficients), 0.0)
        assert np.any(np.array(j3g.j_matrix) != 0.0)

    @pytest.mark.light
    def test_with_param_grad_mask_allows_basis_grads(self):
        """opt_J3_basis_exp/coeff=True: J3 exponent gradients flow."""
        wf, r_up, r_dn = _j3_ao_only_wf()

        def ln_psi(w):
            masked = w.with_param_grad_mask(opt_J3_param=True, opt_J3_basis_exp=True, opt_J3_basis_coeff=True)
            return evaluate_ln_wavefunction(masked, r_up, r_dn)

        g = jax.grad(ln_psi)(wf)
        grad_j3_exp = g.jastrow_data.jastrow_three_body_data.orb_data.radial.exponents
        assert np.any(np.array(grad_j3_exp) != 0.0), "Expected nonzero exponent gradients"

    # ---------------------------------------------------------------------------
    # AGPn / SD direct-MO opt: mo_coefficients_up / _dn stop_gradient coverage.
    #
    # The legacy fixture above promotes the loaded geminal to ``AGP_data`` so the
    # ``mo_coefficients`` is never a variational block (lambda_matrix is). This
    # section covers the direct MO-opt path where ``AGPn_data`` /
    # ``SD_data`` expose ``mo_coefficients_up`` (and ``mo_coefficients_dn`` for
    # ``time_reversal=False``) as variational blocks. We verify that
    # ``with_param_grad_mask`` lets gradients flow into the right channels and
    # stops them in the others.
    # ---------------------------------------------------------------------------

    @pytest.mark.light
    def test_grad_mask_closed_shell_SD_opt_geminal_True_flows_up_only(self):
        """Closed-shell singlet (``time_reversal=True``) SD with
        ``opt_geminal=True`` (= ``opt_geminal=True``):

        * The up-channel ``mo_coefficients`` carries a nonzero gradient.
        * The dn-channel grad equals the up-channel grad up to the
          Convention B pairing ``C_dn = conj(C_up)``: it is non-zero but
          derived, not an independent variational direction.
        """
        from jqmc.determinant import SD_data

        wf, r_up, r_dn = _build_sd_wf("H2_ae_ccpvdz_cart.h5")
        assert isinstance(wf.geminal_data, SD_data)
        assert wf.geminal_data.time_reversal is True

        def ln_psi(w):
            masked = w.with_param_grad_mask(
                opt_J1_param=False,
                opt_J2_param=False,
                opt_J3_param=False,
                opt_geminal=True,
            )
            return evaluate_ln_wavefunction(masked, r_up, r_dn)

        g = jax.grad(ln_psi)(wf)
        up_norm, dn_norm = _grad_norms_per_channel(g)
        assert up_norm > 1e-8, f"closed-shell SD opt_geminal=True: mo_coefficients_up grad must flow; got |grad|_F={up_norm}"

    @pytest.mark.light
    def test_grad_mask_closed_shell_SD_opt_geminal_False_stops_both(self):
        """Closed-shell SD with ``opt_geminal=False``: both up and dn
        ``mo_coefficients`` grads must be stopped."""
        from jqmc.determinant import SD_data

        wf, r_up, r_dn = _build_sd_wf("H2_ae_ccpvdz_cart.h5")
        assert isinstance(wf.geminal_data, SD_data)

        def ln_psi(w):
            masked = w.with_param_grad_mask(
                opt_J1_param=False,
                opt_J2_param=False,
                opt_J3_param=False,
                opt_geminal=False,
            )
            return evaluate_ln_wavefunction(masked, r_up, r_dn)

        g = jax.grad(ln_psi)(wf)
        up_norm, dn_norm = _grad_norms_per_channel(g)
        assert up_norm == 0.0, f"opt_geminal=False must stop mo_coefficients_up grad; got |grad|_F={up_norm}"
        assert dn_norm == 0.0, f"opt_geminal=False must stop mo_coefficients_dn grad; got |grad|_F={dn_norm}"

    @pytest.mark.light
    def test_grad_mask_open_shell_SD_opt_geminal_True_flows_up_and_dn_independently(self):
        """Open-shell (``time_reversal=False``) SD with
        ``opt_geminal=True`` MUST let gradients flow INDEPENDENTLY into
        both ``mo_coefficients_up`` AND ``mo_coefficients_dn``.

        Guards against a silently frozen dn channel:
        ``get_variational_blocks`` must expose ``mo_coefficients_dn`` for
        ``time_reversal=False``.
        """
        from jqmc.determinant import SD_data

        wf, r_up, r_dn = _build_sd_wf("N_ae_ccpvdz_cart.h5")
        assert isinstance(wf.geminal_data, SD_data)
        assert wf.geminal_data.time_reversal is False
        assert int(wf.geminal_data.num_electron_up) != int(wf.geminal_data.num_electron_dn)

        def ln_psi(w):
            masked = w.with_param_grad_mask(
                opt_J1_param=False,
                opt_J2_param=False,
                opt_J3_param=False,
                opt_geminal=True,
            )
            return evaluate_ln_wavefunction(masked, r_up, r_dn)

        g = jax.grad(ln_psi)(wf)
        up_norm, dn_norm = _grad_norms_per_channel(g)
        assert up_norm > 1e-8, f"open-shell opt_geminal=True: mo_coefficients_up grad must flow; got {up_norm}"
        assert dn_norm > 1e-8, f"open-shell opt_geminal=True: mo_coefficients_dn grad must flow INDEPENDENTLY; got {dn_norm}"
        # Independence: dn should not be a mere conjugate of up since
        # ``time_reversal=False`` removes Convention B.
        grad_up = np.asarray(g.geminal_data.orb_data_up_spin.mo_coefficients)
        grad_dn = np.asarray(g.geminal_data.orb_data_dn_spin.mo_coefficients)
        # Different shapes are possible (num_mo same, but the dn / up MO
        # vectors are physically distinct under UHF). At minimum,
        # |grad_dn|_F should not equal |grad_up|_F (would happen only if
        # the dn channel were accidentally tied to up).
        if grad_up.shape == grad_dn.shape:
            assert not np.allclose(grad_dn, grad_up, atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_NONVACUITY_DISTINCTION), (
                "open-shell dn grad accidentally equals up grad -- the dn "
                "channel was likely tied to up by Convention B, but "
                "time_reversal=False should make them independent."
            )

    @pytest.mark.light
    def test_grad_mask_open_shell_SD_opt_geminal_False_stops_both(self):
        """Open-shell SD with ``opt_geminal=False``: both up and dn
        ``mo_coefficients`` grads must be stopped (regression guard against
        the dn channel leaking when the mask is off)."""
        from jqmc.determinant import SD_data

        wf, r_up, r_dn = _build_sd_wf("N_ae_ccpvdz_cart.h5")
        assert isinstance(wf.geminal_data, SD_data)
        assert wf.geminal_data.time_reversal is False

        def ln_psi(w):
            masked = w.with_param_grad_mask(
                opt_J1_param=False,
                opt_J2_param=False,
                opt_J3_param=False,
                opt_geminal=False,
            )
            return evaluate_ln_wavefunction(masked, r_up, r_dn)

        g = jax.grad(ln_psi)(wf)
        up_norm, dn_norm = _grad_norms_per_channel(g)
        assert up_norm == 0.0, f"open-shell opt_geminal=False must stop mo_coefficients_up grad; got {up_norm}"
        assert dn_norm == 0.0, f"open-shell opt_geminal=False must stop mo_coefficients_dn grad; got {dn_norm}"

    # ---------------------------------------------------------------------------
    # AGPn-specific blocks: ``lambda_diag`` and ``lambda_unpaired``.
    #
    # AGPn_data adds two block types on top of the SD MO-coeff block:
    #   * ``lambda_diag``: natural-orbital occupations, always real, exposed
    #     when ``opt_geminal=True``.
    #   * ``lambda_unpaired``: open-shell unpaired sub-block, exposed only
    #     when ``time_reversal=False`` AND the system carries unpaired
    #     electrons (n_unpaired > 0).
    #
    # We verify that ``with_param_grad_mask`` stops gradients on both blocks
    # when ``opt_geminal=False`` and lets them flow when ``=True``.
    # ---------------------------------------------------------------------------

    @pytest.mark.light
    def test_grad_mask_AGPn_opt_geminal_True_flows_lambda_diag(self):
        """AGPn (closed-shell) with ``opt_geminal=True``: the
        ``lambda_diag`` block carries a nonzero gradient."""
        from jqmc.determinant import AGPn_data

        wf, r_up, r_dn = _build_agpn_wf("H2_ae_ccpvdz_cart.h5")
        assert isinstance(wf.geminal_data, AGPn_data)

        def ln_psi(w):
            masked = w.with_param_grad_mask(
                opt_J1_param=False,
                opt_J2_param=False,
                opt_J3_param=False,
                opt_geminal=True,
            )
            return evaluate_ln_wavefunction(masked, r_up, r_dn)

        g = jax.grad(ln_psi)(wf)
        diag_norm, _ = _lambda_grad_norms(g)
        assert diag_norm > 1e-8, f"opt_geminal=True must let lambda_diag grad flow; got {diag_norm}"

    @pytest.mark.light
    def test_grad_mask_AGPn_opt_geminal_False_stops_lambda_diag(self):
        """AGPn (closed-shell) with ``opt_geminal=False``: the
        ``lambda_diag`` grad must be stopped."""
        from jqmc.determinant import AGPn_data

        wf, r_up, r_dn = _build_agpn_wf("H2_ae_ccpvdz_cart.h5")
        assert isinstance(wf.geminal_data, AGPn_data)

        def ln_psi(w):
            masked = w.with_param_grad_mask(
                opt_J1_param=False,
                opt_J2_param=False,
                opt_J3_param=False,
                opt_geminal=False,
            )
            return evaluate_ln_wavefunction(masked, r_up, r_dn)

        g = jax.grad(ln_psi)(wf)
        diag_norm, _ = _lambda_grad_norms(g)
        assert diag_norm == 0.0, f"opt_geminal=False must stop lambda_diag grad; got {diag_norm}"

    @pytest.mark.light
    def test_grad_mask_open_shell_AGPn_opt_geminal_True_flows_lambda_unpaired(self):
        """Open-shell AGPn with ``opt_geminal=True``: the
        ``lambda_unpaired`` block must carry an INDEPENDENT gradient
        (variational on AGPn; constant on SD)."""
        from jqmc.determinant import AGPn_data

        wf, r_up, r_dn = _build_agpn_wf("N_ae_ccpvdz_cart.h5")
        assert isinstance(wf.geminal_data, AGPn_data)
        assert wf.geminal_data.time_reversal is False
        assert int(wf.geminal_data.num_electron_up) != int(wf.geminal_data.num_electron_dn)
        assert np.asarray(wf.geminal_data.lambda_unpaired).size > 0

        def ln_psi(w):
            masked = w.with_param_grad_mask(
                opt_J1_param=False,
                opt_J2_param=False,
                opt_J3_param=False,
                opt_geminal=True,
            )
            return evaluate_ln_wavefunction(masked, r_up, r_dn)

        g = jax.grad(ln_psi)(wf)
        diag_norm, unp_norm = _lambda_grad_norms(g)
        assert diag_norm > 1e-8, f"open-shell opt_geminal=True must let lambda_diag grad flow; got {diag_norm}"
        assert unp_norm > 1e-8, f"open-shell opt_geminal=True must let lambda_unpaired grad flow; got {unp_norm}"

    @pytest.mark.light
    def test_grad_mask_open_shell_AGPn_opt_geminal_False_stops_lambda_unpaired(self):
        """Open-shell AGPn with ``opt_geminal=False``: both
        ``lambda_diag`` and ``lambda_unpaired`` grads must be stopped."""
        from jqmc.determinant import AGPn_data

        wf, r_up, r_dn = _build_agpn_wf("N_ae_ccpvdz_cart.h5")
        assert isinstance(wf.geminal_data, AGPn_data)
        assert wf.geminal_data.time_reversal is False

        def ln_psi(w):
            masked = w.with_param_grad_mask(
                opt_J1_param=False,
                opt_J2_param=False,
                opt_J3_param=False,
                opt_geminal=False,
            )
            return evaluate_ln_wavefunction(masked, r_up, r_dn)

        g = jax.grad(ln_psi)(wf)
        diag_norm, unp_norm = _lambda_grad_norms(g)
        assert diag_norm == 0.0, f"opt_geminal=False must stop lambda_diag grad; got {diag_norm}"
        assert unp_norm == 0.0, f"opt_geminal=False must stop lambda_unpaired grad; got {unp_norm}"
