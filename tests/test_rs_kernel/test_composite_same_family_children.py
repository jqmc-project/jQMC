"""Native orbitals refuse AO composites whose children they cannot hold (issue #165).

The native real orbital (``OrbData``, ``jqmc/rs/src/determinant.rs``) keeps ONE GTO/pGTO
block plus at most ONE plane-wave or B-spline block, and the native complex orbital
(``ComplexOrb``, ``jqmc/rs/src/complex_det.rs``) reads exactly children 0 and 1 as one gto
block plus one extra block. Python accepts any combination of leaves in an
``AOs_composite_data`` (``a + b``), for example a basis augmented with a second GTO set
whose MO coefficient columns are zero-padded. A native real parser that keeps only the
LAST child of each family has an AO rank smaller than the Python ``num_ao`` and the MO
coefficient column count; release builds only ``debug_assert`` the matmul dimensions, so:

* the free binding ``compute_grads_and_laplacian_jastrow_three_body`` would return finite,
  silently wrong values for a zero-padded MO (largest error 2.25) and panic for a dense
  MO (``ndarray: index out of bounds``);
* production ``MCMC.run``, ``GFMC_n.run`` and ``GFMC_t.run`` would abort with a Rust
  ``PanicException`` (a ``BaseException``, which ``except Exception`` does not catch);
* a native complex parser that reads a second complex pGTO child as a plane wave fails
  with an unrelated ``AttributeError`` and drops every child past the second.

The contract is a refusal: every such input raises ``NotImplementedError`` naming the
unsupported composite and the XLA backend, before any evaluation, from the free binding and
from every production driver; never a panic or a value. The inputs are built from committed
fixtures through public constructors only.
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

import inspect

import jax
import numpy as np
import pytest

import jqmc_rs
from jqmc._setting import GFMC_MIN_COLLECT_STEPS, MCMC_MIN_STEPS
from jqmc.atomic_orbital import AOs_composite_data, AOs_plane_wave_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.jqmc_mcmc import MCMC
from jqmc.molecular_orbital import MOs_data
from tests.test_rs_kernel._rs_fixtures import draw_cfg, rs_ham

jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = pytest.mark.with_chdir

#: The refusal message: names the unsupported composite and the XLA remedy.
_REFUSAL = r"unsupported AOs_composite_data.*JQMC_FLAG_RUST_ON_CPU=0"


def _ao_leaf(orb):
    return orb.aos_data if hasattr(orb, "mo_coefficients") else orb


def _scaled_gto(aos):
    """A second, distinct molecular GTO set: the same shells with every exponent x1.7."""
    kw = {k: getattr(aos, k) for k in inspect.signature(type(aos).build).parameters if hasattr(aos, k)}
    kw["exponents"] = np.asarray(aos.exponents) * 1.7
    return type(aos).build(**kw)


def _j_matrix(m, seed=3):
    rng = np.random.default_rng(seed)
    jm = rng.normal(size=(m, m + 1)) * 0.05
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)
    return jm


def _j3_jastrow(orb):
    j3 = Jastrow_three_body_data(orb_data=orb, j_matrix=_j_matrix(int(orb.num_mo)))
    j3.sanity_check()
    return Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=j3,
        jastrow_dtn_data=None,
    )


def _mo(aos, coefficients):
    orb = MOs_data(aos_data=aos, num_mo=int(coefficients.shape[0]), mo_coefficients=coefficients)
    orb.sanity_check()
    return orb


def _free_case(case):
    """``(hamiltonian, J3 Jastrow)`` whose J3 orbital is an MO over an unsupported composite."""
    rng = np.random.default_rng(3)
    if case.startswith("mol_"):
        ham = rs_ham("mol_cart_sd_j123")  # N atom, cart cc-pVDZ J3 basis (15 AOs)
        aos_a = _ao_leaf(ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data)
        n_a = int(aos_a.num_ao)
        comp = AOs_composite_data.build((aos_a, _scaled_gto(aos_a)))
        if case == "mol_gto_gto_padded":
            # Zero-padded augmentation: the same wave function as MOs over aos_a alone.
            c = np.concatenate([rng.normal(size=(5, n_a)) * 0.3, np.zeros((5, n_a))], axis=1)
        else:
            c = rng.normal(size=(5, 2 * n_a)) * 0.3
        return ham, _j3_jastrow(_mo(comp, c))
    ham = rs_ham("pbc_gamma_sphepgto_j123")  # Li Gamma, sphe pGTO J3 basis
    aos_a = _ao_leaf(ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data)
    n_a = int(aos_a.num_ao)
    if case == "pbc_pgto_pgto_dense":
        comp = AOs_composite_data.build((aos_a, aos_a))
    else:  # pbc_pgto_pw_pw: one pGTO child plus TWO real plane-wave children
        st = ham.structure_data
        rec = 2.0 * np.pi * np.linalg.inv(np.asarray(st.cell)).T
        pw_0 = AOs_plane_wave_data.build(st, np.zeros((1, 3)), np.zeros(1))
        pw_g = AOs_plane_wave_data.build(st, rec[:1], np.zeros(1))
        comp = AOs_composite_data.build((aos_a, pw_0, pw_g))
    n = int(comp.num_ao)
    assert n > n_a
    return ham, _j3_jastrow(_mo(comp, rng.normal(size=(n, n)) * 0.3))


@pytest.mark.light
@pytest.mark.parametrize("case", ["mol_gto_gto_padded", "mol_gto_gto_dense", "pbc_pgto_pgto_dense", "pbc_pgto_pw_pw"])
def test_free_j3_grads_laplacian_refuses_same_family_composite(case):
    """The free native J3 grads/Laplacian binding raises NotImplementedError, never a value or a panic.

    Cases: MOs over (cart GTO, cart GTO with exponents x1.7) with zero-padded and with dense
    coefficients (N atom), MOs over (sphe pGTO, the same pGTO) with dense coefficients (Li
    Gamma), and MOs over (sphe pGTO, constant PW, cos PW) (Li Gamma). A regression
    returns finite wrong values in the padded case and panics in the dense cases.
    """
    ham, jas = _free_case(case)
    r_up, r_dn = draw_cfg(ham, seed=0)
    r_up = np.asarray(r_up, dtype=np.float64)
    r_dn = np.asarray(r_dn, dtype=np.float64)
    with pytest.raises(NotImplementedError, match=_REFUSAL):
        jqmc_rs.compute_grads_and_laplacian_jastrow_three_body(jas, r_up, r_dn)


def _pad_with(mo, *extra):
    """``MOs_data(mo.aos_data + extra..., [C | 0 ...])``: the identical wave function on a composite."""
    comp = AOs_composite_data.build((mo.aos_data, *extra))
    c = np.asarray(mo.mo_coefficients)
    pads = [np.zeros((c.shape[0], int(e.num_ao)), dtype=c.dtype) for e in extra]
    return _mo(comp, np.concatenate([c, *pads], axis=1))


def _driver_ham(case):
    """Production Hamiltonian whose determinant (or J3) orbital is a zero-padded composite."""
    if case.startswith("twist_"):
        ham = rs_ham("twist_sphepgto_jnone")  # Li k=1/4: complex pGTO MOs
        gd = ham.wavefunction_data.geminal_data
        up, dn = gd.orb_data_up_spin, gd.orb_data_dn_spin
        copies = 1 if case == "twist_det_pgto_pgto" else 2  # twist_det_pgto_x3: three children
        gd = gd.replace(
            orb_data_up_spin=_pad_with(up, *[up.aos_data] * copies),
            orb_data_dn_spin=_pad_with(dn, *[dn.aos_data] * copies),
        )
        wf = ham.wavefunction_data.replace(geminal_data=gd)
    else:
        ham = rs_ham("mol_cart_sd_j123")
        wf = ham.wavefunction_data
        if case == "mol_det_gto_gto":
            gd = wf.geminal_data
            up, dn = gd.orb_data_up_spin, gd.orb_data_dn_spin
            gd = gd.replace(
                orb_data_up_spin=_pad_with(up, _scaled_gto(up.aos_data)),
                orb_data_dn_spin=_pad_with(dn, _scaled_gto(dn.aos_data)),
            )
            wf = wf.replace(geminal_data=gd)
        else:  # mol_j3_gto_gto: identity-MO J3 over the J3 basis, zero-padded
            aos = _ao_leaf(wf.jastrow_data.jastrow_three_body_data.orb_data)
            n = int(aos.num_ao)
            j3 = Jastrow_three_body_data(
                orb_data=_pad_with(_mo(aos, np.eye(n)), _scaled_gto(aos)), j_matrix=_j_matrix(n, seed=5)
            )
            j3.sanity_check()
            wf = wf.replace(jastrow_data=wf.jastrow_data.replace(jastrow_three_body_data=j3))
    ham = ham.replace(wavefunction_data=wf)
    ham.sanity_check()
    return ham


@pytest.mark.light
@pytest.mark.parametrize("driver", ["MCMC", "GFMC_n", "GFMC_t"])
@pytest.mark.parametrize("case", ["mol_det_gto_gto", "mol_j3_gto_gto", "twist_det_pgto_pgto", "twist_det_pgto_x3"])
def test_native_drivers_refuse_same_family_composite(case, driver, monkeypatch, tmp_path):
    """Production MCMC.run / GFMC_n.run / GFMC_t.run on the native route raise NotImplementedError.

    The determinant MOs (N atom, real; Li k=1/4, complex) or the identity-MO J3 orbital (N
    atom) are rewritten over a composite of their AO basis and a second GTO set of the same
    family (for the complex case also a third copy) with zero coefficient columns. Without the
    refusal the real cases abort with a Rust PanicException and the complex cases with an
    unrelated AttributeError.
    """
    monkeypatch.chdir(tmp_path)  # the drivers write external_control_*.toml into the cwd
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
    ham = _driver_ham(case)
    common = dict(hamiltonian_data=ham, num_walkers=2, mcmc_seed=1, comput_position_deriv=False)
    with pytest.raises(NotImplementedError, match=_REFUSAL):
        if driver == "MCMC":
            sampler = MCMC(**common, Dt=2.0, epsilon_AS=0.0)
            sampler.run(num_mcmc_steps=MCMC_MIN_STEPS)
        elif driver == "GFMC_n":
            sampler = GFMC_n(
                **common,
                alat=0.3,
                non_local_move="tmove",
                num_mcmc_per_measurement=1,
                E_scf=-1.0,
                num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
            )
            sampler.run(num_mcmc_steps=GFMC_MIN_COLLECT_STEPS + 1)
        else:
            sampler = GFMC_t(
                **common, alat=0.3, tau=0.05, non_local_move="tmove", num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS
            )
            sampler.run(num_mcmc_steps=GFMC_MIN_COLLECT_STEPS + 1)
