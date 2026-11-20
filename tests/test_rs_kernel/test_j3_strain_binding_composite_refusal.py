"""The free J3 strain binding refuses same-family composite J3 orbitals (issue #176).

``jqmc_rs._grad_lnJ3_pbc_strain`` builds its J3 pieces through ``pgto_j3_pieces_composite``
(``jqmc/rs/src/lib.rs``) and never parses an ``OrbData``, so the issue #165 refusal in the
native orbital parser does not reach it. ``pgto_j3_pieces_composite`` kept only the LAST
GTO/pGTO child and the LAST plane-wave child of an ``AOs_composite_data``, so its AO rank
was smaller than the ``j_matrix`` rank and the binding aborted with a Rust
``PanicException`` (``ndarray: inputs 26 x 52 and 26 x 1 are not compatible for matrix
multiplication``), a ``BaseException`` that ``except Exception`` does not catch.

The contract is the #165 refusal: such a J3 orbital raises ``NotImplementedError`` naming
the unsupported composite and the XLA backend, never a panic or a value. The plain
(non-composite) pGTO J3 orbital keeps returning a finite 3x3 strain. The inputs are built
from committed fixtures through public constructors only.
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

import jax
import numpy as np
import pytest

import jqmc_rs
from jqmc.atomic_orbital import AOs_composite_data, AOs_plane_wave_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_three_body_data
from jqmc.molecular_orbital import MOs_data
from tests.test_rs_kernel._rs_fixtures import draw_cfg, rs_ham

jax.config.update("jax_enable_x64", True)

# The test reads only committed fixtures through absolute paths.
pytestmark = pytest.mark.with_chdir

#: The refusal message: names the unsupported composite and the XLA remedy.
_REFUSAL = r"unsupported AOs_composite_data.*JQMC_FLAG_RUST_ON_CPU=0"


def _strain_inputs(case):
    """``(J3 Jastrow, r_up, r_dn)`` for Li Gamma with an identity-MO J3 over ``case``'s AO basis."""
    ham = rs_ham("pbc_gamma_sphepgto_j123")  # Li Gamma, sphe pGTO J3 basis
    orb = ham.wavefunction_data.jastrow_data.jastrow_three_body_data.orb_data
    aos = orb.aos_data if hasattr(orb, "mo_coefficients") else orb
    st = ham.structure_data
    rec = 2.0 * np.pi * np.linalg.inv(np.asarray(st.cell)).T
    if case == "plain":
        basis = aos
    elif case == "pgto_pgto":  # the issue's reproduction: (sphe pGTO, the same pGTO)
        basis = AOs_composite_data.build((aos, aos))
    else:  # pgto_pw_pw: one pGTO child plus TWO real plane-wave children
        pw_0 = AOs_plane_wave_data.build(st, np.zeros((1, 3)), np.zeros(1))
        pw_g = AOs_plane_wave_data.build(st, rec[:1], np.zeros(1))
        basis = AOs_composite_data.build((aos, pw_0, pw_g))
    n = int(basis.num_ao)
    rng = np.random.default_rng(3)
    jm = rng.normal(size=(n, n + 1)) * 0.05
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)
    j3 = Jastrow_three_body_data(orb_data=MOs_data(aos_data=basis, num_mo=n, mo_coefficients=np.eye(n)), j_matrix=jm)
    j3.sanity_check()
    jas = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=j3,
        jastrow_dtn_data=None,
    )
    r_up, r_dn = draw_cfg(ham, seed=0)
    return jas, np.asarray(r_up, dtype=np.float64), np.asarray(r_dn, dtype=np.float64)


@pytest.mark.light
@pytest.mark.parametrize("case", ["pgto_pgto", "pgto_pw_pw"])
def test_free_j3_strain_binding_refuses_same_family_composite(case):
    """``_grad_lnJ3_pbc_strain`` raises NotImplementedError for a same-family composite, never a panic.

    Cases (Li Gamma, identity-MO J3): (sphe pGTO, the same pGTO), the issue's reproduction;
    and (sphe pGTO, constant PW, cos PW), two plane-wave children of which a per-family parse
    keeps only the last. A plain pGTO J3 orbital still returns a
    finite 3x3 strain, so the refusal is specific to the unsupported composite.
    """
    jas, r_up, r_dn = _strain_inputs(case)
    with pytest.raises(NotImplementedError, match=_REFUSAL):
        jqmc_rs._grad_lnJ3_pbc_strain(jas, r_up, r_dn)
    jas, r_up, r_dn = _strain_inputs("plain")
    out = np.asarray(jqmc_rs._grad_lnJ3_pbc_strain(jas, r_up, r_dn))
    assert out.shape == (3, 3)
    assert np.all(np.isfinite(out))
