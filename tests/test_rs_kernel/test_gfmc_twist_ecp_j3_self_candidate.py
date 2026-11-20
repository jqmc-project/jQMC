"""XLA GFMC on a multi-electron complex-twist ECP wavefunction with an active J3 (issue #652).

Fixture: the committed QE ``O-lsda_k025`` save (4 up / 2 dn, ccECP, k = (1/4, 1/4, 1/4))
converted by the production converter to a pGTO leaf, plus the active complex-pGTO
J3 of the generated-matrix recipe ``_make_j3``. Every other complex-twist ECP GFMC
fixture holds one electron per spin, where the defect below cannot occur.

Defect caught: inside one projection interval the ECP quadrature rotation is held
fixed, so after a t-move the moved electron sits exactly on a point of its own ECP
candidate mesh. That candidate configuration moves no electron. The cached-J3 ratio
branches identified the moved electron by ``argmax`` over the displacement mask
(index 0 for a no-move candidate) while reading the cached candidate AOs of the
candidate's own electron, so the J3 ratio of that candidate was wrong for every
electron index >= 1 instead of exactly 1. The XLA streaming GFMC_t then stored a
wrong local energy and weight (+8.676 Ha at seed 3, one walker, one step), and the
GFMC_n weight was wrong; native Rust and every cache-free XLA path were right.

Independent references, same seed and same walk:

* native Rust GFMC (``JQMC_FLAG_RUST_ON_CPU`` unset): stored ``e_L`` / ``w_L`` and the
  final walker and RNG keys, at the existing periodic route-parity envelope;
* GFMC_t: Re of the cache-free fixed-configuration ``compute_local_energy`` at the final
  walker (identity mesh, so the continuum measurement e_L is exactly this value).

The rank-one (DLA t-move) consumer of the same cache is checked at kernel level in
``tests/test_jqmc_kernel/test_ecp_self_candidate_j3_ratio.py``.
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

import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._setting import GFMC_MIN_COLLECT_STEPS
from jqmc.hamiltonians import compute_local_energy
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from tests._det_gradlap_chain_fixtures import _o_lsda_twist_ham
from tests.test_jqmc_kernel._generated_matrix_base import _make_j3
from tests.test_rs_kernel._route_spy import assert_native_ran, install_ctx_spy
from tests.test_rs_kernel.test_gfmc_chain_modes import _assert_value_parity

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

_ALAT = 0.30
_TAU = 0.05
_NSUB = 40

_HAM_CACHE: dict[str, object] = {}


def _o_lsda_twist_j3_ham(tmp_dir: str):
    """O-lsda_k025 (pGTO, 4 up / 2 dn, ccECP, J1/J2 from the converter) with an active J3."""
    if "ham" not in _HAM_CACHE:
        ham = _o_lsda_twist_ham(tmp_dir)
        leaf = ham.wavefunction_data.geminal_data.orb_data_up_spin
        j3 = _make_j3(getattr(leaf, "aos_data", leaf))
        jastrow = ham.wavefunction_data.jastrow_data.replace(jastrow_three_body_data=j3)
        _HAM_CACHE["ham"] = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jastrow))
    return _HAM_CACHE["ham"]


@pytest.fixture(scope="module")
def o_lsda_j3_ham(tmp_path_factory):
    return _o_lsda_twist_j3_ham(str(tmp_path_factory.mktemp("o_lsda_k025_j3")))


def _run(driver, ham, seed, non_local_move, backend):
    if backend == "xla":
        os.environ["JQMC_FLAG_RUST_ON_CPU"] = "0"
    else:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)
    try:
        common = dict(
            hamiltonian_data=ham,
            num_walkers=1,
            num_gfmc_collect_steps=GFMC_MIN_COLLECT_STEPS,
            mcmc_seed=seed,
            alat=_ALAT,
            non_local_move=non_local_move,
            random_discretized_mesh=False,
            comput_position_deriv=False,
            comput_stress=False,
        )
        if driver == "n":
            g = GFMC_n(**common, num_mcmc_per_measurement=_NSUB, E_scf=-1.0)
        else:
            g = GFMC_t(**common, tau=_TAU)
        g.run(num_mcmc_steps=1)
        return g
    finally:
        os.environ.pop("JQMC_FLAG_RUST_ON_CPU", None)


@pytest.mark.parametrize(
    ("driver", "non_local_move", "seed"),
    [("t", "tmove", 3), ("n", "tmove", 2)],
)
def test_xla_gfmc_twist_ecp_j3_multi_electron_matches_native(o_lsda_j3_ham, driver, non_local_move, seed, monkeypatch):
    """XLA streaming GFMC stores the native e_L / w_L on the 4+2 twist ccECP J3 walk."""
    label = f"GFMC_{driver}/{non_local_move}/seed{seed}"
    xla = _run(driver, o_lsda_j3_ham, seed, non_local_move, "xla")
    spy = install_ctx_spy(monkeypatch)
    native = _run(driver, o_lsda_j3_ham, seed, non_local_move, "native")
    assert_native_ran(spy, f"run_gfmc_projection_{driver}_complex_batch", at_least=1, context=label)

    # Same walk on both backends (fixed identity mesh): byte-identical walker and keys.
    np.testing.assert_array_equal(native._latest_r_up_carts, xla._latest_r_up_carts)
    np.testing.assert_array_equal(native._latest_r_dn_carts, xla._latest_r_dn_carts)
    np.testing.assert_array_equal(native._jax_PRNG_key_list, xla._jax_PRNG_key_list)
    for name in ("_stored_w_L", "_stored_e_L"):
        values = np.asarray(getattr(xla, name))
        assert values.size > 0 and np.all(np.isfinite(values)), f"{label}: {name} empty or non-finite"
        _assert_value_parity(getattr(native, name), values, f"{label} {name}", periodic=True)

    if driver == "t":
        # Cache-free reference: Re of the stored GFMC_t measurement e_L is the continuum
        # local energy of the final walker (t-move localized ECP kept as-is). Only Re is
        # compared: the fixed-phase ECP candidates are real-projected for the LRDMC sign
        # split, so Im(e_L) differs from the fixed-configuration value by construction.
        r_up = np.asarray(xla._latest_r_up_carts)[0]
        r_dn = np.asarray(xla._latest_r_dn_carts)[0]
        e_ref = np.real(np.asarray(compute_local_energy(o_lsda_j3_ham, r_up, r_dn, jnp.eye(3)))).reshape(1, 1)
        _assert_value_parity(
            e_ref, np.real(np.asarray(xla._stored_e_L)), f"{label} Re stored e_L vs cache-free e_L", periodic=True
        )
