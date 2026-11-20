"""Gates for the GFMC_t.run() compile tax.

A ``GFMC_t.run()`` call must not pay XLA compilations that the projection
work does not require (each costs ~45-50 s for 128 electrons and NW=16 on a
GPU). The three classes the gates name F1-F3:

  F1  the NON-streaming while-loop driver ``_run_projection_loop`` is never
      compiled (nor its single-projection warmup executed): ``use_streaming``
      is hard-enabled and only the streaming branch is dispatched;
  F2  the streaming AOT warmup and the branching steps share one executable,
      so the float64 ``projection_counter_list`` of the branching steps does
      not recompile it;
  F3  a second ``run()`` call compiles nothing: the jitted drivers are not
      recreated per call.

Oracle: XLA compile events counted through the ``jax._src.dispatch`` logger
(``jax_log_compiles``), filtered to the driver names. Counts are
platform-independent, so the gate runs on CPU with a small fixture; the
wall-clock magnitude of the tax is not asserted here.
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

import logging
import os

import jax
import pytest

from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_two_body_data,
)
from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

pytestmark = [pytest.mark.with_chdir]


class _CompileEventCounter(logging.Handler):
    """Collects 'Finished XLA compilation of jit(<name>)' messages."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.events: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - trivial
        msg = record.getMessage()
        if "Finished XLA compilation of" in msg:
            self.events.append(msg)


def _count(events: list[str], jit_name: str) -> int:
    # "jit(_run_projection_loop)" must not match the "..._streaming" variant:
    # the closing parenthesis makes the name exact.
    return sum(1 for m in events if f"jit({jit_name})" in m)


def _h2_ae_hamiltonian() -> Hamiltonian_data:
    structure_data, _, _mos, _, geminal_mo_data, coulomb_potential_data = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvdz_cart.h5"),
        store_tuple=True,
    )
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure_data
        ),
    )
    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    hamiltonian_data.sanity_check()
    return hamiltonian_data


@pytest.mark.light
def test_gfmc_t_run_compiles_each_projection_driver_at_most_once():
    """The projection while-loop drivers compile EXACTLY once per process:

    * the unused non-streaming driver never compiles;
    * the streaming driver compiles once in the first ``run()`` -- i.e. the
      AOT warmup signature matches the branching-step call;
    * a second ``run()`` call compiles neither driver.
    """
    hamiltonian_data = _h2_ae_hamiltonian()

    counter = _CompileEventCounter()
    dispatch_logger = logging.getLogger("jax._src.dispatch")
    jax.config.update("jax_log_compiles", True)
    dispatch_logger.addHandler(counter)
    try:
        gfmc = GFMC_t(
            hamiltonian_data=hamiltonian_data,
            num_walkers=2,
            tau=0.05,
            alat=0.30,
            num_gfmc_collect_steps=2,
            mcmc_seed=3446,
            random_discretized_mesh=True,
        )
        counter.events.clear()
        gfmc.run(num_mcmc_steps=1)
        first_run = list(counter.events)
        counter.events.clear()
        gfmc.run(num_mcmc_steps=1)
        second_run = list(counter.events)
    finally:
        dispatch_logger.removeHandler(counter)
        jax.config.update("jax_log_compiles", False)

    # F1: the dead non-streaming driver must never be compiled.
    n_dead = _count(first_run, "_run_projection_loop") + _count(second_run, "_run_projection_loop")
    assert n_dead == 0, f"non-streaming driver compiled {n_dead}x although streaming is the only dispatched path"
    # F2: one compile serves both the AOT warmup and the branching steps.
    n_first = _count(first_run, "_run_projection_loop_streaming")
    assert n_first == 1, f"streaming driver compiled {n_first}x in the first run() (AOT signature must match the real call)"
    # F3, blanket form: a second run() must not compile ANY module (the
    # driver reuse plus the process-wide caches of the module-level jits
    # leave nothing to compile on this no-force configuration).
    assert not second_run, "second GFMC_t.run() compiled modules:\n" + "\n".join(second_run)
    jax.clear_caches()


@pytest.mark.light
def test_gfmc_t_seed_inverse_reuses_executable_with_current_inputs():
    """Walker positions and geminal parameters are dynamic inputs to the seed executable."""
    import jax.numpy as jnp
    import numpy as np

    from jqmc._precision import get_tolerance
    from jqmc.determinant import compute_geminal_all_elements
    from jqmc.jqmc_gfmc import _seed_gfmc_t_geminal_inverse

    geminal = _h2_ae_hamiltonian().wavefunction_data.geminal_data
    mos_up, mos_dn = geminal.orb_data_up_spin, geminal.orb_data_dn_spin
    updated_geminal = geminal.replace(
        orb_data_up_spin=mos_up.replace(mo_coefficients=1.25 * mos_up.mo_coefficients),
        orb_data_dn_spin=mos_dn.replace(mo_coefficients=1.25 * mos_dn.mo_coefficients),
    )
    r_up = jnp.asarray([[[0.25, -0.2, 0.2]], [[0.6, 0.15, -0.3]]], dtype=jnp.float64)
    r_dn = -r_up
    moved_r_up = r_up + 0.125
    inputs = [(geminal, r_up, r_dn), (geminal, moved_r_up, r_dn), (updated_geminal, moved_r_up, r_dn)]
    expected = [
        np.linalg.inv(np.stack([compute_geminal_all_elements(g, ru, rd) for ru, rd in zip(up, dn, strict=True)]))
        for g, up, dn in inputs
    ]

    counter = _CompileEventCounter()
    dispatch_logger = logging.getLogger("jax._src.dispatch")
    dispatch_logger.addHandler(counter)
    try:
        with jax.log_compiles(True):
            first = _seed_gfmc_t_geminal_inverse(*inputs[0]).block_until_ready()
            counter.events.clear()
            later = [_seed_gfmc_t_geminal_inverse(*args).block_until_ready() for args in inputs[1:]]
    finally:
        dispatch_logger.removeHandler(counter)

    assert not counter.events, "changed seed inputs compiled modules:\n" + "\n".join(counter.events)
    atol, rtol = get_tolerance("det_eval", "strict")
    np.testing.assert_allclose(np.stack([first, *later]), np.stack(expected), atol=atol, rtol=rtol)


@pytest.mark.light
def test_gfmc_n_second_run_compiles_nothing():
    """GFMC_n twin of the second-run gate, strengthened to a blanket assertion: a
    second ``run()`` call on the same instance must not compile ANY module.

    GFMC_n's jitted callables (projection, V-elements, force / stress / SWCT
    kernels) must not be rebuilt per ``run()`` call, or every call recompiles
    the whole set from scratch.
    """
    hamiltonian_data = _h2_ae_hamiltonian()

    counter = _CompileEventCounter()
    dispatch_logger = logging.getLogger("jax._src.dispatch")
    jax.config.update("jax_log_compiles", True)
    dispatch_logger.addHandler(counter)
    try:
        gfmc = GFMC_n(
            hamiltonian_data=hamiltonian_data,
            num_walkers=2,
            num_mcmc_per_measurement=5,
            num_gfmc_collect_steps=2,
            mcmc_seed=3446,
            E_scf=-1.0,
            alat=0.30,
            random_discretized_mesh=True,
        )
        gfmc.run(num_mcmc_steps=1)
        counter.events.clear()
        gfmc.run(num_mcmc_steps=1)
        second_run = list(counter.events)
    finally:
        dispatch_logger.removeHandler(counter)
        jax.config.update("jax_log_compiles", False)

    assert not second_run, "second GFMC_n.run() compiled modules:\n" + "\n".join(second_run)
    jax.clear_caches()


def _h4_pbc_hamiltonian() -> Hamiltonian_data:
    structure_data, _, _mos, _, geminal_mo_data, coulomb_potential_data = read_trexio_file(
        trexio_file=os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H4_pbc_ae_cart.h5"),
        store_tuple=True,
    )
    jastrow_data = Jastrow_data(
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0, jastrow_2b_type="pade", structure_data=structure_data
        ),
    )
    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    hamiltonian_data = Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )
    hamiltonian_data.sanity_check()
    return hamiltonian_data


@pytest.mark.light
def test_gfmc_t_comput_stress_false_skips_stress_and_keeps_force():
    """Stress selection on PBC is independent from position derivatives.

    With the stress bundled into ``comput_position_deriv=True``, the fused
    both-graph runs the mesh-strain S' stream (60-80% of the analytic-force
    flops) every step even when sigma is not wanted.
    ``comput_stress=False`` must (a) keep the HF force numerically equal to
    the bundled run (same subgraph, fp-benign), (b) leave the stored stress
    arrays at zero, and (c) compile strictly FEWER XLA modules (the stress
    kernels and their warmups never exist).
    """
    import numpy as np

    from jqmc._precision import get_tolerance

    hamiltonian_data = _h4_pbc_hamiltonian()

    def _run(comput_stress):
        # Each side must start from the same cold-cache condition. Otherwise
        # earlier tests can suppress only the first side's compile events.
        jax.clear_caches()
        counter = _CompileEventCounter()
        dispatch_logger = logging.getLogger("jax._src.dispatch")
        jax.config.update("jax_log_compiles", True)
        dispatch_logger.addHandler(counter)
        try:
            gfmc = GFMC_t(
                hamiltonian_data=hamiltonian_data,
                num_walkers=2,
                tau=0.05,
                alat=0.30,
                num_gfmc_collect_steps=2,
                mcmc_seed=3446,
                random_discretized_mesh=True,
                comput_position_deriv=True,
                comput_stress=comput_stress,
            )
            gfmc.run(num_mcmc_steps=1)
        finally:
            dispatch_logger.removeHandler(counter)
            jax.config.update("jax_log_compiles", False)
        force = np.array(gfmc._stored_force_HF)
        stress = np.array(gfmc._stored_stress_HF)
        n_compiles = len(counter.events)
        jax.clear_caches()
        return force, stress, n_compiles

    force_full, stress_full, n_full = _run(True)
    force_off, stress_off, n_off = _run(False)
    # (a) force parity (same trajectory: same seed and stress does not
    # touch the projection RNG; the force subgraph is identical).
    atol, rtol = get_tolerance("wf_kinetic", "strict")
    np.testing.assert_allclose(force_off, force_full, atol=atol, rtol=rtol)
    # (b) bundled run produced a nonzero stress; opt-out leaves zeros.
    assert np.any(stress_full != 0.0), "bundled run must record a stress (fixture sanity)"
    assert not np.any(stress_off != 0.0), "comput_stress=False must never write a stress"
    # (c) strictly fewer compiled modules without the stress kernels.
    assert n_off < n_full, f"stress opt-out compiled {n_off} modules vs {n_full} with stress -- nothing was skipped"
    jax.clear_caches()
