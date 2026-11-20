"""GFMC twist-averaged driver tests.

``run_twist_average_gfmc`` mirrors the VMC twist driver
(``run_twist_average_from_hamiltonians``) but drives a per-twist GFMC
chain instead of a VMC one. The two-level estimator math, comm split,
per-``k`` binning, and BZ-mean combine are identical (the aggregator
helpers are reused verbatim); only the per-twist sampler swaps from
:class:`jqmc.jqmc_mcmc.MCMC` to :class:`jqmc.jqmc_gfmc.GFMC_t` /
:class:`jqmc.jqmc_gfmc.GFMC_n`.

These tests pin:

* ``N_twist == 1`` byte-identity to a plain ``GFMC_*`` run on the same
  ``world_comm`` (the GFMC analogue of the MCMC N_twist == 1 identity);
* the GFMC-specific contract: branching / reconfiguration
  stays within twist (``self._mpi_comm`` == intra), ``W_k`` is NEVER
  folded into the GFMC walker weight (it enters only at the Level-2
  combine);
* ``gfmc_kind={'tau', 'n'}`` dispatch; unsupported kinds raise.
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

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._setting import ATOL_BIT_EXACT
from jqmc._setting import ATOL_PYTHON_FP64_BZ_MEAN
from jqmc._setting import RTOL_PYTHON_FP64_ZERO
from jqmc._twist_average import run_twist_average_gfmc
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data

_H_FIXTURE = Path(__file__).parent.parent / "trexio_example_files" / "H_ae_ccpvdz_cart.h5"
_h_available = _H_FIXTURE.is_file()

requires_h_fixture = pytest.mark.skipif(not _h_available, reason="H_ae_ccpvdz_cart.h5 TREXIO fixture not present")


@pytest.fixture(scope="module")
def h_atom_hamiltonian():
    """Tiny molecular fixture (H atom, ccpvdz) for the GFMC twist
    driver smoke tests. Single-twist molecular case; the periodic
    Bloch fixtures live in their own files."""
    out = read_trexio_file(str(_H_FIXTURE), store_tuple=True)
    sd, _, _, _, gem, cpd = out
    ham = Hamiltonian_data(
        structure_data=sd,
        coulomb_potential_data=cpd,
        wavefunction_data=Wavefunction_data(
            jastrow_data=Jastrow_data(),
            geminal_data=gem,
        ),
    )
    ham.sanity_check()
    return ham


class TestGFMCTwistAverageByteIdentity:
    """N_twist == 1 byte-identity to a direct GFMC_* run."""

    # ---------- N_twist == 1 byte-identity ----------
    @pytest.mark.heavy
    @requires_h_fixture
    def test_run_twist_average_gfmc_tau_n_twist_1_byte_identity(self, h_atom_hamiltonian):
        """At ``N_twist == 1`` the GFMC twist driver runs a single GFMC_t
        chain and reproduces the direct ``GFMC_t.run + get_E`` result
        byte-for-byte on a fixed seed -- the orchestrator must not
        perturb the chain (no extra reductions, no W_k weighting since W=1)."""
        from jqmc.jqmc_gfmc import GFMC_t

        common = dict(
            num_walkers=2,
            mcmc_seed=12345,
            tau=0.1,
            alat=0.1,
        )

        # Reference: plain GFMC_t direct call.
        ref_gfmc = GFMC_t(hamiltonian_data=h_atom_hamiltonian, **common)
        ref_gfmc.run(num_mcmc_steps=20)
        E_ref, sigma_ref, _, _ = ref_gfmc.get_E(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=3)

        # Orchestrator at N_twist = 1 must reproduce E_ref / sigma_ref.
        result = run_twist_average_gfmc(
            hamiltonian_list=[h_atom_hamiltonian],
            weights=[1.0],
            gfmc_kind="tau",
            mcmc_seed=common["mcmc_seed"],
            num_mcmc_steps=20,
            num_walkers=common["num_walkers"],
            num_mcmc_warmup_steps=5,
            num_mcmc_bin_blocks=3,
            gfmc_kwargs={"tau": common["tau"], "alat": common["alat"]},
        )
        assert result.energy.E_TA == pytest.approx(float(E_ref), abs=ATOL_BIT_EXACT)
        assert result.energy.E_TA_std == pytest.approx(float(sigma_ref), abs=ATOL_BIT_EXACT)
        assert result.energy.s_twist == pytest.approx(0.0, abs=ATOL_BIT_EXACT)

    # ---------- 14.3.8 _debug variant byte-identity ----------
    @pytest.mark.heavy
    @requires_h_fixture
    def test_run_twist_average_gfmc_tau_debug_variant_byte_identity(self, h_atom_hamiltonian):
        """``gfmc_kind='tau_debug'`` dispatches to ``_GFMC_t_debug``. At
        N_twist=1 the orchestrator reproduces a direct ``_GFMC_t_debug.run +
        get_E`` call byte-for-byte. Pins the four-variant
        coverage contract (the production class + its debug twin share the
        twist-driver entry point)."""
        from jqmc.jqmc_gfmc import _GFMC_t_debug

        common = dict(num_walkers=2, mcmc_seed=12345, tau=0.1, alat=0.1)
        ref = _GFMC_t_debug(hamiltonian_data=h_atom_hamiltonian, **common)
        ref.run(num_mcmc_steps=20)
        E_ref, sigma_ref, _, _ = ref.get_E(num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=3)

        result = run_twist_average_gfmc(
            hamiltonian_list=[h_atom_hamiltonian],
            weights=[1.0],
            gfmc_kind="tau_debug",
            mcmc_seed=common["mcmc_seed"],
            num_mcmc_steps=20,
            num_walkers=common["num_walkers"],
            num_mcmc_warmup_steps=5,
            num_mcmc_bin_blocks=3,
            gfmc_kwargs={"tau": common["tau"], "alat": common["alat"]},
        )
        assert result.energy.E_TA == pytest.approx(float(E_ref), abs=ATOL_BIT_EXACT)
        assert result.energy.E_TA_std == pytest.approx(float(sigma_ref), abs=ATOL_BIT_EXACT)


class TestGFMCTwistAverageDispatches:
    """``gfmc_kind`` dispatch smoke tests for the ``n`` and ``n_debug`` variants."""

    # ---------- gfmc_kind='n' dispatch ----------
    @pytest.mark.heavy
    @requires_h_fixture
    def test_run_twist_average_gfmc_n_kind_smoke(self, h_atom_hamiltonian):
        """``gfmc_kind='n'`` dispatches to GFMC_n. Smoke check that the
        orchestrator picks the right class and a short run completes."""
        result = run_twist_average_gfmc(
            hamiltonian_list=[h_atom_hamiltonian],
            weights=[1.0],
            gfmc_kind="n",
            mcmc_seed=12345,
            num_mcmc_steps=20,
            num_walkers=2,
            num_mcmc_warmup_steps=5,
            num_mcmc_bin_blocks=3,
            gfmc_kwargs={"alat": 0.1, "E_scf": -0.5, "num_mcmc_per_measurement": 16},
        )
        assert np.isfinite(result.energy.E_TA)
        assert np.isfinite(result.energy.E_TA_std)
        assert result.energy.s_twist == 0.0  # N_twist == 1 -> no spread

    # ---------- 14.3.8 _debug variant dispatch ----------
    @pytest.mark.heavy
    @requires_h_fixture
    def test_run_twist_average_gfmc_n_debug_variant_smoke(self, h_atom_hamiltonian):
        """``gfmc_kind='n_debug'`` dispatches to ``_GFMC_n_debug``. Smoke
        pin -- the contract requires all 4 variants to share
        the orchestrator entry point and a short chain on the
        ``_GFMC_n_debug`` path must finish without raising."""
        result = run_twist_average_gfmc(
            hamiltonian_list=[h_atom_hamiltonian],
            weights=[1.0],
            gfmc_kind="n_debug",
            mcmc_seed=12345,
            num_mcmc_steps=20,
            num_walkers=2,
            num_mcmc_warmup_steps=5,
            num_mcmc_bin_blocks=3,
            gfmc_kwargs={"alat": 0.1, "E_scf": -0.5, "num_mcmc_per_measurement": 16},
        )
        assert np.isfinite(result.energy.E_TA)
        assert np.isfinite(result.energy.E_TA_std)


class TestGFMCTwistAverageBZAggregation:
    """BZ-aggregator wiring across N_twist > 1."""

    # ---------- BZ-aggregator wiring ----------
    @pytest.mark.heavy
    @requires_h_fixture
    @pytest.mark.skip_serial(reason="the comm split into two twists needs an MPI world")
    def test_run_twist_average_gfmc_two_twist_uses_bz_combine(self, h_atom_hamiltonian):
        """At ``N_twist == 2`` with identical Hamiltonians + equal weights
        the orchestrator returns the same ``E_TA`` as N_twist=1 (the BZ
        mean of two identical chains with one stored result per chain is
        the same value; ``s_twist`` is exactly zero by construction since
        only one rank holds the result and the inter-comm gather plus the
        BZ mean over two identical entries collapses to one).

        NOTE: this is a *single-process* smoke test for the driver wiring
        -- on a 1-rank ``COMM_WORLD`` the comm split with ``n_twist == 2``
        requires divisibility, so we skip when ``world_size`` cannot be
        split. The real two-twist correctness check is the T1 invariant
        in test_twist_average.py (replication identity at the aggregator
        level); the actual multi-rank MPI driver run is not part of this
        test.
        """
        from mpi4py import MPI

        assert MPI.COMM_WORLD.Get_size() % 2 == 0, "splitting the world into two twists needs an even number of ranks"
        result = run_twist_average_gfmc(
            hamiltonian_list=[h_atom_hamiltonian, h_atom_hamiltonian],
            weights=[1.0, 1.0],
            gfmc_kind="tau",
            mcmc_seed=12345,
            num_mcmc_steps=20,
            num_walkers=2,
            num_mcmc_warmup_steps=5,
            num_mcmc_bin_blocks=3,
            gfmc_kwargs={"tau": 0.1, "alat": 0.1},
        )
        assert np.isfinite(result.energy.E_TA)
        assert result.energy.s_twist >= 0.0


class TestGFMCTwistAverageInputValidation:
    """Input-validation contracts on ``run_twist_average_gfmc``."""

    # ---------- gfmc_kind dispatch / input validation ----------
    @pytest.mark.heavy
    def test_run_twist_average_gfmc_rejects_unknown_kind(self, h_atom_hamiltonian):
        """An unknown ``gfmc_kind`` raises with a clear message; pins the
        dispatch table (``{'tau', 'n', 'tau_debug', 'n_debug'}``)."""
        with pytest.raises(ValueError, match="gfmc_kind"):
            run_twist_average_gfmc(
                hamiltonian_list=[h_atom_hamiltonian],
                weights=[1.0],
                gfmc_kind="invalid",
                num_mcmc_steps=10,
            )

    @pytest.mark.heavy
    def test_run_twist_average_gfmc_rejects_empty_hamiltonian_list(self):
        """Empty ``hamiltonian_list`` raises; pins the input validation."""
        with pytest.raises(ValueError, match="empty"):
            run_twist_average_gfmc(
                hamiltonian_list=[],
                weights=[],
                gfmc_kind="tau",
                num_mcmc_steps=10,
            )

    @pytest.mark.heavy
    def test_run_twist_average_gfmc_rejects_weights_length_mismatch(self, h_atom_hamiltonian):
        """``len(weights) != n_twist`` raises; pins that the caller cannot
        accidentally drop a twist's weight (which would silently produce a
        biased BZ mean)."""
        with pytest.raises(ValueError, match="weights size"):
            run_twist_average_gfmc(
                hamiltonian_list=[h_atom_hamiltonian, h_atom_hamiltonian],
                weights=[1.0],
                gfmc_kind="tau",
                num_mcmc_steps=10,
            )


# ---------- N_twist=1 identity for every GFMC-aggregated estimator ----------


_LITHIUM_TREXIO = Path(__file__).resolve().parent.parent / "trexio_example_files" / "lithium.h5"
requires_lithium_fixture = pytest.mark.skipif(not _LITHIUM_TREXIO.is_file(), reason="lithium.h5 TREXIO fixture not present")


@pytest.fixture(scope="module")
def lithium_hamiltonian():
    """Periodic (PBC Gamma) fixture -- ``get_aT`` is PBC-only, so the
    molecular H-atom fixture above cannot exercise the stress path. Built
    Jastrow-less.  lithium.h5 (2-atom ccECP, 2 valence e) is used instead of
    silicon.h5 (32e): the identity holds for any fixture, and the silicon
    position-derivative e_L kernel carries a 20-30+ GB AD-tape (enough to
    exhaust a 512 GB host) at ~16x lithium's footprint.
    """
    (structure_data, _aos_data, _, _, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        trexio_file=str(_LITHIUM_TREXIO), store_tuple=True
    )
    jastrow_data = Jastrow_data(
        jastrow_one_body_data=None,
        jastrow_two_body_data=None,
        jastrow_three_body_data=None,
    )
    wavefunction_data = Wavefunction_data(jastrow_data=jastrow_data, geminal_data=geminal_mo_data)
    return Hamiltonian_data.build(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )


class TestGFMCTwistAverageEstimatorIdentity:
    """N_twist=1 identity for every estimator the GFMC twist driver
    aggregates (E / aF / aT; GFMC has no generalized-force estimator).

    Mirrors ``TestRunTwistAverageEstimatorQuartetIdentity`` on the MCMC
    side: at ``N_twist = 1`` the orchestrator must reproduce a standalone
    ``GFMC_t`` / ``GFMC_n`` chain's ``get_E`` / ``get_aF`` / ``get_aT``
    exactly (same seed -> identical samples; the BZ combine collapses to
    the identity). The chains are short and unequilibrated -- irrelevant
    for the identity.
    """

    @pytest.mark.heavy
    @requires_lithium_fixture
    @pytest.mark.parametrize(
        "gfmc_kind,gfmc_kwargs",
        [
            ("tau", {"tau": 0.1, "alat": 0.1}),
            ("n", {"alat": 0.1, "E_scf": -0.5}),
        ],
    )
    def test_n_twist_1_identity_E_aF_aT(self, lithium_hamiltonian, gfmc_kind, gfmc_kwargs):
        """N twist 1 identity E aF aT."""
        from jqmc._setting import ATOL_PYTHON_FP64_LOOSE
        from jqmc.jqmc_gfmc import GFMC_n, GFMC_t

        gfmc_cls = {"tau": GFMC_t, "n": GFMC_n}[gfmc_kind]
        if gfmc_kind == "n":
            # GFMC_n has no default projection count (#690); use 16 projections.
            gfmc_kwargs = {**gfmc_kwargs, "num_mcmc_per_measurement": 16}
        num_walkers = 2
        # GFMC's ``mcmc_counter`` counts collected measurements (fewer than
        # raw steps); 30/10/5 clears the get_E warmup/bin guards for both
        # kinds (same knobs as the MPI decomposition test).
        num_steps = 30
        num_warmup = 10
        num_bins = 5
        seed = 12345

        ref = gfmc_cls(
            hamiltonian_data=lithium_hamiltonian,
            mcmc_seed=seed,
            num_walkers=num_walkers,
            comput_position_deriv=True,
            comput_stress=True,
            **gfmc_kwargs,
        )
        ref.run(num_mcmc_steps=num_steps)
        E_ref, sigma_ref, _, _ = ref.get_E(num_mcmc_warmup_steps=num_warmup, num_mcmc_bin_blocks=num_bins)
        F_ref, F_std_ref = ref.get_aF(num_mcmc_warmup_steps=num_warmup, num_mcmc_bin_blocks=num_bins)
        T_ref, T_std_ref = ref.get_aT(num_mcmc_warmup_steps=num_warmup, num_mcmc_bin_blocks=num_bins)

        result = run_twist_average_gfmc(
            hamiltonian_list=[lithium_hamiltonian],
            weights=[1.0],
            gfmc_kind=gfmc_kind,
            mcmc_seed=seed,
            num_mcmc_steps=num_steps,
            num_walkers=num_walkers,
            num_mcmc_warmup_steps=num_warmup,
            num_mcmc_bin_blocks=num_bins,
            compute_force=True,
            compute_stress=True,
            gfmc_kwargs=dict(gfmc_kwargs),
        )

        assert result.energy.E_TA == pytest.approx(float(E_ref), abs=ATOL_PYTHON_FP64_LOOSE)
        assert result.energy.E_TA_std == pytest.approx(float(sigma_ref), abs=ATOL_PYTHON_FP64_LOOSE)
        np.testing.assert_allclose(
            result.force.F_TA, np.asarray(F_ref), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            result.force.F_TA_std, np.asarray(F_std_ref), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            result.stress.F_TA, np.asarray(T_ref), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            result.stress.F_TA_std, np.asarray(T_std_ref), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            result.per_twist_T[0], np.asarray(T_ref), atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO
        )


# ---------- N_twist=2, two physically distinct twists, one process ----------


_LITHIUM_K025_TREXIO = Path(__file__).resolve().parent.parent / "trexio_example_files" / "lithium_k025.h5"
_TWO_TWIST_RAW_WEIGHTS = np.array([1.0, 3.0])
_TWO_TWIST_SEED = 12345
_TWO_TWIST_KNOBS = dict(num_mcmc_steps=30, num_walkers=1, num_mcmc_warmup_steps=10, num_mcmc_bin_blocks=5)
_TWO_TWIST_GFMC_KWARGS = {"tau": {"tau": 0.1, "alat": 0.1}, "n": {"alat": 0.1, "E_scf": -0.5, "num_mcmc_per_measurement": 16}}


def _jastrowless_periodic_hamiltonian(trexio_path):
    (structure_data, _aos_data, _, _, geminal_mo_data, coulomb_potential_data) = read_trexio_file(
        trexio_file=str(trexio_path), store_tuple=True
    )
    wavefunction_data = Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=geminal_mo_data)
    return Hamiltonian_data.build(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=wavefunction_data,
    )


class _RecordingInterComm:
    """Twist-roots comm of the rank that owns twist 1: records its Level-2 SUM operands."""

    def __init__(self):
        self.sent = []

    def Get_rank(self):
        return 0

    def allreduce(self, value, op=None):
        self.sent.append(np.array(value, copy=True))
        return np.array(value, copy=True)


class _ReplayInterComm:
    """Twist-roots comm of the rank that owns twist 0: adds the recorded twist-1 operands."""

    def __init__(self, remote):
        self._remote = remote
        self.calls = 0

    def Get_rank(self):
        return 0

    def allreduce(self, value, op=None):
        remote = self._remote[self.calls]
        self.calls += 1
        return np.asarray(value) + remote


def _run_two_twist_gfmc(gfmc_kind):
    """Real two-rank ``run_twist_average_gfmc`` emulated in one process.

    The world split is replaced so that pass A runs twist 1 (and records its
    inter-comm operands) and pass B runs twist 0 and receives pass A's
    operands through the inter-comm SUM, which is exactly the production
    2-rank data flow. The intra comm is the real 1-rank ``COMM_WORLD``, so each
    per-twist GFMC chain reduces and branches through MPI as in production.
    Standalone per-twist references are built from the same constructor
    arguments the driver documents (seed ``_twist_sampler_seed(seed, k, 2)``,
    no twist weight anywhere in the chain).
    """
    from mpi4py import MPI

    import jqmc._twist_average as twist_average
    from jqmc._twist_average import _twist_sampler_seed
    from jqmc.jqmc_gfmc import GFMC_n, GFMC_t

    hams = [
        _jastrowless_periodic_hamiltonian(_LITHIUM_TREXIO),
        _jastrowless_periodic_hamiltonian(_LITHIUM_K025_TREXIO),
    ]
    world = MPI.COMM_WORLD
    recorder = _RecordingInterComm()

    def _call(twist_index, inter):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                twist_average,
                "split_world_for_twists",
                lambda world_comm, n_twist: SimpleNamespace(intra_comm=world, inter_comm=inter, twist_index=twist_index),
            )
            return run_twist_average_gfmc(
                hamiltonian_list=hams,
                weights=_TWO_TWIST_RAW_WEIGHTS,
                gfmc_kind=gfmc_kind,
                mcmc_seed=_TWO_TWIST_SEED,
                world_comm=world,
                compute_force=True,
                compute_stress=True,
                gfmc_kwargs=dict(_TWO_TWIST_GFMC_KWARGS[gfmc_kind]),
                **_TWO_TWIST_KNOBS,
            )

    _call(1, recorder)
    replay = _ReplayInterComm(recorder.sent)
    result = _call(0, replay)
    assert replay.calls == len(recorder.sent) == 9, "E, E_std, stopped, F, F_std, T, T_std, P, P_std Level-2 SUMs"
    # Both twists ran all their branching steps, so the ``per_twist_stopped`` operand is zeros.
    assert recorder.sent[2].dtype == np.int64
    np.testing.assert_array_equal(recorder.sent[2], np.zeros(len(hams), dtype=np.int64))

    gfmc_cls = {"tau": GFMC_t, "n": GFMC_n}[gfmc_kind]
    warm = _TWO_TWIST_KNOBS["num_mcmc_warmup_steps"]
    bins = _TWO_TWIST_KNOBS["num_mcmc_bin_blocks"]
    standalone = []
    for k_idx, ham in enumerate(hams):
        ref = gfmc_cls(
            hamiltonian_data=ham,
            mcmc_seed=_twist_sampler_seed(_TWO_TWIST_SEED, k_idx, len(hams)),
            num_walkers=_TWO_TWIST_KNOBS["num_walkers"],
            mpi_comm_in=world,
            world_comm_in=world,
            comput_position_deriv=True,
            comput_stress=True,
            **_TWO_TWIST_GFMC_KWARGS[gfmc_kind],
        )
        ref.run(num_mcmc_steps=_TWO_TWIST_KNOBS["num_mcmc_steps"])
        E, E_std, _, _ = ref.get_E(num_mcmc_warmup_steps=warm, num_mcmc_bin_blocks=bins)
        F, F_std = ref.get_aF(num_mcmc_warmup_steps=warm, num_mcmc_bin_blocks=bins)
        T, T_std = ref.get_aT(num_mcmc_warmup_steps=warm, num_mcmc_bin_blocks=bins)
        P, P_std = ref.get_P_hydrostatic(num_mcmc_warmup_steps=warm, num_mcmc_bin_blocks=bins)
        # get_aT / get_P_hydrostatic follow the root-owns convention (zeros on
        # rank != 0), while get_E / get_aF broadcast. The driver broadcasts its
        # per-twist T / P from the intra root, so under ``mpirun -np 2`` the
        # reference must carry the root's values on every rank too.
        T, T_std, P, P_std = world.bcast((T, T_std, P, P_std) if world.Get_rank() == 0 else None, root=0)
        standalone.append(
            dict(
                E=float(E),
                E_std=float(E_std),
                F=np.asarray(F, dtype=np.float64),
                F_std=np.asarray(F_std, dtype=np.float64),
                T=np.asarray(T, dtype=np.float64),
                T_std=np.asarray(T_std, dtype=np.float64),
                P=float(P),
                P_std=float(P_std),
            )
        )
    return result, standalone


_TWO_TWIST_CACHE = {}


def _two_twist_runs(gfmc_kind):
    if gfmc_kind not in _TWO_TWIST_CACHE:
        _TWO_TWIST_CACHE[gfmc_kind] = _run_two_twist_gfmc(gfmc_kind)
    return _TWO_TWIST_CACHE[gfmc_kind]


def _assert_per_twist_equals_standalone(result, standalone):
    """Driver per-twist E/F/T/P (and stds) are byte-identical to the standalone chains."""
    for k_idx, ref in enumerate(standalone):
        np.testing.assert_array_equal(result.per_twist_E[k_idx], ref["E"], err_msg=f"twist {k_idx} E")
        np.testing.assert_array_equal(result.per_twist_E_std[k_idx], ref["E_std"], err_msg=f"twist {k_idx} E_std")
        np.testing.assert_array_equal(result.per_twist_F[k_idx], ref["F"], err_msg=f"twist {k_idx} F")
        np.testing.assert_array_equal(result.per_twist_F_std[k_idx], ref["F_std"], err_msg=f"twist {k_idx} F_std")
        np.testing.assert_array_equal(result.per_twist_T[k_idx], ref["T"], err_msg=f"twist {k_idx} T")
        np.testing.assert_array_equal(result.per_twist_T_std[k_idx], ref["T_std"], err_msg=f"twist {k_idx} T_std")
        np.testing.assert_array_equal(result.per_twist_P_hydro[k_idx, 0], ref["P"], err_msg=f"twist {k_idx} P")
        np.testing.assert_array_equal(result.per_twist_P_hydro_std[k_idx, 0], ref["P_std"], err_msg=f"twist {k_idx} P_std")


def _bump_one_ulp(values, index):
    """Copy of ``values`` with the entry at ``index`` moved to the next float64 up."""
    bumped = np.array(values, dtype=np.float64, copy=True)
    bumped[index] = np.nextafter(bumped[index], np.inf)
    return bumped


def _assert_bz_hand_combine(result, raw_weights):
    """Documented BZ combine: W = w / sum(w); mean = sum W_k X_k; std = sqrt(sum W_k^2 s_k^2)."""
    W = np.asarray(raw_weights, dtype=np.float64) / np.sum(raw_weights)
    E_k, s_k = result.per_twist_E, result.per_twist_E_std
    # The central pair is spelled at each comparison (not forwarded through a
    # ``**dict``) so the GDD receipt can bind the exact symbols of every site.
    np.testing.assert_allclose(
        result.energy.E_TA,
        W[0] * E_k[0] + W[1] * E_k[1],
        err_msg="E_TA",
        atol=ATOL_PYTHON_FP64_BZ_MEAN,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    np.testing.assert_allclose(
        result.energy.E_TA_std,
        np.sqrt(W[0] ** 2 * s_k[0] ** 2 + W[1] ** 2 * s_k[1] ** 2),
        err_msg="E_TA_std",
        atol=ATOL_PYTHON_FP64_BZ_MEAN,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    np.testing.assert_allclose(
        result.energy.s_twist,
        np.sqrt(W[0] * (E_k[0] - result.energy.E_TA) ** 2 + W[1] * (E_k[1] - result.energy.E_TA) ** 2),
        err_msg="s_twist",
        atol=ATOL_PYTHON_FP64_BZ_MEAN,
        rtol=RTOL_PYTHON_FP64_ZERO,
    )
    for label, agg, per, per_std in (
        ("F_TA", result.force, result.per_twist_F, result.per_twist_F_std),
        ("T_TA", result.stress, result.per_twist_T, result.per_twist_T_std),
        ("P_TA", result.P_hydro, result.per_twist_P_hydro, result.per_twist_P_hydro_std),
    ):
        np.testing.assert_allclose(
            agg.F_TA,
            W[0] * per[0] + W[1] * per[1],
            err_msg=label,
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )
        np.testing.assert_allclose(
            agg.F_TA_std,
            np.sqrt(W[0] ** 2 * per_std[0] ** 2 + W[1] ** 2 * per_std[1] ** 2),
            err_msg=f"{label}_std",
            atol=ATOL_PYTHON_FP64_BZ_MEAN,
            rtol=RTOL_PYTHON_FP64_ZERO,
        )


@pytest.mark.heavy
class TestGFMCTwoTwistCombine:
    """Real GFMC_t / GFMC_n on two physically distinct twists (Gamma real, k=0.25 complex), one process."""

    @pytest.mark.parametrize("gfmc_kind", ["tau", "n"])
    def test_two_distinct_twists_combine_equals_hand_weighted_standalone_runs(self, gfmc_kind):
        """Per-twist E/F/T/P equal standalone chains; BZ combine equals the hand formula.

        The per-twist outputs of the driver must be byte-identical to two
        standalone GFMC chains built with the documented per-twist seed and no
        twist weight: this pins that ``W_k`` never enters the walker weight and
        that warmup/binning happen per twist before the combine. The combined
        energy, error, dispersion, force, stress and pressure must equal the
        documented unequal-weight (1:3) BZ formula applied to those per-twist
        values. The twists must be physically distinct so a weight bug cannot
        cancel.
        """
        result, standalone = _two_twist_runs(gfmc_kind)
        _assert_per_twist_equals_standalone(result, standalone)
        assert result.per_twist_E[0] != result.per_twist_E[1], "twists must be physically distinct"
        assert not np.array_equal(result.per_twist_F[0], result.per_twist_F[1]), "twists must be physically distinct"
        _assert_bz_hand_combine(result, _TWO_TWIST_RAW_WEIGHTS)

    @pytest.mark.parametrize("gfmc_kind", ["tau", "n"])
    def test_per_twist_stress_reference_is_root_value_on_every_rank(self, gfmc_kind):
        """Negative controls for the per-twist T / P identity on every MPI rank.

        The standalone stress / pressure reference must be the root's non-zero value
        on every rank (``get_aT`` returns zeros on rank != 0, so a rank-local reference
        would compare the driver's broadcast T against zeros). On this rank, the
        identity must then reject a one-ULP change of one twist's T or P, and
        rank-local zeros in place of the reference. All collectives run before any
        assertion, and the assertions are reduced over the world, so a failure on one
        rank fails every rank instead of leaving the others blocked in a collective.
        """
        from mpi4py import MPI

        from tests._mpi_collective_check import collective_check

        result, standalone = _two_twist_runs(gfmc_kind)
        world = MPI.COMM_WORLD
        gathered = [world.allgather((ref["T"], ref["T_std"], ref["P"], ref["P_std"])) for ref in standalone]
        with collective_check(world, "per-twist stress / pressure reference"):
            _assert_per_twist_equals_standalone(result, standalone)
            for k_idx, ref in enumerate(standalone):
                assert np.any(ref["T"] != 0.0), f"twist {k_idx}: standalone T is all zeros on rank {world.Get_rank()}"
                for rank, (T, T_std, P, P_std) in enumerate(gathered[k_idx]):
                    np.testing.assert_array_equal(T, ref["T"], err_msg=f"twist {k_idx} T on rank {rank}")
                    np.testing.assert_array_equal(T_std, ref["T_std"], err_msg=f"twist {k_idx} T_std on rank {rank}")
                    assert (P, P_std) == (ref["P"], ref["P_std"]), f"twist {k_idx} P on rank {rank}"
            for k_idx in range(len(standalone)):
                bumped_T = dataclasses.replace(result, per_twist_T=_bump_one_ulp(result.per_twist_T, (k_idx, 0, 0)))
                with pytest.raises(AssertionError, match=f"twist {k_idx} T"):
                    _assert_per_twist_equals_standalone(bumped_T, standalone)
                bumped_P = dataclasses.replace(result, per_twist_P_hydro=_bump_one_ulp(result.per_twist_P_hydro, (k_idx, 0)))
                with pytest.raises(AssertionError, match=f"twist {k_idx} P"):
                    _assert_per_twist_equals_standalone(bumped_P, standalone)
                rank_local = [dict(ref) for ref in standalone]
                rank_local[k_idx]["T"] = np.zeros((3, 3))
                with pytest.raises(AssertionError, match=f"twist {k_idx} T"):
                    _assert_per_twist_equals_standalone(result, rank_local)

    @pytest.mark.parametrize("gfmc_kind", ["tau", "n"])
    def test_two_twist_combine_oracle_rejects_perturbed_weight_and_twist_value(self, gfmc_kind):
        """Negative controls on the same real run: a changed twist weight, or one
        twist's energy moved by 1 mHa, must each fail the hand-combine oracle on ``E_TA``.
        """
        result, _ = _two_twist_runs(gfmc_kind)
        _assert_bz_hand_combine(result, _TWO_TWIST_RAW_WEIGHTS)
        with pytest.raises(AssertionError, match="E_TA"):
            _assert_bz_hand_combine(result, np.array([1.0, 2.0]))
        shifted = dataclasses.replace(result, per_twist_E=result.per_twist_E + np.array([0.0, 1.0e-3]))
        with pytest.raises(AssertionError, match="E_TA"):
            _assert_bz_hand_combine(shifted, _TWO_TWIST_RAW_WEIGHTS)
