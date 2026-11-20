"""Column-block two-comm reduction.

Twist-averaged SR/LM splits the parameter space into a SHARED Jastrow
block (reduced over the all-rank ``world_comm``) and a PER-TWIST det
block (reduced over the per-twist ``intra_comm``). The column-block
approximation drops the Jastrow <-> det cross block in the Fisher
information matrix so the SR step decouples into a single shared
Jastrow update + N_twist independent det updates.

These tests pin the math at the helper level
(``MCMC._classify_block_columns_shared`` /
``MCMC._planb_reduce_matrix`` / ``MCMC._planb_reduce_vector``) without
needing a real MPI fixture. The single-twist contract (``world_comm is
intra_comm`` -> byte-identical to the single-comm SUM) is
exercised in every existing complex-Bloch / get_aH test.
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

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
from mpi4py import MPI

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._setting import ATOL_PYTHON_FP64_BZ_MEAN
from jqmc._setting import RTOL_PYTHON_FP64_ZERO
from jqmc.jqmc_mcmc import MCMC


@dataclass
class _Block:
    """Minimal stand-in for jqmc.wavefunction.VariationalParameterBlock --
    only ``name`` and ``size`` are inspected by the column classifier."""

    name: str
    size: int


# ---------- column-block classifier ----------


class TestColumnClassifier:
    @pytest.mark.light
    def test_classify_block_columns_marks_jastrow_shared_det_per_twist(self):
        """The classifier flags every Jastrow-family block as SHARED and
        every det-family block (``lambda_matrix`` / ``mo_coefficients``) as
        PER-TWIST. The mask length equals ``sum(block.size)``."""
        blocks = [
            _Block(name="j1_param", size=2),
            _Block(name="j2_param", size=1),
            _Block(name="j_matrix", size=3),
            _Block(name="lambda_matrix", size=4),
            _Block(name="mo_coefficients", size=5),
            _Block(name="jastrow_dtn_u", size=2),
        ]
        mask = MCMC._classify_block_columns_shared(blocks)
        assert mask.shape == (2 + 1 + 3 + 4 + 5 + 2,)
        assert mask.dtype == np.bool_
        # First 2 + 1 + 3 = 6 entries are Jastrow (shared).
        assert mask[:6].all()
        # Next 4 + 5 = 9 entries are det (per-twist).
        assert not mask[6:15].any()
        # Final 2 entries are Jastrow-DTN (shared).
        assert mask[15:17].all()

    @pytest.mark.light
    def test_classify_block_columns_respects_chosen_param_index(self):
        """``chosen_param_index`` slices the mask to the same subset used by
        ``get_dln_WF`` / ``get_aH`` -- the two-comm routing has to align."""
        blocks = [_Block(name="j1_param", size=2), _Block(name="lambda_matrix", size=3)]
        # Full mask is [True, True, False, False, False]; select indices 0, 2, 3.
        mask = MCMC._classify_block_columns_shared(blocks, chosen_param_index=[0, 2, 3])
        np.testing.assert_array_equal(mask, np.array([True, False, False]))

    @pytest.mark.light
    def test_classify_block_columns_marks_new_direct_mo_opt_blocks_per_twist(self):
        """The det-block names ``mo_coefficients_up`` / ``mo_coefficients_dn`` (AGPn / SD
        direct MO opt) and ``lambda_diag`` / ``lambda_unpaired`` (AGPn
        natural-orbital occupations + open-shell unpaired block) are PER-TWIST:
        the classifier must mark all of them False; otherwise
        the two-comm reduction reduces det columns over ``world_comm`` (wrong) and the
        ``f_vec`` per-block routing breaks.

        A block name missing from ``_DET_BLOCK_NAMES`` defaults to
        ``shared=True``; an LM run with ``lm_subspace_dim>0`` then indexes a
        subspace-sized ``is_shared_col`` with the full-N
        ``chosen_param_index`` and raises an 'index out of bounds' IndexError.
        """
        blocks = [
            _Block(name="j2_param", size=1),
            _Block(name="mo_coefficients_up", size=4),
            _Block(name="mo_coefficients_dn", size=4),
            _Block(name="lambda_diag", size=2),
            _Block(name="lambda_unpaired", size=3),
        ]
        mask = MCMC._classify_block_columns_shared(blocks)
        # j2_param shared, all det blocks per-twist.
        assert mask[0] is np.True_ or mask[0] == True  # j2_param
        # 4 mo_up + 4 mo_dn + 2 lambda_diag + 3 lambda_unpaired = 13 det entries.
        assert not mask[1:14].any(), f"new direct-MO opt block names must all classify as per-twist (False); got mask={mask}"
        assert mask.size == 14

    @pytest.mark.light
    def test_classify_block_columns_subset_size_matches_chosen_param_index(self):
        """Regression guard for the ``is_shared_col[chosen_param_index]``
        IndexError surfaced under LM + ``lm_subspace_dim>0``: the returned
        mask must have length ``len(chosen_param_index)``, not the full
        block-sum length. The L3173 extension path in ``get_aH`` relies on
        this invariant to build the post-collective-variable shared mask.
        """
        blocks = [
            _Block(name="j2_param", size=1),
            _Block(name="mo_coefficients_up", size=700),
            _Block(name="mo_coefficients_dn", size=966),
            _Block(name="lambda_diag", size=266),
        ]
        cpi = [0, 565, 420, 690, 489, 1363, 1064, 1106, 350, 495]  # top-10 by SN (full-N indices)
        mask = MCMC._classify_block_columns_shared(blocks, chosen_param_index=cpi)
        assert mask.size == len(cpi) == 10
        # idx 0 is j2_param -> shared True; the rest land in det blocks -> False.
        assert bool(mask[0]) is True
        assert not mask[1:].any()


# ---------- single-twist short-circuit (world is intra) ----------


class TestSingleTwistShortCircuit:
    @pytest.mark.light
    def test_planb_reduce_matrix_single_twist_byte_identical(self):
        """At a single-twist run ``world_comm is intra_comm``; the two-comm
        matrix reducer short-circuits to a single comm reduction + a single
        normalisation, which must be byte-identical to the single-comm
        path (``Allreduce(M_local) / total_w``)."""
        rng = np.random.default_rng(0)
        K = 6
        M_local = rng.standard_normal((K, K))
        is_shared = np.array([True, True, False, False, False, True])
        total_w = 12.34
        comm = MPI.COMM_WORLD
        # On a single-process run any comm is its own self; ``world is intra``
        # picks up the short-circuit branch regardless of mpi_size.
        M_global = MCMC._planb_reduce_matrix(M_local, is_shared, comm, comm, total_w, total_w)
        # Single-comm reference (single Allreduce + single normaliser).
        from jqmc.jqmc_mcmc import allreduce_sum

        M_ref = allreduce_sum(comm, M_local) / total_w
        np.testing.assert_array_equal(M_global, M_ref)

    @pytest.mark.light
    def test_planb_reduce_vector_single_twist_byte_identical(self):
        """Same byte-identity contract for the vector (``f``) reducer."""
        rng = np.random.default_rng(1)
        v_local = rng.standard_normal(5)
        is_shared = np.array([True, False, True, False, False])
        total_w = 9.87
        comm = MPI.COMM_WORLD
        v_global = MCMC._planb_reduce_vector(v_local, is_shared, comm, comm, total_w, total_w)
        from jqmc.jqmc_mcmc import allreduce_sum

        v_ref = allreduce_sum(comm, v_local) / total_w
        np.testing.assert_array_equal(v_global, v_ref)


# ---------- two-twist routing emulated on a single process ----------
#
# A real two-twist run needs two distinct comms (world and intra). On a
# single-process pytest harness ``MPI.COMM_WORLD`` is the only real
# comm. We emulate the two-comm math by passing the SAME comm twice but
# with DIFFERENT normalisers -- the routing decision keys on identity
# (``world_comm is intra_comm``), so to force the two-branch path we
# build a thin wrapper that compares as a distinct object.


class _DistinctComm:
    """Wrapper that delegates to a real comm but compares as non-identical
    via ``is`` -- enough to drive the two-comm two-branch path."""

    def __init__(self, comm):
        self._comm = comm

    def allreduce(self, *args, **kwargs):
        return self._comm.allreduce(*args, **kwargs)

    def Allreduce(self, *args, **kwargs):
        return self._comm.Allreduce(*args, **kwargs)

    def Get_size(self):
        return self._comm.Get_size()

    def Get_rank(self):
        return self._comm.Get_rank()


class TestTwoCommPlanBRouting:
    @pytest.mark.light
    def test_planb_reduce_matrix_two_comm_uses_block_specific_normalisers(self):
        """When ``world_comm is not intra_comm`` the helper routes Jastrow
        columns through world (with ``total_w_world`` normaliser) and det
        columns through intra (with ``total_w_intra``). Cross blocks are
        zeroed (the column-block approximation drops them).
        """
        rng = np.random.default_rng(42)
        K = 5
        M_local = rng.standard_normal((K, K))
        is_shared = np.array([True, True, False, False, False])

        world = _DistinctComm(MPI.COMM_WORLD)
        intra = MPI.COMM_WORLD  # different object than `world` (== short-circuits)
        total_w_world = 100.0
        total_w_intra = 25.0

        M_global = MCMC._planb_reduce_matrix(M_local, is_shared, world, intra, total_w_world, total_w_intra)

        # JJ block: M_local[J, J] / total_w_world (no Allreduce SUM on
        # single-process; sum == identity).
        M_JJ_ref = M_local[np.ix_([0, 1], [0, 1])] / total_w_world
        np.testing.assert_allclose(
            M_global[np.ix_([0, 1], [0, 1])], M_JJ_ref, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
        )

        # DD block: M_local[D, D] / total_w_intra.
        M_DD_ref = M_local[np.ix_([2, 3, 4], [2, 3, 4])] / total_w_intra
        np.testing.assert_allclose(
            M_global[np.ix_([2, 3, 4], [2, 3, 4])], M_DD_ref, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
        )

        # Cross blocks (JD and DJ) are zero -- the column-block approximation drops them.
        np.testing.assert_array_equal(M_global[np.ix_([0, 1], [2, 3, 4])], 0.0)
        np.testing.assert_array_equal(M_global[np.ix_([2, 3, 4], [0, 1])], 0.0)

    @pytest.mark.light
    def test_planb_reduce_vector_two_comm_uses_block_specific_normalisers(self):
        """Vector counterpart: Jastrow entries normalised by world, det
        entries by intra; no cross-block analogue (the vector has no
        off-diagonal slot)."""
        rng = np.random.default_rng(43)
        v_local = rng.standard_normal(6)
        is_shared = np.array([True, False, True, False, False, True])

        world = _DistinctComm(MPI.COMM_WORLD)
        intra = MPI.COMM_WORLD
        total_w_world = 3.0
        total_w_intra = 0.5

        v_global = MCMC._planb_reduce_vector(v_local, is_shared, world, intra, total_w_world, total_w_intra)

        J_idx = np.array([0, 2, 5])
        D_idx = np.array([1, 3, 4])
        np.testing.assert_allclose(
            v_global[J_idx], v_local[J_idx] / total_w_world, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
        )
        np.testing.assert_allclose(
            v_global[D_idx], v_local[D_idx] / total_w_intra, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO
        )

    @pytest.mark.light
    def test_planb_reduce_matrix_handles_all_det_no_jastrow(self):
        """Edge case: a det-only optimisation (Jastrow frozen). The Jastrow
        branch is skipped; the result is identical to the single-comm intra
        reduction on the full matrix."""
        rng = np.random.default_rng(44)
        K = 4
        M_local = rng.standard_normal((K, K))
        is_shared = np.zeros(K, dtype=bool)
        world = _DistinctComm(MPI.COMM_WORLD)
        intra = MPI.COMM_WORLD
        total_w_world = 99.0
        total_w_intra = 11.0
        M_global = MCMC._planb_reduce_matrix(M_local, is_shared, world, intra, total_w_world, total_w_intra)
        # All entries go through the intra branch, normalised by total_w_intra.
        np.testing.assert_allclose(M_global, M_local / total_w_intra, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO)

    @pytest.mark.light
    def test_planb_reduce_matrix_handles_all_jastrow_no_det(self):
        """Edge case: Jastrow-only optimisation (det frozen, e.g. a frozen-WF
        twist average baseline). The det branch is skipped; the result is
        identical to the single-comm world reduction on the full matrix."""
        rng = np.random.default_rng(45)
        K = 4
        M_local = rng.standard_normal((K, K))
        is_shared = np.ones(K, dtype=bool)
        world = _DistinctComm(MPI.COMM_WORLD)
        intra = MPI.COMM_WORLD
        total_w_world = 99.0
        total_w_intra = 11.0
        M_global = MCMC._planb_reduce_matrix(M_local, is_shared, world, intra, total_w_world, total_w_intra)
        # All entries go through the world branch, normalised by total_w_world.
        np.testing.assert_allclose(M_global, M_local / total_w_world, atol=ATOL_PYTHON_FP64_BZ_MEAN, rtol=RTOL_PYTHON_FP64_ZERO)
