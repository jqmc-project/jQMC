"""Per-twist checkpoint tests.

A twist-averaged run writes two HDF5 files under ``checkpoint_dir``:

* ``twist_{idx:03d}.h5`` -- per-twist driver state (intra-rank merged)
  plus ``_meta`` attrs ``twist_index`` / ``target_steps`` / ``converged``
  and the cached ``(E_k, sigma_k)`` so resume does not have to re-run
  any completed twist's ``get_E``;
* ``shared_state.h5`` -- the shared Jastrow block (single source of
  truth; under twist-averaged optimization it moves every SR step).

These tests pin:

* the round-trip of the per-twist summary (write + load gives back the
  same ``(E_k, sigma_k)`` and ``converged`` flag);
* the ``is_twist_complete`` predicate's three-attr contract (file
  present + ``converged`` + ``target_steps`` at least the requested
  count) -- a file with ``converged=False`` or a stored target below
  the requested count is NOT certified;
* shared-state round-trip (write Jastrow + load gives back an equal
  Jastrow_data instance, dtype-preserving).

The end-to-end "skip-completed-twists" path that uses these helpers
inside ``run_twist_average_from_hamiltonians`` is exercised by
``tests/test_jqmc_kernel/test_twist_resume.py`` (byte-identity).
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

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._twist_checkpoint import (
    is_twist_complete,
    load_twist_checkpoint_manifest,
    load_twist_checkpoint_summary,
    save_shared_state,
    load_shared_state,
    shared_state_path,
    twist_checkpoint_path,
)


class TestIsTwistComplete:
    """``is_twist_complete`` predicate contract (4 tests)."""

    @pytest.mark.light
    def test_is_twist_complete_returns_false_when_file_missing(self, tmp_path):
        """Missing ``twist_{idx}.h5`` -> NOT complete. Pins the "fresh run"
        entry path: the driver's resume sweep must not certify a twist with
        no checkpoint on disk."""
        assert is_twist_complete(str(tmp_path), 0, target_steps=100) is False

    @pytest.mark.light
    def test_is_twist_complete_keys_on_converged_attr(self, tmp_path):
        """A file with ``converged=False`` is NOT complete even when
        ``target_steps`` matches. Pins the contract
        that partial files (chain ran out of wall-time, walker count was
        short) are skipped rather than re-sampled."""
        path = twist_checkpoint_path(str(tmp_path), 0)
        with h5py.File(path, "w") as f:
            meta = f.create_group("_meta")
            meta.attrs["twist_index"] = 0
            meta.attrs["target_steps"] = 100
            meta.attrs["converged"] = False
        assert is_twist_complete(str(tmp_path), 0, target_steps=100) is False

    @pytest.mark.light
    def test_is_twist_complete_keys_on_target_steps_at_least(self, tmp_path):
        """A file with ``converged=True`` but ``target_steps=50`` is NOT
        complete when the caller asks for 100. Pins the "extend a finished
        run by raising num_mcmc_steps" path: re-running with a longer chain
        must re-run the twist, not silently reuse the shorter cached
        result."""
        path = twist_checkpoint_path(str(tmp_path), 0)
        with h5py.File(path, "w") as f:
            meta = f.create_group("_meta")
            meta.attrs["twist_index"] = 0
            meta.attrs["target_steps"] = 50
            meta.attrs["converged"] = True
        assert is_twist_complete(str(tmp_path), 0, target_steps=100) is False
        # ... but is complete when the caller asks for <= the stored target.
        assert is_twist_complete(str(tmp_path), 0, target_steps=50) is True
        assert is_twist_complete(str(tmp_path), 0, target_steps=25) is True

    @pytest.mark.light
    def test_is_twist_complete_passes_when_all_three_present(self, tmp_path):
        """File present + converged=True + target_steps >= requested -> True."""
        path = twist_checkpoint_path(str(tmp_path), 7)
        with h5py.File(path, "w") as f:
            meta = f.create_group("_meta")
            meta.attrs["twist_index"] = 7
            meta.attrs["target_steps"] = 200
            meta.attrs["converged"] = True
        assert is_twist_complete(str(tmp_path), 7, target_steps=200) is True


class TestCheckpointSummary:
    """Per-twist summary round-trip (1 test)."""

    @pytest.mark.light
    def test_load_twist_checkpoint_summary_reads_attrs_and_datasets(self, tmp_path):
        """``load_twist_checkpoint_summary`` returns a dict with ``E_k`` /
        ``sigma_k`` attrs and ``F_k`` / ``sigma_F_k`` datasets when they
        were written. Pins the "resume skips completed twist + injects its
        cached (E_k, sigma_k) into the BZ aggregator" flow."""
        path = twist_checkpoint_path(str(tmp_path), 3)
        F = np.array([[1.0, -2.0, 0.5], [0.1, 0.2, 0.3]])
        sigma_F = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
        with h5py.File(path, "w") as f:
            meta = f.create_group("_meta")
            meta.attrs["twist_index"] = 3
            meta.attrs["target_steps"] = 100
            meta.attrs["converged"] = True
            meta.attrs["normalized_twist_weight"] = 0.125
            meta.attrs["num_twists"] = 8
            meta.attrs["E_k"] = -1.2345
            meta.attrs["sigma_k"] = 0.0017
            meta.create_dataset("F_k", data=F)
            meta.create_dataset("sigma_F_k", data=sigma_F)
        summary = load_twist_checkpoint_summary(str(tmp_path), 3)
        assert summary["twist_index"] == 3
        assert summary["target_steps"] == 100
        assert summary["converged"] is True
        assert summary["normalized_twist_weight"] == pytest.approx(0.125)
        assert summary["num_twists"] == 8
        assert summary["E_k"] == pytest.approx(-1.2345)
        assert summary["sigma_k"] == pytest.approx(0.0017)
        np.testing.assert_array_equal(summary["F_k"], F)
        np.testing.assert_array_equal(summary["sigma_F_k"], sigma_F)


class TestCheckpointManifest:
    """Self-contained twist-directory manifest contract."""

    @staticmethod
    def _write_entry(
        checkpoint_dir,
        filename_index,
        *,
        metadata_index=None,
        weight=1.0,
        num_twists=1,
        converged=True,
    ):
        path = twist_checkpoint_path(str(checkpoint_dir), filename_index)
        with h5py.File(path, "w") as f:
            meta = f.create_group("_meta")
            meta.attrs["twist_index"] = filename_index if metadata_index is None else metadata_index
            meta.attrs["converged"] = converged
            if weight is not None:
                meta.attrs["normalized_twist_weight"] = weight
            if num_twists is not None:
                meta.attrs["num_twists"] = num_twists
        return path

    @pytest.mark.light
    def test_load_twist_checkpoint_manifest_returns_canonical_paths_and_normalized_weights(self, tmp_path):
        """A complete canonical directory returns index-ordered paths and BZ weights.

        Proportional stored weights are normalized once by the manifest loader;
        unrelated files cannot enter the returned twist set.
        """
        checkpoint_dir = tmp_path / "restart.h5"
        checkpoint_dir.mkdir()
        path_0 = self._write_entry(checkpoint_dir, 0, weight=2.0, num_twists=2)
        path_1 = self._write_entry(checkpoint_dir, 1, weight=6.0, num_twists=2)
        (checkpoint_dir / "shared_state.h5").touch()

        manifest = load_twist_checkpoint_manifest(str(checkpoint_dir))

        assert manifest == [(path_0, 0.25), (path_1, 0.75)]

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("scenario", "error_pattern"),
        [
            ("missing-directory", "expected a twist checkpoint directory"),
            ("empty-directory", "no canonical twist_NNN.h5 checkpoints"),
            ("metadata-index-mismatch", "does not match filename index"),
            ("unconverged", "twist is not converged"),
            ("pre-upgrade", "pre-upgrade checkpoint directories are unsupported"),
            ("inconsistent-count", "inconsistent num_twists metadata"),
            ("incomplete-indices", "incomplete twist checkpoint set"),
            ("non-finite-weight", "weights must be finite"),
            ("negative-weight", "weights must be non-negative"),
            ("zero-weight-sum", "sum of weights must be positive"),
        ],
    )
    def test_load_twist_checkpoint_manifest_rejects_incomplete_or_inconsistent_directories(
        self, tmp_path, scenario, error_pattern
    ):
        """Invalid manifest identity, completeness, count, and weight states fail closed."""
        checkpoint_dir = tmp_path / "restart.h5"
        if scenario != "missing-directory":
            checkpoint_dir.mkdir()

        if scenario == "metadata-index-mismatch":
            self._write_entry(checkpoint_dir, 0, metadata_index=1)
        elif scenario == "unconverged":
            self._write_entry(checkpoint_dir, 0, converged=False)
        elif scenario == "pre-upgrade":
            self._write_entry(checkpoint_dir, 0, weight=None, num_twists=None)
        elif scenario == "inconsistent-count":
            self._write_entry(checkpoint_dir, 0, weight=0.5, num_twists=2)
            self._write_entry(checkpoint_dir, 1, weight=0.5, num_twists=3)
        elif scenario == "incomplete-indices":
            self._write_entry(checkpoint_dir, 1, weight=1.0, num_twists=2)
        elif scenario == "non-finite-weight":
            self._write_entry(checkpoint_dir, 0, weight=np.nan)
        elif scenario == "negative-weight":
            self._write_entry(checkpoint_dir, 0, weight=-1.0)
        elif scenario == "zero-weight-sum":
            self._write_entry(checkpoint_dir, 0, weight=0.0, num_twists=2)
            self._write_entry(checkpoint_dir, 1, weight=0.0, num_twists=2)

        with pytest.raises(ValueError, match=error_pattern):
            load_twist_checkpoint_manifest(str(checkpoint_dir))


class TestSharedState:
    """Shared Jastrow state round-trip (1 test)."""

    @pytest.mark.light
    def test_shared_state_round_trip_preserves_jastrow(self, tmp_path):
        """``save_shared_state`` + ``load_shared_state`` round-trips the
        Jastrow_data block. Pins the "shared = the
        optimized Jastrow ... checkpointed exactly once" contract: the file
        is the single source of truth, dropped onto every twist on resume."""
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = Path(__file__).parent.parent / "trexio_example_files" / "H_ae_ccpvdz_cart.h5"
        if not fixture.is_file():
            pytest.skip("H_ae_ccpvdz_cart.h5 fixture not present")

        sd, _, _, _, gem, cpd = read_trexio_file(str(fixture), store_tuple=True)
        jastrow = Jastrow_data(jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.234))
        ham = Hamiltonian_data(
            structure_data=sd,
            coulomb_potential_data=cpd,
            wavefunction_data=Wavefunction_data(jastrow_data=jastrow, geminal_data=gem),
        )
        save_shared_state(str(tmp_path), ham)
        assert os.path.isfile(shared_state_path(str(tmp_path)))
        loaded = load_shared_state(str(tmp_path))
        # The 2b parameter survives the round-trip up to the post-load dtype
        # alignment (active-zone cast). In FP64 full mode this is bit-exact;
        # in mixed mode it is the loosest jastrow / det / ao zone tolerance.
        # See jqmc._setting.get_tol_hdf5_roundtrip for the contract.
        from jqmc._setting import get_tol_hdf5_roundtrip

        atol, _rtol = get_tol_hdf5_roundtrip("strict")
        assert float(loaded.jastrow_two_body_data.jastrow_2b_param) == pytest.approx(1.234, abs=atol)


class TestTwistIdentityGuard:
    """Regression: resume must never blend stale per-twist energies.

    Before the guard, ``is_twist_complete`` keyed on converged+target_steps
    only: a checkpoint written for a different k-point or a re-optimized
    wavefunction was silently blended into the BZ average.
    """

    @staticmethod
    def _write(path_dir, *, k_point=None, ham_digest=None, E_k=-1.23):
        import h5py

        from jqmc._twist_checkpoint import twist_checkpoint_path

        path = twist_checkpoint_path(str(path_dir), 0)
        with h5py.File(path, "w") as f:
            meta = f.create_group("_meta")
            meta.attrs["twist_index"] = 0
            meta.attrs["target_steps"] = 10
            meta.attrs["converged"] = True
            meta.attrs["E_k"] = E_k
            if k_point is not None:
                meta.attrs["k_point"] = np.asarray(k_point, dtype=np.float64)
            if ham_digest is not None:
                meta.attrs["ham_digest"] = ham_digest

    @pytest.mark.light
    def test_stale_k_point_is_refused(self, tmp_path):
        """Stale k point is refused."""
        from jqmc._twist_checkpoint import is_twist_complete

        self._write(tmp_path, k_point=[0.25, 0.0, 0.0], ham_digest="abc")
        with pytest.raises(ValueError, match="refusing to blend"):
            is_twist_complete(str(tmp_path), 0, 10, k_point=[0.5, 0.0, 0.0], ham_digest="abc")

    @staticmethod
    def _stale_and_resume_digests(change):
        """Digests of a checkpointed wavefunction and of a changed one.

        The reference is not a restated hash. Most changes alter the trial
        wavefunction up to normalization, hence ``E_k``: a Jastrow parameter,
        the Jastrow presence, the imaginary part of the up MOs, and the
        ``*-entry`` cases, which perturb one coefficient of the occupied dn
        MO or one AGP lambda entry and so change a relative weight. The
        uniform ``dn-mo`` and ``agp-lambda`` scalings change only the overall
        normalization (the same ``E_k``); they are kept as conservative
        parameter-identity controls, since a changed stored parameter must
        still not be reused. The base is a complex-MO twist of
        ``lithium_twist_222`` with a two-body Jastrow.
        """
        from dataclasses import replace

        from jqmc._twist_checkpoint import twist_identity_digest
        from jqmc.determinant import Geminal_data
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = Path(__file__).parent.parent / "trexio_example_files" / "lithium_twist_222.h5"
        structure, _, _, _, geminal, coulomb = read_trexio_file(str(fixture), store_tuple=True, twist_ik=1)

        def hamiltonian(geminal_data, jastrow_2b_param):
            jastrow = Jastrow_data()
            if jastrow_2b_param is not None:
                jastrow = Jastrow_data(
                    jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
                        jastrow_2b_param=jastrow_2b_param, structure_data=structure
                    )
                )
            return Hamiltonian_data(
                structure_data=structure,
                coulomb_potential_data=coulomb,
                wavefunction_data=Wavefunction_data(jastrow_data=jastrow, geminal_data=geminal_data),
            )

        up, dn = geminal.orb_data_up_spin, geminal.orb_data_dn_spin
        assert np.iscomplexobj(np.asarray(up.mo_coefficients))
        stored_geminal, resumed_geminal, resumed_j2 = geminal, geminal, 1.0
        if change == "jastrow-2b-param":
            resumed_j2 = 0.5
        elif change == "jastrow-removed":
            resumed_j2 = None
        elif change == "dn-mo":
            resumed_geminal = replace(
                geminal, orb_data_dn_spin=replace(dn, mo_coefficients=np.asarray(dn.mo_coefficients) * 1.1)
            )
        elif change == "dn-mo-entry":
            coefficients = np.array(dn.mo_coefficients)
            coefficients[0, 0] = coefficients[0, 0] * 1.1 + 0.1
            resumed_geminal = replace(geminal, orb_data_dn_spin=replace(dn, mo_coefficients=coefficients))
        elif change == "up-mo-imag-dropped":
            resumed_geminal = replace(
                geminal, orb_data_up_spin=replace(up, mo_coefficients=np.asarray(up.mo_coefficients).real)
            )
        elif change == "agp-lambda":
            stored_geminal = Geminal_data.convert_from_MOs_to_AOs(geminal)
            resumed_geminal = replace(stored_geminal, lambda_matrix=np.asarray(stored_geminal.lambda_matrix) * 1.1)
        elif change == "agp-lambda-entry":
            stored_geminal = Geminal_data.convert_from_MOs_to_AOs(geminal)
            lambda_matrix = np.array(stored_geminal.lambda_matrix)
            lambda_matrix[0, 0] = lambda_matrix[0, 0] * 1.1 + 0.1
            resumed_geminal = replace(stored_geminal, lambda_matrix=lambda_matrix)
        else:
            raise AssertionError(change)
        return twist_identity_digest(hamiltonian(stored_geminal, 1.0)), twist_identity_digest(
            hamiltonian(resumed_geminal, resumed_j2)
        )

    @pytest.mark.light
    @pytest.mark.parametrize(
        "change",
        [
            "literal",
            "jastrow-2b-param",
            "jastrow-removed",
            "dn-mo",
            "dn-mo-entry",
            "up-mo-imag-dropped",
            "agp-lambda",
            "agp-lambda-entry",
        ],
    )
    def test_stale_wavefunction_digest_is_refused(self, tmp_path, change):
        """Stale wavefunction digest is refused.

        Besides a literal mismatch, a change of any re-optimizable part of
        the trial wavefunction (Jastrow parameters and presence, dn-spin MOs,
        the imaginary part of complex MOs, the AGP lambda matrix) must change
        ``twist_identity_digest`` so a stale ``E_k`` is never reused. This
        holds both for changes that alter ``E_k`` (relative coefficients) and
        for uniform rescalings that only renormalize the wavefunction.
        """
        from jqmc._twist_checkpoint import is_twist_complete

        if change == "literal":
            stored, resumed = "abc", "zzz"
        else:
            stored, resumed = self._stale_and_resume_digests(change)
        self._write(tmp_path, k_point=[0.25, 0.0, 0.0], ham_digest=stored)
        with pytest.raises(ValueError, match="identity digest mismatch"):
            is_twist_complete(str(tmp_path), 0, 10, k_point=[0.25, 0.0, 0.0], ham_digest=resumed)

    @pytest.mark.light
    def test_legacy_checkpoint_without_identity_recomputes(self, tmp_path):
        """Legacy checkpoint without identity recomputes."""
        from jqmc._twist_checkpoint import is_twist_complete

        self._write(tmp_path)  # pre-upgrade checkpoint: no identity attrs
        assert is_twist_complete(str(tmp_path), 0, 10, ham_digest="abc") is False

    @pytest.mark.light
    def test_non_finite_stored_energy_is_refused(self, tmp_path):
        """Non finite stored energy is refused."""
        from jqmc._twist_checkpoint import is_twist_complete

        self._write(tmp_path, k_point=[0.0, 0.0, 0.0], ham_digest="abc", E_k=float("nan"))
        with pytest.raises(ValueError, match="non-finite E_k"):
            is_twist_complete(str(tmp_path), 0, 10, k_point=[0.0, 0.0, 0.0], ham_digest="abc")

    @pytest.mark.light
    def test_matching_identity_stays_complete(self, tmp_path):
        """Matching identity stays complete."""
        from jqmc._twist_checkpoint import is_twist_complete

        self._write(tmp_path, k_point=[0.25, 0.0, 0.0], ham_digest="abc")
        assert is_twist_complete(str(tmp_path), 0, 10, k_point=[0.25, 0.0, 0.0], ham_digest="abc") is True

    @pytest.mark.light
    @pytest.mark.parametrize("route", ["mcmc", "gfmc_tau"])
    @pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
    def test_resume_refuses_another_twist_count_or_bz_weight(self, tmp_path, route):
        """A one-twist resume refuses twists stored for another twist set.

        jqmc-tool combines every ``twist_NNN.h5`` of the directory with its
        stored ``normalized_twist_weight``, so reusing a twist whose stored
        count or weight differs from the input, or leaving a twist file beyond
        the input's twist count, would make the directory report a BZ average
        other than the run's. The stored state is the ``_meta`` of a completed
        H2 Gamma twist (its real k point and identity digest). Each refusal
        happens before any sampling and leaves the files unchanged.
        """
        from jqmc._setting import GFMC_MIN_BIN_BLOCKS, GFMC_MIN_COLLECT_STEPS, GFMC_MIN_STEPS, GFMC_MIN_WARMUP_STEPS
        from jqmc._twist_average import (
            _twist_k_point_from_hamiltonian,
            run_twist_average_from_hamiltonians,
            run_twist_average_gfmc,
        )
        from jqmc._twist_checkpoint import twist_identity_digest
        from jqmc.hamiltonians import Hamiltonian_data
        from jqmc.jastrow_factor import Jastrow_data
        from jqmc.trexio_wrapper import read_trexio_file
        from jqmc.wavefunction import Wavefunction_data

        fixture = Path(__file__).parent.parent / "trexio_example_files" / "H2_pbc_ae_cart.h5"
        structure, _, _, _, geminal, coulomb = read_trexio_file(str(fixture), store_tuple=True)
        ham = Hamiltonian_data(
            structure_data=structure,
            coulomb_potential_data=coulomb,
            wavefunction_data=Wavefunction_data(geminal_data=geminal, jastrow_data=Jastrow_data()),
        )
        k_point = _twist_k_point_from_hamiltonian(ham)
        steps = GFMC_MIN_STEPS

        def write_twist(index, *, weight, num_twists):
            path = Path(twist_checkpoint_path(str(tmp_path), index))
            with h5py.File(path, "w") as f:
                meta = f.create_group("_meta")
                meta.attrs.update(twist_index=index, target_steps=steps, converged=True, E_k=-1.0, sigma_k=0.1)
                meta.attrs.update(ham_digest=twist_identity_digest(ham), normalized_twist_weight=weight, num_twists=num_twists)
                if k_point is not None:
                    meta.attrs["k_point"] = np.asarray(k_point, dtype=np.float64)
            return path

        def resume():
            common = dict(hamiltonian_list=[ham], weights=[1.0], mcmc_seed=1, num_walkers=2, num_mcmc_steps=steps)
            common.update(checkpoint_dir=str(tmp_path), resume=True)
            if route == "mcmc":
                run_twist_average_from_hamiltonians(
                    **common, num_mcmc_warmup_steps=5, num_mcmc_bin_blocks=3, mcmc_kwargs={"num_mcmc_per_measurement": 1}
                )
            else:
                run_twist_average_gfmc(
                    **common,
                    gfmc_kind="tau",
                    num_mcmc_warmup_steps=GFMC_MIN_WARMUP_STEPS,
                    num_mcmc_bin_blocks=GFMC_MIN_BIN_BLOCKS,
                    gfmc_kwargs={"alat": 0.5, "num_gfmc_collect_steps": GFMC_MIN_COLLECT_STEPS},
                )

        # Twist 0 of a two-twist run with weights 1:3.
        stored = write_twist(0, weight=0.25, num_twists=2)
        saved = stored.read_bytes()
        with pytest.raises(ValueError, match=r"num_twists=2 but resume expects num_twists=1: refusing to blend"):
            resume()
        assert stored.read_bytes() == saved

        stored = write_twist(0, weight=0.25, num_twists=1)
        saved = stored.read_bytes()
        with pytest.raises(ValueError, match=r"normalized BZ weight 0\.25 but resume expects 1\.0: refusing to blend"):
            resume()
        assert stored.read_bytes() == saved

        # A consistent twist 0 next to a twist file the one-twist input does not own.
        stored = write_twist(0, weight=1.0, num_twists=1)
        extra = write_twist(1, weight=0.75, num_twists=2)
        saved = (stored.read_bytes(), extra.read_bytes())
        with pytest.raises(ValueError, match=r"twist_001\.h5 beyond the 1 twist\(s\) of this run: refusing to resume"):
            resume()
        assert (stored.read_bytes(), extra.read_bytes()) == saved


_H_FIXTURE = Path(__file__).parent.parent / "trexio_example_files" / "H_ae_ccpvdz_cart.h5"
requires_h_fixture = pytest.mark.skipif(not _H_FIXTURE.is_file(), reason="H_ae_ccpvdz_cart.h5 fixture not present")


def _build_combined_source_and_checkpoints(tmp_path, *, converged: bool, n_twist: int = 2, jastrow_2b_param: float = 2.75):
    """Write a tiny combined twist input + one optimized per-twist checkpoint each.

    Mirrors what a twist-averaged optimisation leaves behind: the combined input the
    run started from, ``twist_NNN.h5`` per twist (a merged MCMC checkpoint whose
    ``hamiltonian_data`` carries that twist's optimized MOs) and the canonical
    ``shared_state.h5`` Jastrow.  Returns
    ``(source_path, checkpoint_dir, expected_mo_up_per_twist)``.
    """
    from jqmc._twist_average import load_twist, save_twist_average_input
    from jqmc.hamiltonians import (
        Hamiltonian_data,
        _save_dataclass_to_hdf5,
    )
    from jqmc.jastrow_factor import Jastrow_data, Jastrow_two_body_data
    from jqmc.trexio_wrapper import read_trexio_file
    from jqmc.wavefunction import Wavefunction_data

    structure, _, _, _, geminal, coulomb = read_trexio_file(str(_H_FIXTURE), store_tuple=True)
    template = Hamiltonian_data(
        structure_data=structure,
        coulomb_potential_data=coulomb,
        wavefunction_data=Wavefunction_data(jastrow_data=Jastrow_data(), geminal_data=geminal),
    )
    mo_up = np.asarray(geminal.orb_data_up_spin.mo_coefficients, dtype=np.float64)
    mo_dn = np.asarray(geminal.orb_data_dn_spin.mo_coefficients, dtype=np.float64)
    source = tmp_path / "hamiltonian_data.h5"
    save_twist_average_input(
        filepath=str(source),
        template_hamiltonian=template,
        k_points=[None] * n_twist,
        weights=[1.0] * n_twist,
        mo_coefficients_up_per_twist=[mo_up] * n_twist,
        mo_coefficients_dn_per_twist=[mo_dn] * n_twist,
    )

    checkpoint_dir = tmp_path / "restart.h5"
    checkpoint_dir.mkdir(exist_ok=True)
    marker_jastrow = Jastrow_data(
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=jastrow_2b_param,
            jastrow_2b_type="exp",
        )
    )
    expected_up = []
    optimized = None
    for twist_index in range(n_twist):
        optimized = load_twist(str(source), twist_index)
        opt_geminal = optimized.wavefunction_data.geminal_data
        up = np.asarray(opt_geminal.orb_data_up_spin.mo_coefficients) * (1.0 + 0.1 * (twist_index + 1))
        if opt_geminal.time_reversal:
            dn = np.conj(up[: opt_geminal.num_electron_dn])
        else:
            dn = np.asarray(opt_geminal.orb_data_dn_spin.mo_coefficients)
        opt_geminal = opt_geminal.replace(
            orb_data_up_spin=opt_geminal.orb_data_up_spin.replace(mo_coefficients=up),
            orb_data_dn_spin=opt_geminal.orb_data_dn_spin.replace(mo_coefficients=dn),
        )
        optimized = optimized.replace(
            wavefunction_data=optimized.wavefunction_data.replace(
                jastrow_data=marker_jastrow,
                geminal_data=opt_geminal,
            )
        )
        with h5py.File(twist_checkpoint_path(str(checkpoint_dir), twist_index), "w") as h5:
            meta = h5.create_group("_meta")
            meta.attrs["twist_index"] = twist_index
            meta.attrs["target_steps"] = 10
            meta.attrs["converged"] = bool(converged)
            _save_dataclass_to_hdf5(h5.create_group("hamiltonian_data"), optimized)
        expected_up.append(np.asarray(up))
    save_shared_state(str(checkpoint_dir), optimized)
    return str(source), str(checkpoint_dir), expected_up


class TestWriteTwistOptimizedHamiltonian:
    """``write_twist_optimized_hamiltonian`` dump contract (4 tests).

    This is the routine the twist-averaged optimizer calls at every dump point,
    and it is what makes ``hamiltonian_data_opt_step_N.h5`` a COMPLETE
    combined Hamiltonian instead of the world-rank-0 twist only.  Its output
    has to be a legal ``hamiltonian_h5`` for a fresh VMC / MCMC / LRDMC run, so
    these tests read it back through the very loaders the CLI uses
    (``is_twist_average_input`` / ``load_twist_average_input`` / ``load_twist``)
    rather than poking at raw HDF5 groups.
    """

    @pytest.mark.light
    @requires_h_fixture
    def test_dump_is_a_loadable_combined_input_with_every_optimized_twist(self, tmp_path):
        """The dump round-trips as a combined input: same ``n_twist`` and weights as
        the source, each twist carrying ITS OWN optimized MOs and the canonical
        shared Jastrow.  This is the property that lets a downstream job be
        started from any dumped opt step."""
        from jqmc._twist_average import (
            is_twist_average_input,
            load_twist,
            load_twist_average_input,
        )
        from jqmc._twist_checkpoint import write_twist_optimized_hamiltonian

        source, checkpoint_dir, expected_up = _build_combined_source_and_checkpoints(tmp_path, converged=True)
        output = str(tmp_path / "hamiltonian_data_opt_step_4.h5")

        write_twist_optimized_hamiltonian(source, checkpoint_dir, output)
        assert is_twist_average_input(output)
        meta_out = load_twist_average_input(output)
        meta_in = load_twist_average_input(source)
        assert meta_out.n_twist == meta_in.n_twist == 2
        np.testing.assert_array_equal(meta_out.weights, meta_in.weights)
        for twist_index in range(meta_out.n_twist):
            published = load_twist(output, twist_index)
            np.testing.assert_array_equal(
                published.wavefunction_data.geminal_data.orb_data_up_spin.mo_coefficients,
                expected_up[twist_index],
            )
            assert published.wavefunction_data.jastrow_data.jastrow_two_body_data.jastrow_2b_param == 2.75

    @pytest.mark.light
    @requires_h_fixture
    def test_unconverged_per_twist_checkpoints_still_dump(self, tmp_path):
        """Intermediate dumps are the normal case: the per-twist files of a
        still running optimisation carry ``converged=False`` yet each one is a
        complete snapshot, so the dump must be a valid combined input anyway.
        """
        from jqmc._twist_average import is_twist_average_input, load_twist
        from jqmc._twist_checkpoint import write_twist_optimized_hamiltonian

        source, checkpoint_dir, expected_up = _build_combined_source_and_checkpoints(tmp_path, converged=False)
        output = str(tmp_path / "hamiltonian_data_opt_step_2.h5")

        write_twist_optimized_hamiltonian(source, checkpoint_dir, output)
        assert is_twist_average_input(output)
        np.testing.assert_array_equal(
            load_twist(output, 1).wavefunction_data.geminal_data.orb_data_up_spin.mo_coefficients,
            expected_up[1],
        )

    @pytest.mark.light
    @requires_h_fixture
    def test_missing_per_twist_checkpoint_raises(self, tmp_path):
        """A partial checkpoint directory can never be dumped: silently
        emitting a file whose twist 1 still holds the DFT starting MOs would
        blend un-optimized physics into the BZ average."""
        from jqmc._twist_checkpoint import write_twist_optimized_hamiltonian

        source, checkpoint_dir, _ = _build_combined_source_and_checkpoints(tmp_path, converged=True)
        os.remove(twist_checkpoint_path(checkpoint_dir, 1))
        with pytest.raises(RuntimeError, match="per-twist checkpoints are missing"):
            write_twist_optimized_hamiltonian(source, checkpoint_dir, str(tmp_path / "out.h5"))
        assert not os.path.exists(tmp_path / "out.h5")

    @pytest.mark.light
    @requires_h_fixture
    def test_missing_shared_jastrow_raises(self, tmp_path):
        """Without ``shared_state.h5`` there is no canonical Jastrow: the
        per-twist snapshots may each hold a stale copy, so refuse rather than
        dump an unsynchronised wavefunction."""
        from jqmc._twist_checkpoint import write_twist_optimized_hamiltonian

        source, checkpoint_dir, _ = _build_combined_source_and_checkpoints(tmp_path, converged=True)
        os.remove(shared_state_path(checkpoint_dir))
        with pytest.raises(RuntimeError, match="shared Jastrow is missing"):
            write_twist_optimized_hamiltonian(source, checkpoint_dir, str(tmp_path / "out.h5"))
