"""Tests for the jqmc.checkpoint infrastructure."""

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

import errno
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.with_chdir]
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jqmc._checkpoint import (
    _convert_attr,
    get_checkpoint_num_ranks,
    load_checkpoint_meta,
    load_observables_from_checkpoint,
    load_rank_checkpoint,
    save_rank_checkpoint,
    validate_restored_walker_finiteness_collectively,
    validate_restored_walker_shapes,
)

# ---------------------------------------------------------------------------
# Per-rank save / load tests
# ---------------------------------------------------------------------------


class TestRankCheckpoint:
    """Tests for save_rank_checkpoint / load_rank_checkpoint."""

    @staticmethod
    def _make_sample_data():
        """Create a minimal set of per-rank data dicts."""
        return dict(
            driver_type="MCMC",
            driver_config={
                "mcmc_seed": 12345,
                "num_walkers": 4,
                "Dt": 2.0,
                "epsilon_AS": 1e-1,
                "num_mcmc_per_measurement": 16,
            },
            rng_state={
                "jax_PRNG_key_list": np.arange(8, dtype=np.uint32).reshape(4, 2),
                "mpi_seed": 12345,
            },
            walker_state={
                "latest_r_up_carts": np.random.randn(4, 3, 3),
                "latest_r_dn_carts": np.random.randn(4, 2, 3),
            },
            observables={
                "e_L": np.random.randn(100, 4),
                "e_L2": np.random.randn(100, 4),
                "w_L": np.ones((100, 4)),
                "param_grads": {
                    "j1_param": np.random.randn(100, 4, 5),
                },
            },
        )

    @pytest.mark.light
    def test_save_load_roundtrip(self, tmp_path):
        """Data survives save -> merge -> load."""
        data = self._make_sample_data()
        rank_file = str(tmp_path / "._restart_rank0.h5")
        save_rank_checkpoint(rank_file, **data)

        # Check file was created
        assert os.path.exists(rank_file)

        # Now merge into a single checkpoint (simulate rank 0)

        # We need a Hamiltonian_data for merging -- use a minimal mock
        merged = str(tmp_path / "restart.h5")
        _merge_single_rank(rank_file, merged, tmp_path)

        # Load back
        loaded = load_rank_checkpoint(merged, rank=0)

        # Verify driver_config
        assert loaded["driver_config"]["mcmc_seed"] == 12345
        assert loaded["driver_config"]["num_walkers"] == 4
        assert loaded["driver_config"]["Dt"] == 2.0

        # Verify rng_state
        np.testing.assert_array_equal(
            loaded["rng_state"]["jax_PRNG_key_list"],
            data["rng_state"]["jax_PRNG_key_list"],
        )
        assert loaded["rng_state"]["mpi_seed"] == 12345

        # Verify walker_state
        np.testing.assert_array_equal(
            loaded["walker_state"]["latest_r_up_carts"],
            data["walker_state"]["latest_r_up_carts"],
        )

        # Verify observables
        np.testing.assert_array_equal(loaded["observables"]["e_L"], data["observables"]["e_L"])
        np.testing.assert_array_equal(
            loaded["observables"]["param_grads"]["j1_param"],
            data["observables"]["param_grads"]["j1_param"],
        )

    @pytest.mark.light
    def test_with_optimizer_state(self, tmp_path):
        """Optimizer state (method + hyperparams) roundtrips."""
        data = self._make_sample_data()
        data["optimizer_state"] = {
            "method": "sr",
            "hyperparameters": {"delta": 1e-2, "epsilon": 1e-3},
        }
        rank_file = str(tmp_path / "._restart_rank0.h5")
        save_rank_checkpoint(rank_file, **data)

        merged = str(tmp_path / "restart.h5")
        _merge_single_rank(rank_file, merged, tmp_path)

        loaded = load_rank_checkpoint(merged, rank=0)
        opt = loaded["optimizer_state"]
        assert opt is not None
        assert opt["method"] == "sr"
        assert opt["hyperparameters"]["delta"] == 1e-2

    @pytest.mark.light
    def test_multi_rank_merge(self, tmp_path):
        """Two ranks can be merged and read independently."""
        for rank in range(2):
            data = self._make_sample_data()
            data["driver_config"]["mcmc_seed"] = 1000 + rank
            rank_file = str(tmp_path / f"._restart_rank{rank}.h5")
            save_rank_checkpoint(rank_file, **data)

        merged = str(tmp_path / "restart.h5")
        _merge_multi_rank(2, merged, tmp_path)

        assert get_checkpoint_num_ranks(merged) == 2

        for rank in range(2):
            loaded = load_rank_checkpoint(merged, rank=rank)
            assert loaded["driver_config"]["mcmc_seed"] == 1000 + rank

    @pytest.mark.light
    @pytest.mark.parametrize("failure", ["hamiltonian-write", "rank-copy", "publish"])
    def test_failed_merge_preserves_checkpoint_and_rank_inputs(self, tmp_path, monkeypatch, failure):
        """An interrupted two-rank merge keeps the previous checkpoint and allows retry."""
        import h5py

        from jqmc import _checkpoint
        from tests.test_jqmc_kernel.test_integration_checkpoint_mcmc import _build_hamiltonian

        output = tmp_path / "restart.h5"
        rank_files = [tmp_path / f"._restart_rank{rank}.h5" for rank in range(2)]
        merge_options = dict(
            mpi_size=2,
            driver_type="MCMC",
            hamiltonian_data=_build_hamiltonian("H2_ecp_ccpvdz_cart.h5", "none"),
            tmp_pattern=str(tmp_path / "._restart_rank{rank}.h5"),
        )
        data = self._make_sample_data()
        for rank, path in enumerate(rank_files):
            data["driver_config"]["mcmc_seed"] = 1000 + rank
            save_rank_checkpoint(str(path), **data)
        _checkpoint.merge_rank_checkpoints(str(output), **merge_options)
        previous = output.read_bytes()

        for rank, path in enumerate(rank_files):
            data["driver_config"]["mcmc_seed"] = 2000 + rank
            save_rank_checkpoint(str(path), **data)
        rank_inputs = [path.read_bytes() for path in rank_files]

        def fail_write(*args, **kwargs):
            raise OSError(errno.ENOSPC, f"injected {failure}")

        copy = h5py.Group.copy

        def copy_until_last_rank(group, source, destination):
            if destination == "rank_1":
                fail_write()
            return copy(group, source, destination)

        with monkeypatch.context() as injected:
            if failure == "hamiltonian-write":
                injected.setattr(_checkpoint, "_save_dataclass_to_hdf5", fail_write)
            elif failure == "rank-copy":
                injected.setattr(h5py.Group, "copy", copy_until_last_rank)
            else:
                injected.setattr(_checkpoint.os, "replace", fail_write)
            with pytest.raises(OSError, match=f"injected {failure}"):
                _checkpoint.merge_rank_checkpoints(str(output), **merge_options)

        assert output.read_bytes() == previous
        assert [path.read_bytes() for path in rank_files] == rank_inputs
        assert set(tmp_path.iterdir()) == {output, *rank_files}
        assert get_checkpoint_num_ranks(str(output)) == 2
        for rank in range(2):
            assert load_rank_checkpoint(str(output), rank)["driver_config"]["mcmc_seed"] == 1000 + rank

        _checkpoint.merge_rank_checkpoints(str(output), **merge_options)
        assert set(tmp_path.iterdir()) == {output}
        for rank in range(2):
            assert load_rank_checkpoint(str(output), rank)["driver_config"]["mcmc_seed"] == 2000 + rank


# ---------------------------------------------------------------------------
# Checkpoint metadata and required payload tests
# ---------------------------------------------------------------------------


class TestCheckpointMetadata:
    """Tests for current metadata and required rank payloads."""

    @pytest.mark.light
    def test_meta_fields(self, tmp_path):
        """Merged checkpoint has correct _meta attributes."""
        data = TestRankCheckpoint._make_sample_data()
        rank_file = str(tmp_path / "._restart_rank0.h5")
        save_rank_checkpoint(rank_file, **data)

        merged = str(tmp_path / "restart.h5")
        from jqmc._checkpoint import merge_rank_checkpoints
        from tests.test_jqmc_kernel.test_integration_checkpoint_mcmc import _build_hamiltonian

        merge_rank_checkpoints(
            merged,
            mpi_size=1,
            driver_type="MCMC",
            hamiltonian_data=_build_hamiltonian("H2_ecp_ccpvdz_cart.h5", "none"),
            tmp_pattern=str(tmp_path / "._restart_rank{rank}.h5"),
        )

        meta = load_checkpoint_meta(merged)
        assert "format_version" not in meta
        assert "pbc_schema_version" not in meta
        assert meta["driver_type"] == "MCMC"
        assert meta["mpi_size"] == 1
        assert "timestamp" in meta
        assert "jqmc_version" in meta

    @pytest.mark.light
    def test_missing_rank_payload_is_rejected(self, tmp_path):
        """Metadata alone cannot substitute for the required rank data."""
        import h5py

        filepath = str(tmp_path / "missing_rank.h5")
        with h5py.File(filepath, "w") as f:
            f.create_group("_meta").attrs["mpi_size"] = 1

        with pytest.raises(KeyError, match="rank_0"):
            load_rank_checkpoint(filepath, rank=0)


# ---------------------------------------------------------------------------
# Observable-only loading tests
# ---------------------------------------------------------------------------


class TestLoadObservables:
    """Tests for load_observables_from_checkpoint."""

    @pytest.mark.light
    def test_load_selected_keys(self, tmp_path):
        """Only requested observable keys are loaded."""
        data = TestRankCheckpoint._make_sample_data()
        rank_file = str(tmp_path / "._restart_rank0.h5")
        save_rank_checkpoint(rank_file, **data)

        merged = str(tmp_path / "restart.h5")
        _merge_single_rank(rank_file, merged, tmp_path)

        obs_list = load_observables_from_checkpoint(merged, keys=["e_L", "w_L"])
        assert len(obs_list) == 1
        assert "e_L" in obs_list[0]
        assert "w_L" in obs_list[0]
        assert "e_L2" not in obs_list[0]
        for name in ("e_L", "w_L"):
            np.testing.assert_array_equal(obs_list[0][name], data["observables"][name])

    @pytest.mark.light
    def test_load_all_keys(self, tmp_path):
        """All observables loaded when keys=None."""
        data = TestRankCheckpoint._make_sample_data()
        rank_file = str(tmp_path / "._restart_rank0.h5")
        save_rank_checkpoint(rank_file, **data)

        merged = str(tmp_path / "restart.h5")
        _merge_single_rank(rank_file, merged, tmp_path)

        obs_list = load_observables_from_checkpoint(merged)
        assert len(obs_list) == 1
        assert "e_L" in obs_list[0]
        assert "e_L2" in obs_list[0]
        assert "w_L" in obs_list[0]

        for name, values in data["observables"].items():
            if isinstance(values, dict):
                assert set(obs_list[0][name]) == set(values)
                for parameter, block in values.items():
                    np.testing.assert_array_equal(obs_list[0][name][parameter], block)
            else:
                np.testing.assert_array_equal(obs_list[0][name], values)


# ---------------------------------------------------------------------------
# Helpers for _convert_attr
# ---------------------------------------------------------------------------


class TestConvertAttr:
    @pytest.mark.light
    def test_bytes_to_str(self):
        """Bytes to str."""
        assert _convert_attr(b"hello") == "hello"

    @pytest.mark.light
    def test_numpy_int(self):
        """Numpy int."""
        result = _convert_attr(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    @pytest.mark.light
    def test_numpy_float(self):
        """Numpy float."""
        result = _convert_attr(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    @pytest.mark.light
    def test_numpy_bool(self):
        """Numpy bool."""
        result = _convert_attr(np.bool_(True))
        assert result is True

    @pytest.mark.light
    def test_none(self):
        """None."""
        assert _convert_attr(None) is None


# ---------------------------------------------------------------------------
# Test utilities -- minimal merge helpers that don't require Hamiltonian_data
# ---------------------------------------------------------------------------


def _merge_single_rank(rank_file: str, merged_path: str, tmp_path) -> None:
    """Merge a single rank file with a dummy hamiltonian_data group."""
    import h5py

    if os.path.exists(merged_path):
        os.remove(merged_path)
    with h5py.File(merged_path, "w") as out:
        meta = out.create_group("_meta")
        meta.attrs["driver_type"] = "MCMC"
        meta.attrs["mpi_size"] = 1
        meta.attrs["jqmc_version"] = "test"
        meta.attrs["timestamp"] = "2026-03-05T00:00:00+00:00"

        # dummy hamiltonian_data group
        out.create_group("hamiltonian_data")

        with h5py.File(rank_file, "r") as tmp:
            out.copy(tmp, "rank_0")

    # cleanup tmp
    if os.path.exists(rank_file):
        os.remove(rank_file)


def _merge_multi_rank(num_ranks: int, merged_path: str, tmp_path) -> None:
    """Merge multiple rank files with a dummy hamiltonian_data group."""
    import h5py

    if os.path.exists(merged_path):
        os.remove(merged_path)
    with h5py.File(merged_path, "w") as out:
        meta = out.create_group("_meta")
        meta.attrs["driver_type"] = "MCMC"
        meta.attrs["mpi_size"] = num_ranks
        meta.attrs["jqmc_version"] = "test"
        meta.attrs["timestamp"] = "2026-03-05T00:00:00+00:00"

        out.create_group("hamiltonian_data")

        for rank in range(num_ranks):
            rank_file = str(tmp_path / f"._restart_rank{rank}.h5")
            with h5py.File(rank_file, "r") as tmp:
                out.copy(tmp, f"rank_{rank}")
            os.remove(rank_file)


class TestRestoredWalkerShapes:
    """Restart rejects walker blocks that do not match the restored Hamiltonian."""

    @pytest.mark.light
    def test_matching_shapes_pass(self):
        """Walker blocks of shape (num_walkers, electrons, 3), and an empty block for a spin without electrons, are accepted."""
        validate_restored_walker_shapes(
            np.zeros((4, 3, 3)), np.zeros((4, 2, 3)), num_walkers=4, num_electron_up=3, num_electron_dn=2, filepath="r.h5"
        )
        # No down electrons: any empty array is accepted.
        validate_restored_walker_shapes(
            np.zeros((4, 1, 3)), np.zeros((4, 0, 3)), num_walkers=4, num_electron_up=1, num_electron_dn=0, filepath="r.h5"
        )

    @pytest.mark.light
    @pytest.mark.parametrize(
        ("up_shape", "dn_shape"),
        [((4, 2, 3), (4, 2, 3)), ((3, 3, 3), (4, 2, 3)), ((4, 3, 3), (4, 2, 2)), ((4, 3, 3), (4, 3, 3))],
        ids=["up-electrons", "walkers", "coords", "dn-electrons"],
    )
    def test_mismatched_shapes_raise(self, up_shape, dn_shape):
        """A wrong electron count, walker count or coordinate width for either spin is rejected."""
        with pytest.raises(ValueError, match="restored walker array"):
            validate_restored_walker_shapes(
                np.zeros(up_shape), np.zeros(dn_shape), num_walkers=4, num_electron_up=3, num_electron_dn=2, filepath="r.h5"
            )

    @pytest.mark.light
    @pytest.mark.parametrize(
        "spin,bad_value",
        [("up", np.nan), ("up", np.inf), ("down", -np.inf)],
    )
    def test_nonfinite_coordinates_raise(self, spin, bad_value):
        """A restart cannot pass NaN/Inf coordinates into the first sampling kernel."""
        r_up = np.zeros((2, 2, 3))
        r_dn = np.zeros((2, 1, 3))
        (r_up if spin == "up" else r_dn)[1, 0, 2] = bad_value
        from mpi4py import MPI

        validate_restored_walker_shapes(
            r_up,
            r_dn,
            num_walkers=2,
            num_electron_up=2,
            num_electron_dn=1,
            filepath="restart.h5",
        )
        with pytest.raises(FloatingPointError, match=r"(?s)non-finite.*walker=1.*electron=0.*component=2"):
            validate_restored_walker_finiteness_collectively(
                MPI.COMM_SELF,
                driver="MCMC",
                filepath="restart.h5",
                r_up_carts=r_up,
                r_dn_carts=r_dn,
            )


@pytest.mark.light
@pytest.mark.parametrize("driver_kind", ["MCMC", "GFMC_t", "GFMC_n"])
def test_driver_config_roundtrip_uses_constructor_values(driver_kind, tmp_path):
    """All declared scalar frozen fields survive real driver serialization.

    This existing Light-selected file exercises real writers/loaders without
    sampling. References are explicit non-default constructor inputs, not the
    saved dictionary. New frozen fields must be supplied here too; a newly
    skipped restore cannot hide behind a constructor default or fake data dict.
    """
    from jqmc._restart_policy import TOML_TO_CFG
    from jqmc.jqmc_miscs import cli_parameters
    from tests.test_jqmc_kernel.test_checkpoint_nondefault_config import (
        _DRIVER_TYPES,
        _config_attribute,
        _nondefault_constructor_settings,
        _saved_driver,
    )
    from tests.test_jqmc_kernel.test_integration_checkpoint_mcmc import _build_hamiltonian

    settings = _nondefault_constructor_settings(driver_kind)
    section = {"MCMC": "mcmc", "GFMC_t": "lrdmc-tau", "GFMC_n": "lrdmc-bra"}[driver_kind]
    required = {
        TOML_TO_CFG[(owner, key)]
        for owner in ("control", section)
        for key, policy in cli_parameters[owner + "_restart_policy"].items()
        if policy == "frozen" and key != "hamiltonian_h5"
    }
    assert required <= settings.keys(), f"explicit non-default constructor witness missing: {required - settings.keys()}"
    for owner in ("control", section):
        for key, policy in cli_parameters[owner + "_restart_policy"].items():
            if policy == "frozen" and key != "hamiltonian_h5":
                assert settings[TOML_TO_CFG[(owner, key)]] != cli_parameters[owner][key], (owner, key)
    ham = _build_hamiltonian("H2_ecp_ccpvdz_cart.h5", "1b+2b")
    original = _DRIVER_TYPES[driver_kind](hamiltonian_data=ham, **settings)
    restored = _saved_driver(original, ham, tmp_path / "roundtrip")
    for name, expected in settings.items():
        assert _config_attribute(original, name) == expected, (driver_kind, name, "constructor")
        assert _config_attribute(restored, name) == expected, (driver_kind, name, "restored")


_RANK_1_RESTART_RECORDS = {
    "agree": ({"agreed": {"mcmc_counter": 8, "non_local_move": "dltmove"}}, []),
    "disagree": (
        {"agreed": {"mcmc_counter": 8, "non_local_move": "tmove"}},
        ["non_local_move differs across ranks: rank_0='dltmove', rank_1='tmove'"],
    ),
    "missing": (
        {"missing": ["driver_config/non_local_move"], "agreed": {"mcmc_counter": 8}},
        ["rank_1/driver_config/non_local_move: required restart key is missing"],
    ),
}


@pytest.mark.light
@pytest.mark.parametrize("case", sorted(_RANK_1_RESTART_RECORDS))
def test_restart_state_problems_agree_disagree_missing(case):
    """The gathered per-rank restart records give one verdict, the same on every rank.

    Two synthetic records stand for what two ranks restored from their own
    groups. Agreement passes; a differing value names both ranks and values; a
    missing key names the rank and the key and is not also reported as a
    disagreement. (The two-rank CLI tests exercise the collective wrapper.)
    """
    from jqmc._checkpoint import restart_state_problems

    rank_1, expected = _RANK_1_RESTART_RECORDS[case]
    rank_0 = {"rank": 0, "missing": [], "errors": [], "agreed": {"mcmc_counter": 8, "non_local_move": "dltmove"}}
    record_1 = {"rank": 1, "missing": [], "errors": [], **rank_1}
    assert restart_state_problems([rank_0, record_1]) == expected, case
