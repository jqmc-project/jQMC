"""Restart-inheritance policy visibility + fail-fast guard.

Light tests for the classification tables (``*_restart_policy``), the pure
frozen-mismatch guard, and the ``[frozen/mutable on restart]`` TOML labels. The
authoritative drift test that cross-checks the frozen set against the real
``MCMC.save_to_hdf5`` ``driver_config`` lives in the kernel suite
(``test_jqmc_mcmc.py::test_restart_policy_matches_checkpoint``) because it must
build and serialise a real MCMC.
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

from types import SimpleNamespace

import pytest

from jqmc import jqmc_cli
from jqmc._restart_policy import TOML_TO_CFG, assert_restart_frozen_consistency, is_frozen, restored_frozen_values
from jqmc.jqmc_miscs import cli_parameters
from jqmc_workflow._input_generator import generate_input_toml

pytestmark = [pytest.mark.with_chdir]


@pytest.mark.light
def test_all_params_classified():
    """Every driver-section key carries a frozen/mutable restart policy."""
    for section in ("control", "mcmc", "vmc", "lrdmc-bra", "lrdmc-tau", "coulomb"):
        keys = set(cli_parameters[section])
        policy = cli_parameters[f"{section}_restart_policy"]
        assert set(policy) == keys, f"{section}: unclassified={keys - set(policy)}, stray={set(policy) - keys}"
        assert all(v in ("frozen", "mutable") for v in policy.values())


# Frozen keys that are required inputs (no default in cli_parameters) and a
# valid value to probe the guard with.
_REQUIRED_FROZEN_PROBES = {("lrdmc-bra", "num_mcmc_per_measurement"): 40}


@pytest.mark.light
@pytest.mark.parametrize("section", ["mcmc", "vmc", "lrdmc-bra", "lrdmc-tau"])
def test_frozen_keys_are_mapped(section):
    """Every declared scalar frozen field rejects conflicts, independent of its name.

    The policy table is the expected contract; the guard's skip set is not an
    oracle. New frozen keys automatically enter this test and require a real
    mapping plus scalar equality semantics. Embedded Hamiltonians are not scalar
    settings and retain their explicit separate sentinel.
    """
    checked = 0
    for owner_section in ("control", section, "coulomb"):
        policy = cli_parameters[f"{owner_section}_restart_policy"]
        for key, kind in policy.items():
            if kind != "frozen":
                continue
            assert (owner_section, key) in TOML_TO_CFG, f"frozen {owner_section}.{key} missing from TOML_TO_CFG"
            cfg_key = TOML_TO_CFG[(owner_section, key)]
            if key == "hamiltonian_h5":
                assert cfg_key == "<hamiltonian_data>"
                continue
            value = cli_parameters[owner_section][key]
            if value is None:
                # A required frozen key has no default; probe it with a valid value.
                value = _REQUIRED_FROZEN_PROBES[(owner_section, key)]
            assert isinstance(value, (bool, int, float, str)), (owner_section, key, value)
            changed = (
                not value if isinstance(value, bool) else value + 1 if isinstance(value, (int, float)) else "different-" + value
            )
            for saved, explicit in ((value, changed), (changed, value)):
                checkpoint = {cfg_key: saved}
                with pytest.raises(ValueError, match=rf"\[{owner_section}\] {key}.*frozen on restart"):
                    assert_restart_frozen_consistency(section, {owner_section: {key: explicit}}, checkpoint)
                assert_restart_frozen_consistency(section, {owner_section: {key: saved}}, checkpoint)
                assert_restart_frozen_consistency(section, {owner_section: {}}, checkpoint)
            checked += 1
        for key, kind in policy.items():
            if kind == "mutable":
                # Even an unlike value is outside the frozen-key guard; the
                # ordinary CLI owns validation of mutable parameter types.
                assert_restart_frozen_consistency(section, {owner_section: {key: object()}}, {})
    assert checked > 0


@pytest.mark.light
def test_restart_raises_on_frozen_mismatch():
    """Explicit frozen conflict -> ValueError; matching / omitted / mutable -> ok."""
    ckpt = {
        "Dt": 2.0,
        "num_mcmc_per_measurement": 40,
        "mcmc_seed": 34456,
        "num_walkers": 4,
        "comput_stress": False,
    }

    # explicit conflict on a frozen [mcmc] key -> raise
    with pytest.raises(ValueError, match="frozen on restart"):
        assert_restart_frozen_consistency("mcmc", {"mcmc": {"Dt": 1.5}}, ckpt)

    # explicit conflict on a frozen [control] key -> raise
    with pytest.raises(ValueError, match="frozen on restart"):
        assert_restart_frozen_consistency("mcmc", {"control": {"mcmc_seed": 999}}, ckpt)

    # stress storage is a separate frozen capability from atomic force
    with pytest.raises(ValueError, match="frozen on restart"):
        assert_restart_frozen_consistency("mcmc", {"mcmc": {"compute_stress": True}}, ckpt)

    # Force storage is also frozen: reject both explicit conflicts, preserve
    # matching values, and do not interpret an omitted switch as False.
    for saved in (False, True):
        force_ckpt = {**ckpt, "comput_position_deriv": saved}
        with pytest.raises(ValueError, match=r"\[mcmc\] atomic_force.*frozen on restart"):
            assert_restart_frozen_consistency("mcmc", {"mcmc": {"atomic_force": not saved}}, force_ckpt)
        assert_restart_frozen_consistency("mcmc", {"mcmc": {"atomic_force": saved}}, force_ckpt)
        assert_restart_frozen_consistency("mcmc", {"mcmc": {}}, force_ckpt)

    # explicit but matching frozen value -> ok
    assert_restart_frozen_consistency("mcmc", {"mcmc": {"Dt": 2.0}}, ckpt)
    assert_restart_frozen_consistency("mcmc", {"mcmc": {"compute_stress": False}}, ckpt)

    # frozen key omitted from the TOML -> ok (no false positive from defaults)
    assert_restart_frozen_consistency("mcmc", {"mcmc": {"num_mcmc_steps": 999}}, ckpt)

    # a differing *mutable* key is not guarded -> ok
    assert_restart_frozen_consistency("mcmc", {"mcmc": {"num_mcmc_warmup_steps": 123}}, ckpt)

    # Omitted frozen keys are logged with the checkpoint value (#316); explicit and mutable keys are not.
    ckpt = {
        "Dt": 1.5,
        "num_mcmc_per_measurement": 20,
        "mcmc_seed": 7,
        "num_walkers": 8,
        "epsilon_AS": 0.0,
        "comput_position_deriv": True,
        "mpc": True,
        "mpc_cutoff": 11.0,
    }
    restored = restored_frozen_values("mcmc", {"control": {"mcmc_seed": 7}, "mcmc": {"num_mcmc_steps": 10}}, ckpt)
    by_key = {(section, key): value for section, key, value in restored}
    assert by_key[("control", "number_of_walkers")] == 8
    assert by_key[("mcmc", "Dt")] == 1.5
    assert by_key[("mcmc", "num_mcmc_per_measurement")] == 20
    assert ("control", "mcmc_seed") not in by_key  # given explicitly
    assert ("mcmc", "num_mcmc_steps") not in by_key  # mutable
    assert by_key[("mcmc", "atomic_force")] is True
    assert by_key[("coulomb", "mpc")] is True
    assert by_key[("coulomb", "mpc_cutoff")] == 11.0

    # The two omitted switches independently take their own restored values.
    for force, stress in ((True, True), (True, False), (False, True)):
        driver = SimpleNamespace(comput_position_deriv=force, comput_stress=stress)
        assert jqmc_cli._restart_force_stress_switches("mcmc", {"mcmc": {}}, False, False, driver) == (force, stress)
    assert by_key[("control", "hamiltonian_h5")] is None  # restored, but not a scalar
    assert {(section, key) for section, key, _ in restored} == {
        (section, key)
        for section in ("control", "mcmc", "coulomb")
        for key in cli_parameters[section]
        if is_frozen(section, key) and (section, key) != ("control", "mcmc_seed")
    }


@pytest.mark.light
def test_lrdmc_and_vmc_restart_raise_on_frozen_mismatch():
    """The frozen-key guard covers VMC-optimize and both LRDMC drivers.

    A restart whose continuation TOML changes the checkpoint physics (alat,
    non_local_move, tau, collect steps) is refused instead of silently restoring
    the checkpoint values and ignoring the TOML -- a silent wrong-energy path.
    """
    gfmc_ckpt = {
        "alat": 0.30,
        "tau": 0.10,
        "non_local_move": "tmove",
        "num_gfmc_collect_steps": 0,
        "num_mcmc_per_measurement": 40,
        "E_scf": 0.0,
        "use_swct": False,
        "epsilon_PW": 0.0,
        "comput_stress": False,
    }
    with pytest.raises(ValueError, match="frozen on restart"):
        assert_restart_frozen_consistency("lrdmc-bra", {"lrdmc-bra": {"alat": 0.20}}, gfmc_ckpt)
    with pytest.raises(ValueError, match="frozen on restart"):
        assert_restart_frozen_consistency("lrdmc-tau", {"lrdmc-tau": {"tau": 0.05}}, gfmc_ckpt)
    with pytest.raises(ValueError, match="frozen on restart"):
        assert_restart_frozen_consistency("lrdmc-bra", {"lrdmc-bra": {"non_local_move": "dltmove"}}, gfmc_ckpt)
    with pytest.raises(ValueError, match="frozen on restart"):
        assert_restart_frozen_consistency("lrdmc-tau", {"lrdmc-tau": {"num_gfmc_collect_steps": 5}}, gfmc_ckpt)

    for section in ("lrdmc-bra", "lrdmc-tau"):
        for saved in (False, True):
            force_ckpt = {**gfmc_ckpt, "comput_position_deriv": saved}
            with pytest.raises(ValueError, match=rf"\[{section}\] atomic_force.*frozen on restart"):
                assert_restart_frozen_consistency(section, {section: {"atomic_force": not saved}}, force_ckpt)
            assert_restart_frozen_consistency(section, {section: {"atomic_force": saved}}, force_ckpt)
            assert_restart_frozen_consistency(section, {section: {}}, force_ckpt)

    mcmc_ckpt = {"Dt": 2.0, "epsilon_AS": 0.0, "num_mcmc_per_measurement": 40}
    with pytest.raises(ValueError, match="frozen on restart"):
        assert_restart_frozen_consistency("vmc", {"vmc": {"Dt": 1.0}}, mcmc_ckpt)

    # matching / omitted / mutable stay silent for the new sections too
    assert_restart_frozen_consistency("lrdmc-bra", {"lrdmc-bra": {"alat": 0.30}}, gfmc_ckpt)
    assert_restart_frozen_consistency("lrdmc-tau", {"lrdmc-tau": {"num_gfmc_bin_blocks": 99}}, gfmc_ckpt)
    assert_restart_frozen_consistency("vmc", {"vmc": {"opt_geminal": True}}, mcmc_ckpt)


@pytest.mark.light
def test_toml_comments_show_restart_label(tmp_path):
    """Generated commented TOML shows the [frozen/mutable on restart] contract."""
    path = tmp_path / "mcmc.toml"
    generate_input_toml(
        "mcmc",
        overrides={"mcmc": {"num_mcmc_steps": 100}, "coulomb": {"mpc": True, "mpc_cutoff": 11.0}},
        filename=str(path),
        with_comments=True,
    )
    lines = path.read_text().splitlines()

    def comment_above(key):
        for i, ln in enumerate(lines):
            if ln.strip().startswith(f"{key} =") and i > 0:
                return lines[i - 1]
        return ""

    assert "[frozen on restart]" in comment_above("Dt")
    assert "[mutable on restart]" in comment_above("num_mcmc_steps")
    assert "[frozen on restart]" in comment_above("mpc")
    assert "[frozen on restart]" in comment_above("mpc_cutoff")
