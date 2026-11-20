"""Restart-inheritance guard for the CLI.

On restart the MCMC driver rebuilds itself from the checkpoint
(``MCMC.load_from_hdf5``), so a subset of the TOML keys are restored from the
checkpoint and any conflicting value in the continuation TOML is silently
ignored. ``jqmc.jqmc_miscs`` classifies every key as ``"frozen"`` (restored
from the checkpoint) or ``"mutable"`` (re-read every run) in the
``*_restart_policy`` tables; this module turns a *frozen* conflict into a loud
``ValueError`` instead of a silent no-op.

``TOML_TO_CFG`` maps each frozen ``(section, key)`` to the ``driver_config``
key that ``MCMC.save_to_hdf5`` writes. A drift test cross-checks this map
against the real ``driver_config`` so a newly-added checkpoint field cannot
silently escape classification.
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

from typing import Any

# (toml_section, toml_key) -> checkpoint driver_config key.
# Sentinels in _GUARD_SKIP_CFG are frozen for *labelling* purposes but are not
# scalar-comparable, so the value guard skips them:
#   * "<hamiltonian_data>"    -- the wavefunction is embedded, not a scalar.
# Atomic-force and stress storage are separate, scalar-comparable capabilities.
TOML_TO_CFG: dict[tuple[str, str], str] = {
    ("control", "mcmc_seed"): "mcmc_seed",
    ("control", "number_of_walkers"): "num_walkers",
    ("control", "hamiltonian_h5"): "<hamiltonian_data>",
    ("mcmc", "num_mcmc_per_measurement"): "num_mcmc_per_measurement",
    ("mcmc", "Dt"): "Dt",
    ("mcmc", "epsilon_AS"): "epsilon_AS",
    ("mcmc", "use_swct"): "use_swct",
    ("mcmc", "parameter_derivatives"): "comput_log_WF_param_deriv",
    ("mcmc", "atomic_force"): "comput_position_deriv",
    ("mcmc", "compute_stress"): "comput_stress",
    ("coulomb", "mpc"): "mpc",
    ("coulomb", "mpc_cutoff"): "mpc_cutoff",
    # VMC-optimize shares the MCMC driver checkpoint.
    ("vmc", "num_mcmc_per_measurement"): "num_mcmc_per_measurement",
    ("vmc", "Dt"): "Dt",
    ("vmc", "epsilon_AS"): "epsilon_AS",
    # GFMC_n (lrdmc-bra): keys restored by GFMC.load_from_hdf5 (jqmc_gfmc.py).
    ("lrdmc-bra", "num_mcmc_per_measurement"): "num_mcmc_per_measurement",
    ("lrdmc-bra", "alat"): "alat",
    ("lrdmc-bra", "non_local_move"): "non_local_move",
    ("lrdmc-bra", "num_gfmc_collect_steps"): "num_gfmc_collect_steps",
    ("lrdmc-bra", "E_scf"): "E_scf",
    ("lrdmc-bra", "atomic_force"): "comput_position_deriv",
    ("lrdmc-bra", "compute_stress"): "comput_stress",
    ("lrdmc-bra", "use_swct"): "use_swct",
    ("lrdmc-bra", "epsilon_PW"): "epsilon_PW",
    # GFMC_t (lrdmc-tau).
    ("lrdmc-tau", "tau"): "tau",
    ("lrdmc-tau", "alat"): "alat",
    ("lrdmc-tau", "non_local_move"): "non_local_move",
    ("lrdmc-tau", "num_gfmc_collect_steps"): "num_gfmc_collect_steps",
    ("lrdmc-tau", "atomic_force"): "comput_position_deriv",
    ("lrdmc-tau", "compute_stress"): "comput_stress",
    ("lrdmc-tau", "use_swct"): "use_swct",
    ("lrdmc-tau", "epsilon_PW"): "epsilon_PW",
}

# cfg targets that are frozen-for-labelling but NOT scalar-comparable, so the
# value guard skips them (they remain in TOML_TO_CFG for the drift test).
_GUARD_SKIP_CFG: frozenset[str] = frozenset({"<hamiltonian_data>"})


def assert_restart_frozen_consistency(
    section: str,
    dict_toml: dict[str, Any],
    checkpoint_driver_config: dict[str, Any],
) -> None:
    """Raise ``ValueError`` on an explicit frozen-key conflict at restart.

    Only keys the user *explicitly* wrote in ``dict_toml`` are checked. Omitted
    keys are filled with ``jqmc_miscs`` defaults downstream, so comparing the
    merged value would raise on keys the user never touched -- hence the raw
    ``dict_toml`` (not the merged parameter table) is used here.

    Args:
        section: job-type section, e.g. ``"mcmc"``.
        dict_toml: the raw parsed user TOML (NOT merged with defaults).
        checkpoint_driver_config: driver_config read from the restart
            checkpoint (see
            :func:`jqmc._checkpoint.load_driver_config_from_checkpoint`).

    Raises:
        ValueError: a frozen key was explicitly set in the TOML to a value that
            differs from the checkpoint's, or is absent from the checkpoint.
    """
    from .jqmc_miscs import cli_parameters

    for toml_section in ("control", section, "coulomb"):
        policy = cli_parameters.get(f"{toml_section}_restart_policy", {})
        user_section = dict_toml.get(toml_section, {}) or {}
        for key, pol in policy.items():
            if pol != "frozen" or key not in user_section:
                continue
            cfg_key = TOML_TO_CFG.get((toml_section, key))
            if cfg_key is None or cfg_key in _GUARD_SKIP_CFG:
                continue
            toml_val = user_section[key]
            if cfg_key not in checkpoint_driver_config:
                # The writer always stores every frozen key: an absent one is a
                # damaged or legacy checkpoint, never a silent pass (#807).
                raise ValueError(
                    f"[{toml_section}] {key} = {toml_val!r} is set in the restart TOML, but the checkpoint lacks "
                    f"the frozen key {cfg_key!r} (damaged or legacy checkpoint); it cannot be continued."
                )
            ckpt_val = checkpoint_driver_config[cfg_key]
            if toml_val != ckpt_val:
                raise ValueError(
                    f"[{toml_section}] {key} = {toml_val!r} in the restart TOML conflicts "
                    f"with the checkpoint value {ckpt_val!r} (this key is 'frozen on restart': "
                    f"the driver restores it from the checkpoint and ignores the TOML). "
                    f"Remove {key} from the continuation TOML, or start a fresh (non-restart) run."
                )


def is_frozen(toml_section: str, key: str) -> bool:
    """Whether ``[toml_section] key`` is restored from the checkpoint on restart."""
    from .jqmc_miscs import cli_parameters

    return cli_parameters.get(f"{toml_section}_restart_policy", {}).get(key) == "frozen"


def restored_frozen_values(
    section: str,
    dict_toml: dict[str, Any],
    checkpoint_driver_config: dict[str, Any],
) -> list[tuple[str, str, Any]]:
    """``(toml_section, key, value)`` of the frozen keys omitted from the restart TOML.

    These keys take the checkpoint value, not the ``jqmc_miscs`` default, so
    the CLI logs them with that value. ``value`` is ``None`` for keys that are
    not scalars in the checkpoint (e.g. the embedded Hamiltonian).

    Args:
        section: job-type section, e.g. ``"mcmc"``.
        dict_toml: the raw parsed user TOML (NOT merged with defaults).
        checkpoint_driver_config: driver_config read from the restart checkpoint.
    """
    from .jqmc_miscs import cli_parameters

    restored = []
    for toml_section in ("control", section, "coulomb"):
        policy = cli_parameters.get(f"{toml_section}_restart_policy", {})
        user_section = dict_toml.get(toml_section, {}) or {}
        for key, pol in policy.items():
            if pol != "frozen" or key in user_section:
                continue
            cfg_key = TOML_TO_CFG.get((toml_section, key))
            if cfg_key is None:
                continue
            value = None if cfg_key == "<hamiltonian_data>" else checkpoint_driver_config.get(cfg_key)
            restored.append((toml_section, key, value))
    return restored
