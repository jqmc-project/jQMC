"""Real CLI on two-twist complex pGTO + PW and pGTO + B-spline composite inputs.

The existing CLI twist journeys run two twists only for the pGTO-only TREXIO
input; the pGTO + PW and pGTO + B-spline composite kinds use the one-twist
Li_k025 save, and the eight-twist example09 smoke is pGTO-only. Here the
committed eight-twist Li_k_twist QE save is converted with the production
converter to ``kind = "pgto+pw"`` or ``"pgto+bspline"`` and reduced to its
first two twists, so that the per-twist ``composite_child`` override is
exercised for two different twists in one actual MPI CLI run (two ranks, one
twist each), for both MCMC and LRDMC (tau).
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

import hashlib
import shutil

import h5py
import numpy as np
from pathlib import Path

import pytest
import toml

from jqmc._checkpoint import load_hamiltonian_from_checkpoint, load_rank_checkpoint
from jqmc._precision import is_mixed_precision_enabled
from jqmc._setting import GFMC_MIN_BIN_BLOCKS, GFMC_MIN_COLLECT_STEPS, GFMC_MIN_STEPS, GFMC_MIN_WARMUP_STEPS
from jqmc._twist_average import load_twist_average_input
from jqmc.atomic_orbital import AOs_bspline_complex_data, AOs_composite_data, AOs_plane_wave_complex_data
from jqmc.qe_to_jqmc import convert_qe_to_twist_average_input
from jqmc.upf_wrapper import parse_gaussinfo
from tests._test_data_cache import qe_example_files_root
from tests.test_jqmc_kernel.test_integration_cli_twist_dispatch import (
    _MPIRUN,
    _assert_twist_summary_from_checkpoints,
    _run_cli,
)

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]

_QE_SAVE = qe_example_files_root() / "Li_k_twist" / "out" / "li_twist.save"
_UPF = _QE_SAVE / "Li.ccECP.upf"

_N_TWIST = 2
# Per composite kind: the saved per-twist child tag, the rebuilt leaf type and
# its payload attribute (Miller triples, or complex spline control points).
_KINDS = {
    "pgto+pw": ("pw_complex", "g_indices", AOs_plane_wave_complex_data),
    "pgto+bspline": ("bspline_complex", "spline_coefficients", AOs_bspline_complex_data),
}


def _build_two_twist_composite_input(directory, kind):
    """Convert all eight Li_k_twist orbitals to the composite ``kind``, then keep the first two twists.

    The reduction copies ``/shared`` and the ``twist_000`` / ``twist_001``
    groups byte for byte and sets ``n_twist = 2`` and ``bz_weight_sum`` to the
    sum of the two retained raw weights, the values the production writer
    would finalise for those two twists.
    """
    eight = directory / "hamiltonian_eight_twists.h5"
    two = directory / "hamiltonian_two_twists.h5"
    convert_qe_to_twist_average_input(
        qe_save_dir=str(_QE_SAVE),
        output_path=str(eight),
        upf_paths={"Li": str(_UPF)},
        ecp_by_element={"Li": parse_gaussinfo(str(_UPF))},
        kind=kind,
        j1_param=None,
        j2_param=1.0,
        j2_type="pade",
    )
    with h5py.File(eight, "r") as source, h5py.File(two, "w") as target:
        assert int(source["_meta"].attrs["n_twist"]) == 8
        meta = target.create_group("_meta")
        for name, value in source["_meta"].attrs.items():
            meta.attrs[name] = value
        source.copy(source["shared"], target, name="shared")
        twists = target.create_group("twists")
        weights = []
        for index in range(_N_TWIST):
            name = f"twist_{index:03d}"
            source.copy(source["twists"][name], twists, name=name)
            weights.append(float(np.asarray(source["twists"][name]["weight"])))
        meta.attrs["n_twist"] = _N_TWIST
        meta.attrs["bz_weight_sum"] = float(np.sum(weights))
    return two


def _raw_twist_inputs(ham_h5, kind):
    """Per-twist k, composite child payload and MO blocks read directly from the input HDF5."""
    tag, dataset, _leaf = _KINDS[kind]
    rows = []
    with h5py.File(ham_h5, "r") as handle:
        for index in range(_N_TWIST):
            group = handle["twists"][f"twist_{index:03d}"]
            assert group["composite_child"].attrs["kind"] == tag
            rows.append(
                {
                    "k": np.asarray(group["k_point"], dtype=np.float64),
                    "payload": np.asarray(group["composite_child"][dataset]),
                    "mo_up": np.asarray(group["mo_coefficients_up"]),
                    "mo_dn": np.asarray(group["mo_coefficients_dn"]),
                }
            )
    return rows


def _child_payload(leaf, kind, down):
    """The rebuilt k-dependent child mapped back to the saved ``+k`` payload layout, and its twist.

    The down-spin leaf is the time-reversal sibling: twist -k with negated
    Miller triples (plane waves) or conjugated control points (B-spline). The
    inverse map is written out here rather than taken from production.
    """
    twist = np.asarray(leaf.k_point, dtype=np.float64)
    if kind == "pgto+pw":
        payload = np.asarray(leaf._g_indices, dtype=np.int64)
        return (-payload, -twist) if down else (payload, twist)
    payload = np.asarray(leaf.spline_coefficients)
    # Time reversal may share the stored table and conjugate it lazily.
    # Reconstruct its logical payload independently of the production accessor.
    if leaf.conjugate_coefficients:
        payload = np.conj(payload)
    return (np.conj(payload), -twist) if down else (payload, twist)


def _assert_composite_twist_checkpoints(ham_h5, checkpoint_dir, job_type, parameters, kind):
    """Each completed twist carries its own composite k-dependent child, MOs and twist metadata."""
    _tag, _dataset, leaf_type = _KINDS[kind]
    raw = _raw_twist_inputs(ham_h5, kind)
    assert not np.array_equal(raw[0]["k"], raw[1]["k"])
    assert raw[0]["payload"].shape != raw[1]["payload"].shape or not np.array_equal(raw[0]["payload"], raw[1]["payload"])
    for index, expected in enumerate(raw):
        checkpoint = checkpoint_dir / f"twist_{index:03d}.h5"
        geminal = load_hamiltonian_from_checkpoint(str(checkpoint)).wavefunction_data.geminal_data
        for spin, mo_key in (("orb_data_up_spin", "mo_up"), ("orb_data_dn_spin", "mo_dn")):
            orbital = getattr(geminal, spin)
            np.testing.assert_array_equal(np.asarray(orbital.mo_coefficients), expected[mo_key])
            aos = orbital.aos_data
            assert isinstance(aos, AOs_composite_data) and len(aos.children) == 2
            child = aos.children[1]
            assert isinstance(child, leaf_type)
            payload, twist = _child_payload(child, kind, down=spin == "orb_data_dn_spin")
            np.testing.assert_array_equal(twist, expected["k"])
            np.testing.assert_array_equal(payload, expected["payload"].reshape(payload.shape))
            assert aos.num_ao == np.asarray(orbital.mo_coefficients).shape[1]
        with h5py.File(checkpoint, "r") as handle:
            info = handle["_meta"]
            assert info.attrs["twist_index"] == index
            assert info.attrs["num_twists"] == _N_TWIST
            assert bool(info.attrs["converged"])
            assert info.attrs["target_steps"] == parameters[job_type]["num_mcmc_steps"]
            assert info.attrs["normalized_twist_weight"] == 1 / _N_TWIST
            np.testing.assert_array_equal(np.asarray(info.attrs["k_point"], dtype=np.float64), expected["k"])
            assert np.isfinite(info.attrs["E_k"]) and info.attrs["sigma_k"] > 0
        state = load_rank_checkpoint(str(checkpoint), rank=0)
        stored = np.asarray(state["observables"]["e_L"])
        assert stored.shape == (parameters[job_type]["num_mcmc_steps"], 1)
        assert np.isfinite(stored).all()
        # The production sampler's final walker positions (one walker) are stored and finite.
        r_up_carts = np.asarray(state["walker_state"]["latest_r_up_carts"])
        r_dn_carts = np.asarray(state["walker_state"]["latest_r_dn_carts"])
        assert r_up_carts.ndim == r_dn_carts.ndim == 3 and r_up_carts.shape[::2] == r_dn_carts.shape[::2] == (1, 3)
        assert np.isfinite(r_up_carts).all() and np.isfinite(r_dn_carts).all()


def _parameters(job_type, ham_h5, checkpoint_dir):
    mode = "mixed" if is_mixed_precision_enabled() else "full"
    control = {
        "job_type": job_type,
        "mcmc_seed": 34467,
        "number_of_walkers": 1,
        "max_time": 86400,
        "restart": False,
        "restart_chk": str(checkpoint_dir),
        "hamiltonian_h5": str(ham_h5),
        "verbosity": "low",
    }
    if job_type == "mcmc":
        section = {
            "num_mcmc_steps": 12,
            "num_mcmc_warmup_steps": 2,
            "num_mcmc_bin_blocks": 2,
            "num_mcmc_per_measurement": 1,
            "Dt": 2.0,
            "epsilon_AS": 0.0,
            "atomic_force": False,
            "compute_stress": False,
            "parameter_derivatives": False,
        }
    else:
        section = {
            "num_mcmc_steps": GFMC_MIN_STEPS,
            "tau": 0.10,
            "alat": 0.30,
            "non_local_move": "tmove",
            "num_gfmc_warmup_steps": GFMC_MIN_WARMUP_STEPS,
            "num_gfmc_bin_blocks": GFMC_MIN_BIN_BLOCKS,
            "num_gfmc_collect_steps": GFMC_MIN_COLLECT_STEPS,
            "atomic_force": False,
            "compute_stress": False,
        }
    return {"control": control, job_type: section, "precision": {"mode": mode}}


def _snapshot(checkpoint_dir):
    return {
        path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
        for path in sorted(checkpoint_dir.glob("twist_*.h5"))
    }


@pytest.mark.parametrize("job_type", ["mcmc", "lrdmc-tau"])
@pytest.mark.parametrize("kind", ["pgto+pw", "pgto+bspline"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_cli_two_twist_composite_runs_and_resumes(tmp_path, monkeypatch, kind, job_type):
    """The CLI runs a two-twist composite input and keeps each twist's own k-dependent child.

    Convert the committed eight-twist Li_k_twist QE save (ccECP, Pade J2 = 1,
    no J1) with ``kind = "pgto+pw"`` or ``"pgto+bspline"`` (converter defaults)
    and keep twists 0 and 1, k = (1/4, 1/4,
    1/4) and (1/4, 1/4, -1/4), raw weight 1/4 each. Launch the actual CLI under
    two MPI ranks with one walker, seed 34467 and the requested precision mode
    in the TOML. MCMC uses 12 measurements (warmup 2, bins 2, one move each);
    LRDMC-tau uses tau 0.10, alat 0.30, tmove and the central minimum
    step/warmup/bin/collect counts. Electron positions come from the production
    sampler without test-imposed cond(G) screening.

    In each precision separately, require the two-twist dispatch banner and
    precision line, both twist checkpoints and the shared state; for each
    twist, the up composite child and the time-reversal image of the down
    child equal the input ``composite_child`` payload (Miller triples or
    complex spline control points) and twist read directly with h5py, the MOs equal
    the input datasets, and the twist index, count, normalized weight 1/2, k,
    target steps, convergence flag and finite stored energies are present.
    A second CLI invocation with restart enabled must leave both checkpoints
    byte- and timestamp-identical, and both displayed E_TA summaries must equal
    independent arithmetic on the saved per-twist values. This is a wiring and
    metadata contract, not an energy accuracy or statistical convergence test.
    """
    assert _MPIRUN is not None, "the two-twist composite CLI contract needs an MPI launcher"
    assert (_QE_SAVE / "data-file-schema.xml").is_file() and _UPF.is_file()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "JQMC_RS_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("OMPI_MCA_rmaps_base_oversubscribe", "1")
    monkeypatch.setenv("OMPI_MCA_hwloc_base_binding_policy", "none")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
    ham_h5 = _build_two_twist_composite_input(tmp_path, kind)
    meta = load_twist_average_input(str(ham_h5))
    assert meta.n_twist == _N_TWIST
    np.testing.assert_array_equal(np.asarray(meta.k_points), [[0.25, 0.25, 0.25], [0.25, 0.25, -0.25]])
    np.testing.assert_array_equal(meta.weights, [0.25, 0.25])
    assert meta.bz_weight_sum == 0.5

    checkpoint_dir = tmp_path / "restart_twists"
    parameters = _parameters(job_type, ham_h5, checkpoint_dir)
    toml_path = tmp_path / "input.toml"
    toml_path.write_text(toml.dumps(parameters))
    fresh = _run_cli(toml_path, tmp_path, n_ranks=_N_TWIST)
    fresh_output = fresh.stdout + fresh.stderr
    assert fresh.returncode == 0, fresh_output
    assert f"Twist-averaged dispatch: {_N_TWIST} twists" in fresh_output
    mode = parameters["precision"]["mode"]
    assert f"**section:precision**\n  mode={mode}\n" in fresh_output
    assert sorted(path.name for path in checkpoint_dir.glob("twist_*.h5")) == ["twist_000.h5", "twist_001.h5"]
    assert (checkpoint_dir / "shared_state.h5").is_file()
    _assert_composite_twist_checkpoints(ham_h5, checkpoint_dir, job_type, parameters, kind)

    before = _snapshot(checkpoint_dir)
    parameters["control"]["restart"] = True
    toml_path.write_text(toml.dumps(parameters))
    resumed = _run_cli(toml_path, tmp_path, n_ranks=_N_TWIST)
    resumed_output = resumed.stdout + resumed.stderr
    assert resumed.returncode == 0, resumed_output
    assert f"**section:precision**\n  mode={mode}\n" in resumed_output
    assert _snapshot(checkpoint_dir) == before, "completed composite twists were rewritten on restart"
    _assert_twist_summary_from_checkpoints([fresh_output, resumed_output], checkpoint_dir, _N_TWIST)
    shutil.rmtree(checkpoint_dir)
