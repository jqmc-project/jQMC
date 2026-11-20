"""Pure PW/B-spline J3 must reach native GFMC measurement and chunk derivatives.

Same-input native/XLA parity runs in both precisions. Full precision additionally
checks the nuclear force and strain against rebuilt-input fourth-order differences
of the production continuum measurement energy, with a fixed mesh rotation. Mixed
never treats an fp32 finite-difference stencil as a correctness oracle.
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
import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    # Direct children keep their private CWD, but must import this checkout's tests.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jqmc_rs
from jqmc._precision import get_tolerance_xla_rust_consistency, is_mixed_precision_enabled
from jqmc._setting import ATOL_FD_VS_ANALYTIC_TIGHT, RTOL_FD_VS_ANALYTIC_TIGHT, ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_ZERO
from tests.canonical_wavefunction_fixtures import build_canonical_hamiltonian
from tests.test_jqmc_kernel.test_gfmc_measurement_fd import _fourth_order, _H, _measurement_energy, _rebuilt
from tests.test_rs_kernel.test_gfmc_complex_force_stress import _GRID, _NN, _rot, _rt_np, _WEIGHTS
from jqmc.derivatives import gfmc

pytestmark = [pytest.mark.heavy, pytest.mark.with_chdir]
_PATTERNS = ["WF-PATTERN-008", "WF-PATTERN-010"]
_ZONES = (
    "ao_eval",
    "ao_grad_lap",
    "mo_eval",
    "mo_grad",
    "mo_lap",
    "det_eval",
    "det_grad_lap",
    "jastrow_eval",
    "jastrow_grad_lap",
    "wf_kinetic",
    "ewald",
    "coulomb",
)


def _pure_leaf_hamiltonian(pattern):
    ham = build_canonical_hamiltonian(pattern)
    jd = ham.wavefunction_data.jastrow_data
    j3 = jd.jastrow_three_body_data
    from jqmc.atomic_orbital import AOs_bspline_complex_data, AOs_plane_wave_complex_data

    leaf = getattr(j3.orb_data, "aos_data", j3.orb_data)
    assert isinstance(leaf, (AOs_plane_wave_complex_data, AOs_bspline_complex_data)), type(leaf)
    assert np.any(np.asarray(j3.j_matrix) != 0), "J3 must be active"
    assert jd.jastrow_one_body_data is not None or jd.jastrow_dtn_data is not None, "nuclear Jastrow cross term must be active"
    return ham


def _rebuilt_fd(ham, r_up, r_dn, RT, nlm, quantity, h):
    energy = _measurement_energy("t", nlm)
    if quantity == "force":
        out = np.zeros_like(np.asarray(ham.structure_data.positions))
        for atom, axis in np.ndindex(out.shape):
            out[atom, axis] = _fourth_order(
                lambda x: float(energy(_rebuilt(ham, np.eye(3), (atom, axis, x)), r_up, r_dn, RT)), h
            )
    else:
        out = np.zeros((3, 3))
        for axis, other in np.ndindex(out.shape):

            def strained(x):
                F = np.eye(3)
                F[axis, other] += x
                rebuilt = _rebuilt(ham, F)
                jd = rebuilt.wavefunction_data.jastrow_data
                if jd.jastrow_dtn_data is not None:
                    # Independent finite deformation for MATH-JAS-COMOVING-STRAIN-001:
                    # the stored independent coefficients/maps stay fixed; the production
                    # forward energy reads their comoving constraint transport at this scale.
                    original_scale = ham.wavefunction_data.jastrow_data.jastrow_dtn_data.cutoff_scale
                    dtn = jd.jastrow_dtn_data.replace(cutoff_scale=original_scale * np.cbrt(np.linalg.det(F)))
                    rebuilt = rebuilt.replace(
                        wavefunction_data=rebuilt.wavefunction_data.replace(jastrow_data=jd.replace(jastrow_dtn_data=dtn))
                    )
                return float(energy(rebuilt, r_up @ F.T, r_dn @ F.T, RT))

            out[axis, other] = _fourth_order(strained, h)
    return out


@pytest.mark.parametrize("pattern", _PATTERNS)
@pytest.mark.parametrize("nlm", ["tmove", "dltmove"])
@pytest.mark.parametrize("quantity", ["force", "stress"])
def test_pure_leaf_j3_native_measurement_force_stress(pattern, nlm, quantity):
    """Retain every pure-leaf gradient and strain channel, with an independent full FD anchor."""
    ham = _pure_leaf_hamiltonian(pattern)
    gd = ham.wavefunction_data.geminal_data
    rng = np.random.default_rng(5)
    r_up = jnp.asarray(rng.normal(scale=0.9, size=(int(gd.num_electron_up), 3)))
    r_dn = jnp.asarray(rng.normal(scale=0.9, size=(int(gd.num_electron_dn), 3)))
    RT = _rot(7)
    wf = ham.wavefunction_data
    args = (
        wf.geminal_data,
        wf.jastrow_data,
        ham.coulomb_potential_data,
        np.asarray(r_up),
        np.asarray(r_dn),
        0.3,
        _rt_np(RT),
        _WEIGHTS,
        _GRID,
        _NN,
        nlm == "dltmove",
    )
    if quantity == "force":
        native = tuple(np.asarray(x) for x in jqmc_rs._gfmc_grad_e_L_ecp_pbc_complex(*args))
        debug = jqmc_rs._gfmc_grad_e_L_ecp_pbc_complex(*args, debug=True)
        oracles = (gfmc.grad_e_L_t_wrt_positions, gfmc.grad_e_L_n_wrt_positions)
    else:
        native = (np.asarray(jqmc_rs._gfmc_stress_ecp_pbc_complex(*args)),)
        debug = (jqmc_rs._gfmc_stress_ecp_pbc_complex(*args, debug=True),)
        oracles = (gfmc.grad_e_L_t_wrt_strain, gfmc.grad_e_L_n_wrt_strain)
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    shapes = ((len(ham.structure_data.positions), 3), r_up.shape, r_dn.shape) if quantity == "force" else ((3, 3),)
    for got, ref, shape in zip(native, debug, shapes, strict=True):
        assert got.shape == np.asarray(ref).shape == shape
        assert np.all(np.isfinite(got)) and np.all(np.isfinite(ref))
        np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol)
    for oracle in oracles:
        xla = oracle(ham, r_up, r_dn, RT, 0.3, non_local_move=nlm)
        if quantity == "stress":
            xla = (xla,)
        for got, ref in zip(native, xla, strict=True):
            np.testing.assert_allclose(got, np.real(np.asarray(ref)), atol=atol, rtol=rtol)
    if not is_mixed_precision_enabled():
        fd = _rebuilt_fd(ham, r_up, r_dn, RT, nlm, quantity, _H)
        fd_2h = _rebuilt_fd(ham, r_up, r_dn, RT, nlm, quantity, 2 * _H)
        assert np.all(np.isfinite(fd)) and np.any(np.abs(fd) > ATOL_FD_VS_ANALYTIC_TIGHT)
        np.testing.assert_allclose(fd, fd_2h, atol=ATOL_FD_VS_ANALYTIC_TIGHT, rtol=RTOL_FD_VS_ANALYTIC_TIGHT)
        np.testing.assert_allclose(native[0], fd, atol=ATOL_FD_VS_ANALYTIC_TIGHT, rtol=RTOL_FD_VS_ANALYTIC_TIGHT)
    jax.clear_caches()


_DRIVER_FIELDS = ("_stored_w_L", "_stored_e_L", "_stored_e_L2", "_stored_force_HF", "_stored_stress_HF")
_COORD_FIELDS = ("_latest_r_up_carts", "_latest_r_dn_carts")


def _driver_bootstrap_probe(mode, output):
    """Report genuine child imports without starting a physical GFMC trajectory."""
    import jqmc
    import tests
    from jqmc._precision import configure, precision_policy_snapshot

    jax.config.update("jax_enable_x64", True)
    configure(mode)
    jqmc_rs.configure_native_precision(precision_policy_snapshot())
    metadata = {
        "mode": mode,
        "requested_policy": precision_policy_snapshot(),
        "standalone_dispatch": jqmc_rs.native_precision_support(),
        "script_source": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tests_package_source": str(Path(tests.__file__).resolve()),
        "jqmc_source": str(Path(jqmc.__file__).resolve()),
        "native_binary": str(Path(jqmc_rs.__file__).resolve()),
        "native_binary_sha256": hashlib.sha256(Path(jqmc_rs.__file__).read_bytes()).hexdigest(),
        "native_source_fnv64": jqmc_rs.__source_fnv64__,
        "cwd": str(Path.cwd()),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "receipt_authority": os.environ.get("GDD_RECEIPT_AUTHORITY"),
        "physical_driver_executed": False,
    }
    Path(output).write_text(json.dumps(metadata, indent=2) + "\n")


def _driver_backend(pattern, driver, backend, mode, output):
    """One original-issue trajectory per backend in a fresh process, with no statistical accessor."""
    from jqmc._precision import configure, precision_policy_snapshot
    from jqmc.jqmc_gfmc import GFMC_n, GFMC_t
    from tests.test_rs_kernel._route_spy import assert_arg_reached, assert_native_ran, install_ctx_spy

    jax.config.update("jax_enable_x64", True)
    configure(mode)
    jqmc_rs.configure_native_precision(precision_policy_snapshot())
    jqmc_rs.configure_threads(int(os.environ.get("JQMC_RS_NUM_THREADS", "4")))
    ham = _pure_leaf_hamiltonian(pattern)
    with pytest.MonkeyPatch.context() as patch:
        if backend == "xla":
            patch.setenv("JQMC_FLAG_RUST_ON_CPU", "0")
        else:
            patch.setenv("JQMC_FLAG_RUST_ON_CPU", "1")
        context_policies = []
        raw_build = jqmc_rs.build_obs_ctx
        requested_policy = precision_policy_snapshot()

        def capture_policy(*args, **kwargs):
            ctx = raw_build(*args, **kwargs)
            actual_policy = ctx.precision_policy()
            assert actual_policy == requested_policy
            context_policies.append({"requested": actual_policy, "dispatch": ctx.precision_dispatch_support()})
            return ctx

        patch.setattr(jqmc_rs, "build_obs_ctx", capture_policy)
        spy = install_ctx_spy(patch)
        common = dict(
            hamiltonian_data=ham,
            num_walkers=4,
            num_gfmc_collect_steps=1,
            mcmc_seed=34467,
            alat=0.3,
            random_discretized_mesh=True,
            comput_position_deriv=True,
            comput_stress=True,
            use_swct=False,
        )
        run = GFMC_t(**common, tau=0.1) if driver == "t" else GFMC_n(**common, num_mcmc_per_measurement=2)
        run.run(num_mcmc_steps=6)
        values = {name: np.asarray(getattr(run, name)) for name in (*_DRIVER_FIELDS, *_COORD_FIELDS, "_jax_PRNG_key_list")}
        values["native_source_fnv64"] = np.asarray(jqmc_rs.__source_fnv64__)
        np.savez(output, **values)
        import jqmc
        from mpi4py import MPI

        root = Path(__file__).resolve().parents[2]
        assert Path(jqmc.__file__).resolve().is_relative_to(root)
        assert Path(jqmc_rs.__file__).resolve().is_relative_to(root)
        metadata = {
            "pattern": pattern,
            "driver": driver,
            "backend": backend,
            "mode": mode,
            "requested_policy": requested_policy,
            "standalone_dispatch": jqmc_rs.native_precision_support(),
            "production_context_policies": context_policies,
            "native_observable_method_calls": dict(Counter(call.method for call in spy.calls)),
            "actual_projection_fold_flags": [
                {"want_force": bool(c.value("want_force")), "want_stress": bool(c.value("want_stress"))}
                for c in spy.calls_to(f"run_gfmc_projection_{driver}_complex_batch")
            ],
            "diagnostic_context_getter_calls": 2 * len(context_policies),
            "adapter_method_calls": dict(Counter(call.method for call in spy.adapter_calls)),
            "jqmc_source": jqmc.__file__,
            "native_binary": jqmc_rs.__file__,
            "native_binary_sha256": hashlib.sha256(Path(jqmc_rs.__file__).read_bytes()).hexdigest(),
            "native_source_fnv64": jqmc_rs.__source_fnv64__,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "actual_mpi_size": MPI.COMM_WORLD.Get_size(),
            "actual_mpi_rank": MPI.COMM_WORLD.Get_rank(),
            "num_walkers": 4,
            "num_gfmc_collect_steps": run.num_gfmc_collect_steps,
            "num_mcmc_steps": 6,
            "mcmc_seed": 34467,
            "alat": 0.3,
            "random_discretized_mesh": True,
            "tau": 0.1 if driver == "t" else None,
            "num_mcmc_per_measurement": 2 if driver == "n" else None,
            "audit_authority": 0,
        }
        Path(output).with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        if backend == "rust":
            assert_native_ran(spy, f"run_gfmc_projection_{driver}_complex_batch", at_least=6, context=pattern)
            assert_arg_reached(spy, f"run_gfmc_projection_{driver}_complex_batch", "want_force", True, context=pattern)
            assert_arg_reached(spy, f"run_gfmc_projection_{driver}_complex_batch", "want_stress", True, context=pattern)
        else:
            assert not spy.calls, "XLA reference executed a native observable context"
        assert run.num_gfmc_collect_steps == 1


@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated", seed=34467)
@pytest.mark.parametrize("pattern", _PATTERNS)
@pytest.mark.parametrize("driver", ["t", "n"])
def test_pure_leaf_j3_gfmc_driver_force_stress(pattern, driver, tmp_path):
    """Both public GFMC drivers retain J3 on the unchanged original six-step trajectories."""
    ham = _pure_leaf_hamiltonian(pattern)
    mode = "mixed" if is_mixed_precision_enabled() else "full"
    results = []
    for backend in ("xla", "rust"):
        output = tmp_path / f"{driver}-{pattern}-{backend}.npz"
        env = {key: value for key, value in os.environ.items() if not key.startswith("GDD_")}
        env.pop("PYTHONPATH", None)
        env["GDD_RECEIPT_AUTHORITY"] = "0"
        env["JAX_COMPILATION_CACHE_DIR"] = str(tmp_path / f"jax-cache-{backend}")
        with (tmp_path / f"{backend}.log").open("w") as log:
            child = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), pattern, driver, backend, mode, str(output)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        assert child.returncode == 0, (tmp_path / f"{backend}.log").read_text()
        metadata = json.loads(output.with_suffix(".json").read_text())
        from jqmc._precision import precision_policy_snapshot

        assert metadata["mode"] == mode and metadata["pattern"] == pattern and metadata["driver"] == driver
        assert metadata["requested_policy"] == precision_policy_snapshot()
        assert metadata["standalone_dispatch"] == jqmc_rs.native_precision_support()
        assert metadata["backend"] == backend and metadata["num_gfmc_collect_steps"] == 1
        assert metadata["test_source_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with np.load(output) as data:
            assert str(data["native_source_fnv64"]) == jqmc_rs.__source_fnv64__
            results.append({key: data[key] for key in data.files})
    xla, native = results
    atol, rtol = get_tolerance_xla_rust_consistency(_ZONES, level="medium")
    for name in _DRIVER_FIELDS:
        got, ref = native[name], xla[name]
        assert got.shape == ref.shape and got.size and np.all(np.isfinite(got)) and np.all(np.isfinite(ref))
        np.testing.assert_allclose(got, ref, atol=atol, rtol=rtol, err_msg=f"{pattern} GFMC_{driver} {name}")
    gd = ham.wavefunction_data.geminal_data
    for name, count in zip(_COORD_FIELDS, (int(gd.num_electron_up), int(gd.num_electron_dn)), strict=True):
        assert native[name].shape == xla[name].shape == (4, count, 3)
        np.testing.assert_allclose(native[name], xla[name], atol=ATOL_PYTHON_FP64_LOOSE, rtol=RTOL_PYTHON_FP64_ZERO)
    np.testing.assert_array_equal(native["_jax_PRNG_key_list"], xla["_jax_PRNG_key_list"])


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--bootstrap-probe":
        _driver_bootstrap_probe(sys.argv[2], sys.argv[3])
    else:
        _driver_backend(*sys.argv[1:])
