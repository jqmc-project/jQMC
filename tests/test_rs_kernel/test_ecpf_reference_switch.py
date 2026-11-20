"""Same-input ECP cache/reference comparisons with isolated native profiling."""

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

import multiprocessing
import os
import re
import tempfile
import traceback

import jax
import numpy as np
import pytest

from jqmc._precision import get_tolerance_min, precision_policy_snapshot
from jqmc.coulomb_potential import icosahedron_sym_mesh_Nv12
from tests.test_jqmc_kernel._generated_matrix_base import _build_ham_for_cell
from tests.test_rs_kernel.test_gfmc_forces import _cfg as _random_molecular_positions
from tests.test_rs_kernel.test_gfmc_forces import _pbc_cfg as _random_periodic_positions
from tests.test_rs_kernel.test_gfmc_forces import _rand_j3
from tests.test_rs_kernel.test_vmc_stress import _pbc_geminv

pytestmark = [pytest.mark.heavy, pytest.mark.without_chdir]


def _profiled_call(function):
    """Capture native phase-call counts, never timing as a correctness bound."""
    import jqmc_rs

    with tempfile.TemporaryFile() as capture:
        previous = os.dup(2)
        try:
            os.dup2(capture.fileno(), 2)
            jqmc_rs._prof_dump("clear")
            capture.seek(0)
            capture.truncate()
            value = function()
            jqmc_rs._prof_dump("observed")
        finally:
            os.dup2(previous, 2)
            os.close(previous)
        capture.seek(0)
        records = capture.read().decode()
    counts = {name: int(count) for name, count in re.findall(r"^\s+(\S+)\s+[\d.]+ ms\s+\((\d+) calls,", records, re.M)}
    return value, counts


def _native_pair_worker(connection, ham, ru, rd, inverse, operation, dltmove, policy):
    """Use one native policy and identical objects on both sides in a fresh process."""
    try:
        import jqmc_rs

        jqmc_rs.configure_native_precision(policy)
        jqmc_rs.configure_threads(2)
        jqmc_rs._prof_enable()
        wf = ham.wavefunction_data
        weights = np.asarray(icosahedron_sym_mesh_Nv12.weights)
        points = np.asarray(icosahedron_sym_mesh_Nv12.grid_points)

        def evaluate():
            if operation == "vmc-stress":
                value = jqmc_rs.vmc_grad_e_L_eps_pbc(
                    wf.geminal_data, wf.jastrow_data, ham.coulomb_potential_data, ru, rd, weights, points, inverse
                )
                return {"stress": np.asarray(value)}
            function = jqmc_rs._gfmc_grad_e_L_ecp_pbc if operation == "periodic-force" else jqmc_rs._gfmc_grad_e_L_ecp
            values = function(
                wf.geminal_data,
                wf.jastrow_data,
                ham.coulomb_potential_data,
                ru,
                rd,
                0.30,
                np.eye(3),
                weights,
                points,
                1,
                dltmove,
            )
            return {name: np.asarray(value) for name, value in zip(("R", "up", "down"), values, strict=True)}

        os.environ.pop("JQMC_RS_ECPF_SLOW", None)
        fast, fast_counts = _profiled_call(evaluate)
        os.environ["JQMC_RS_ECPF_SLOW"] = "1"
        reference, reference_counts = _profiled_call(evaluate)
        connection.send(("ok", fast, reference, fast_counts, reference_counts))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _execute_pair(ham, ru, rd, inverse, operation, dltmove):
    """Keep the native profiler's global switch out of the pytest worker."""
    ham = jax.tree_util.tree_map(lambda leaf: np.asarray(leaf) if isinstance(leaf, jax.Array) else leaf, ham)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_native_pair_worker,
        args=(child, ham, ru, rd, inverse, operation, dltmove, precision_policy_snapshot()),
    )
    process.start()
    child.close()
    try:
        assert parent.poll(180), "native ECP comparison worker timed out"
        message = parent.recv()
        process.join(30)
        assert process.exitcode == 0, "native ECP comparison worker did not exit normally"
        assert message[0] == "ok", message
        return message[1:]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(30)
        parent.close()


def _switch_values(operation, trexio_file, with_j3, dltmove, seed):
    """Nonzero ECP force/stress contributions agree across actual cache/reference routes.

    Committed water (spherical molecular GTO, 4up/4down), Li atom (Cartesian
    molecular GTO, 1up/0down), and Gamma lithium (spherical pGTO, 1up/1down)
    retain their SD/MO and nonzero nonlocal ECP. Pade J1+J2 optionally adds
    a symmetric nonzero J3 at seed 7. Electron positions use NumPy PCG64
    seed 0 near nuclei, without cond(G) screening. The 12-point ECP mesh
    is unrotated, alat is 0.30 and the nearest-nucleus count is one.

    Both GFMC move rules exercise molecular/periodic force; the stress
    switch applies to VMC, not the GFMC stress leaf. The same native
    precision policy and input objects feed both evaluations. Existing
    same-mode strict zone tolerances compare every nonempty output.
    Empty Li down-spin has only a shape check. Profiler CALL COUNTS prove
    cached candidates execute only on the fast side; timing is not a
    verdict. A child process confines profiling state, not the input or
    numerical acceptance. This is route parity, not an independent
    physical oracle; the separate XLA AD tests remain necessary.
    """
    ham = _build_ham_for_cell(trexio_file, True, {"j3": "gto"} if with_j3 else None, "SD")
    if with_j3:
        ham = ham.replace(wavefunction_data=_rand_j3(ham.wavefunction_data))
    jd = ham.wavefunction_data.jastrow_data
    jd = jd.replace(
        jastrow_one_body_data=jd.jastrow_one_body_data.replace(jastrow_1b_type="pade"),
        jastrow_two_body_data=jd.jastrow_two_body_data.replace(jastrow_2b_type="pade"),
    )
    ham = ham.replace(wavefunction_data=ham.wavefunction_data.replace(jastrow_data=jd))
    coefficients = np.asarray(ham.coulomb_potential_data._coefficients_non_local_part)
    assert coefficients.size and np.any(coefficients != 0), "nonlocal ECP must be active"
    periodic = bool(ham.structure_data.pbc_flag)
    if periodic:
        ru, rd = _random_periodic_positions(ham, seed=seed)
    else:
        ru, rd = _random_molecular_positions(ham, seed=seed)
    inverse = _pbc_geminv(ham, ru, rd, False) if operation == "vmc-stress" else None
    fast, reference, fast_counts, reference_counts = _execute_pair(ham, ru, rd, inverse, operation, dltmove)
    expected_phase = {
        "molecular-force": "ecpf.jas_fast",
        "periodic-force": "ecpf.jas_fast_pbc",
        "vmc-stress": "eps.jas_fast",
    }[operation]
    assert fast_counts.get(expected_phase, 0) > 0, "fast candidates did not execute"
    assert reference_counts.get(expected_phase, 0) == 0, "reference incorrectly used fast candidates"
    fields = ("stress",) if operation == "vmc-stress" else ("R", "up", "down")
    assert set(fast) == set(reference) == set(fields), "complete physical output fields required"
    for field in fields:
        actual, expected = fast[field], reference[field]
        if field == "down" and rd.shape == (0, 3):
            assert actual.shape == expected.shape == (0, 3)
            continue
        assert actual.size and actual.shape == expected.shape
        assert np.isfinite(actual).all() and np.isfinite(expected).all()
        assert np.any(expected != 0), field
    actual = np.concatenate([fast[field].ravel() for field in fields])
    expected = np.concatenate([reference[field].ravel() for field in fields])
    size = 9 if operation == "vmc-stress" else 3 * (len(ham.structure_data.positions) + len(ru) + len(rd))
    assert actual.shape == expected.shape == (size,), "complete physical output vector required"
    return actual, expected


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize(
    "trexio_file,with_j3,dltmove",
    [
        ("water_ccecp_ccpvqz.h5", False, False),
        ("water_ccecp_ccpvqz.h5", False, True),
        ("water_ccecp_ccpvqz.h5", True, False),
        ("water_ccecp_ccpvqz.h5", True, True),
        ("Li_ecp_ccpvdz_cart.h5", True, False),
        ("Li_ecp_ccpvdz_cart.h5", True, True),
    ],
)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_molecular_ecpf_fast_matches_recomputed(trexio_file, with_j3, dltmove, seed):
    """Molecular ECP nuclear/up/down gradients agree across distinct cache/reference routes.

    Committed water has spherical GTO/MO SD and 4up/4down; Li has Cartesian
    GTO/MO SD and 1up/0down. Nonzero nonlocal ECP, Pade J1+J2 and optional
    symmetric J3 (seed 7) are required. PCG64 seed 0 fixes one nuclear-near
    configuration without cond(G) filtering. Both move rules use alat 0.30,
    the unrotated 12-point mesh and one nearest nucleus. Native profiling
    in a child proves the switch changes the candidate route. The entire
    physical output vector uses the existing same-mode strict policy;
    empty Li down-spin retains a shape check, not empty numerical evidence.
    This does not replace the independent XLA AD oracle.
    """
    actual, expected = _switch_values("molecular-force", trexio_file, with_j3, dltmove, seed)
    atol, rtol = get_tolerance_min(
        ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap"), "strict"
    )
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("with_j3", [False, True])
@pytest.mark.parametrize("dltmove", [False, True])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_periodic_ecpf_fast_matches_recomputed(with_j3, dltmove, seed):
    """Gamma ECP nuclear/up/down gradients agree across distinct cache/reference routes.

    Committed lithium.h5 supplies spherical pGTO/MO SD, 1up/1down and
    nonzero nonlocal ECP. Pade J1+J2 optionally includes symmetric J3 at
    seed 7. PCG64 seed 0 supplies one configuration at 0.4-Bohr Gaussian
    offsets from nuclei, without cond(G) screening. Both move rules use
    alat 0.30, the unrotated 12-point mesh and one nearest nucleus.
    Fast candidate calls must disappear under the reference switch;
    every output component uses the existing same-mode strict policy.
    Profiling is confined to a child. Independent XLA AD tests remain.
    """
    actual, expected = _switch_values("periodic-force", "lithium.h5", with_j3, dltmove, seed)
    atol, rtol = get_tolerance_min(
        ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap", "ewald"), "strict"
    )
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("with_j3", [False, True])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_vmc_ecp_stress_fast_matches_recomputed(with_j3, seed):
    """All nine VMC ECP stress components agree across distinct cache/reference routes.

    Committed Gamma lithium.h5 has spherical pGTO/MO SD, 1up/1down and
    nonzero nonlocal ECP. Pade J1+J2 optionally includes symmetric J3
    (seed 7). PCG64 seed 0 supplies one 0.4-Bohr nuclear-near configuration
    without cond(G) filtering. Both sides share the same base inverse,
    native precision policy and unrotated 12-point mesh. Fast stress
    candidate calls must disappear under the switch. Existing same-mode
    strict tolerances compare all nine entries, not a tensor-scale bound.
    Profiling stays in a child. This is VMC, not GFMC stress, and does not
    replace independent XLA AD or stress physics tests.
    """
    actual, expected = _switch_values("vmc-stress", "lithium.h5", with_j3, False, seed)
    atol, rtol = get_tolerance_min(
        ("ao_eval", "ao_grad_lap", "det_eval", "wf_kinetic", "coulomb", "jastrow_eval", "jastrow_grad_lap", "ewald"), "strict"
    )
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)


def _install_corruption(monkeypatch, operation, fault):
    """Corrupt only returned values or route witnesses after a real native calculation."""
    import sys

    module = sys.modules[__name__]
    original = _execute_pair

    def corrupted(*args, **kwargs):
        fast, reference, fast_counts, reference_counts = original(*args, **kwargs)
        if fault == "value":
            field = "stress" if operation == "vmc-stress" else "R"
            fast[field] = reference[field] + 2 * (1 + np.abs(reference[field]))
        elif fault == "reference-route":
            reference_counts = dict(fast_counts)
        else:
            field = "stress" if operation == "vmc-stress" else "down"
            del fast[field]
            del reference[field]
        return fast, reference, fast_counts, reference_counts

    monkeypatch.setattr(module, "_execute_pair", corrupted)
    return {
        "value": "Not equal to tolerance",
        "reference-route": "reference incorrectly used fast candidates",
        "missing-field": "complete physical output fields required",
    }[fault]


@pytest.mark.parametrize("trexio_file", ["water_ccecp_ccpvqz.h5"])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("fault", ["value", "reference-route", "missing-field"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_molecular_ecpf_verdict_rejects_corruption(monkeypatch, trexio_file, seed, fault):
    """After a passing water ECP baseline, reject wrong values, route witnesses or missing spin output.

    The molecular force fixture has Pade J1+J2+J3, tmove and seed 0,
    with no cond(G) filtering. Both numerical and structural failures are
    checked explicitly, never an unrelated child-process exception.
    """
    test_molecular_ecpf_fast_matches_recomputed(trexio_file, True, False, seed)
    message = _install_corruption(monkeypatch, "molecular-force", fault)
    with pytest.raises(AssertionError, match=message):
        test_molecular_ecpf_fast_matches_recomputed(trexio_file, True, False, seed)


@pytest.mark.parametrize("trexio_file", ["lithium.h5"])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("fault", ["value", "reference-route", "missing-field"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_periodic_ecpf_verdict_rejects_corruption(monkeypatch, trexio_file, seed, fault):
    """After a passing Gamma Li force baseline, reject wrong values, routes or missing spin output.

    The periodic force fixture has Pade J1+J2+J3, tmove and seed 0,
    with no cond(G) filtering. A passing baseline precedes each violation.
    """
    assert trexio_file == "lithium.h5"
    test_periodic_ecpf_fast_matches_recomputed(True, False, seed)
    message = _install_corruption(monkeypatch, "periodic-force", fault)
    with pytest.raises(AssertionError, match=message):
        test_periodic_ecpf_fast_matches_recomputed(True, False, seed)


@pytest.mark.parametrize("trexio_file", ["lithium.h5"])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("fault", ["value", "reference-route", "missing-field"])
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_vmc_ecp_stress_verdict_rejects_corruption(monkeypatch, trexio_file, seed, fault):
    """After a passing Gamma Li VMC stress baseline, reject wrong values, routes or missing stress.

    The stress fixture has Pade J1+J2+J3 and seed 0, with no cond(G)
    filtering. Both sides still perform the real native calculation.
    """
    assert trexio_file == "lithium.h5"
    test_vmc_ecp_stress_fast_matches_recomputed(True, seed)
    message = _install_corruption(monkeypatch, "vmc-stress", fault)
    with pytest.raises(AssertionError, match=message):
        test_vmc_ecp_stress_fast_matches_recomputed(True, seed)
