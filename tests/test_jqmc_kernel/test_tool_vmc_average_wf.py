"""``jqmc-tool vmc average-wf``: uniform average of VMC optimization-step wavefunctions (issue #749).

Independent references. The leaves that must be averaged are found by the test
itself, by comparing the input HDF5 arrays, and are also listed by hand for
every optimized-parameter combination. The expected value of each averaged
leaf is ``np.mean`` over the stacked input arrays read with h5py. The command's
own helpers are never used to build an expectation.

Optimization steps are produced in two ways. Light tests take committed TREXIO
fixtures and apply seeded random-walk updates through the production
``apply_block_update`` owners (J3/lambda symmetrization, S-metric polar
retraction of the MOs, ``C_dn = conj(C_up)`` under time reversal); the
combined steps are then assembled by the production twist dump writer
``write_twist_optimized_hamiltonian``. The heavy test runs a real VMC
optimization through the ``jqmc`` command line.
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

import copy
import hashlib
import logging
import re
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import toml
from typer.testing import CliRunner

from jqmc import jqmc_cli
from jqmc._checkpoint import load_observables_from_checkpoint
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._twist_average import load_twist, load_twist_average_input
from jqmc._twist_checkpoint import save_shared_state, twist_checkpoint_path, write_twist_optimized_hamiltonian
from jqmc.hamiltonians import Hamiltonian_data, _save_dataclass_to_hdf5
from jqmc.jqmc_tool import hamiltonian_app, trexio_app, vmc_app
from jqmc.wavefunction import evaluate_ln_wavefunction


pytestmark = [pytest.mark.without_chdir]

_TREXIO = Path(__file__).resolve().parents[1] / "trexio_example_files"
_PROVENANCE = "opt_step_average_"
_NO_FLAGS = dict(
    opt_J1_param=False,
    opt_J2_param=False,
    opt_J3_param=False,
    opt_geminal=False,
    opt_J3_basis_exp=False,
    opt_J3_basis_coeff=False,
    opt_lambda_basis_exp=False,
    opt_lambda_basis_coeff=False,
    opt_JDTN_u_param=False,
    opt_JDTN_chi_param=False,
    opt_JDTN_f_param=False,
)
_ALL_ZONES = ("ao_eval", "mo_eval", "det_eval", "jastrow_eval")

_WF = "/wavefunction_data"
_GEM = f"{_WF}/geminal_data"
_JAS = f"{_WF}/jastrow_data"
_MO = (f"{_GEM}/orb_data_up_spin/mo_coefficients", f"{_GEM}/orb_data_dn_spin/mo_coefficients")
_J1 = f"{_JAS}/jastrow_one_body_data/jastrow_1b_param"
_J2 = f"{_JAS}/jastrow_two_body_data/jastrow_2b_param"
_J3 = f"{_JAS}/jastrow_three_body_data/j_matrix"
_J3_RADIAL = f"{_JAS}/jastrow_three_body_data/orb_data/radial"
_DTN = f"{_JAS}/jastrow_dtn_data"


def _flags(**selected):
    return {**_NO_FLAGS, **selected}


# Each case: base fixture, optimized-parameter flags, and the hand-listed leaves those flags change.
_CASES = {
    "molecular-sd-j1j2j3-mo": (
        "molecular-sd",
        _flags(opt_J1_param=True, opt_J2_param=True, opt_J3_param=True, opt_geminal=True),
        {_J1, _J2, _J3, *_MO},
    ),
    "molecular-agp-j2-lambda": ("molecular-agp", _flags(opt_J2_param=True, opt_geminal=True), {_J2, f"{_GEM}/lambda_matrix"}),
    "molecular-agpn-j3-mo-lambda": (
        "molecular-agpn",
        _flags(opt_J3_param=True, opt_geminal=True),
        {_J3, *_MO, f"{_GEM}/lambda_diag"},
    ),
    "molecular-dtn": (
        "molecular-dtn",
        _flags(opt_JDTN_u_param=True, opt_JDTN_chi_param=True, opt_JDTN_f_param=True),
        {f"{_DTN}/alpha_indep", f"{_DTN}/beta_indep", f"{_DTN}/gamma_indep"},
    ),
    "molecular-sd-j2-j3basis": (
        "molecular-sd",
        _flags(opt_J2_param=True, opt_J3_basis_exp=True, opt_J3_basis_coeff=True),
        {_J2, f"{_J3_RADIAL}/exponents", f"{_J3_RADIAL}/coefficients"},
    ),
    "pbc-gamma-sd-j2j3-mo": ("pbc-gamma-sd", _flags(opt_J2_param=True, opt_J3_param=True, opt_geminal=True), {_J2, _J3, *_MO}),
}
_TWIST_LEAVES = {
    "/shared/wavefunction_data/jastrow_data/jastrow_one_body_data/jastrow_1b_param",
    "/shared/wavefunction_data/jastrow_data/jastrow_two_body_data/jastrow_2b_param",
} | {f"/twists/twist_{index:03d}/mo_coefficients_{spin}" for index in range(8) for spin in ("up", "dn")}


# --------------------------------------------------------------------------- fixtures


def _invoke(app, args):
    return CliRunner().invoke(app, [str(arg) for arg in args])


@pytest.fixture(scope="module")
def bases(tmp_path_factory):
    """Converted Hamiltonian inputs (real CLI conversions of committed TREXIO fixtures)."""
    root = tmp_path_factory.mktemp("average_wf_bases")
    paths = {}

    def convert(key, trexio, *args):
        paths[key] = root / f"{key}.h5"
        result = _invoke(trexio_app, ["convert-to", _TREXIO / trexio, "-o", paths[key], *args])
        assert result.exit_code == 0, (result.output, result.exception)

    convert("molecular-sd", "H2_ecp_ccpvtz_cart.h5", "-j1", "1.0", "-j2", "1.0", "-j3", "ao-small")
    convert("molecular-dtn", "H2_ecp_ccpvtz_cart.h5", "--jastrow-dtn", "u_chi_f")
    convert("pbc-gamma-sd", "H2_pbc_ae_cart.h5", "-j2", "1.0", "-j3", "ao-small")
    convert("twist-combined", "lithium_twist_222.h5", "-j1", "1.0", "-j2", "1.0", "--twist-average")
    for key, target in (("molecular-agp", "jagp"), ("molecular-agpn", "jagpn")):
        paths[key] = root / f"{key}.h5"
        result = _invoke(hamiltonian_app, ["conv-wf", paths["molecular-sd"], "-c", target, "-o", paths[key]])
        assert result.exit_code == 0, (result.output, result.exception)
    return paths


def _perturbed(block, rng, scale):
    """A seeded random step of one published block (exponents stay positive)."""
    values = np.asarray(block.values)
    delta = scale * rng.standard_normal(values.shape)
    if np.iscomplexobj(values):
        delta = delta + 1j * scale * rng.standard_normal(values.shape)
    new = values * np.exp(delta.real) if block.name.endswith("basis_exp") else values + delta
    return block.replace(values=np.asarray(new, dtype=values.dtype))


def _step(wavefunction, flags, rng, scale, token):
    """Apply one random update of every published block through the production owners."""
    jastrow, geminal = wavefunction.jastrow_data, wavefunction.geminal_data
    for block in wavefunction.get_variational_blocks(state_token=token, **flags):
        updated = _perturbed(block, rng, scale)
        jastrow = jastrow.apply_block_update(updated)
        geminal = geminal.apply_block_update(updated)
    return wavefunction.replace(jastrow_data=jastrow, geminal_data=geminal)


def _write_steps(base, directory, flags, count, *, seed, scale=2.0e-3, first_step=1):
    """``count`` consecutive single-twist steps of a random walk starting from ``base``."""
    directory.mkdir(parents=True, exist_ok=True)
    hamiltonian = Hamiltonian_data.load_from_hdf5(base)
    rng = np.random.default_rng(seed)
    paths = []
    for step in range(first_step, first_step + count):
        wavefunction = _step(hamiltonian.wavefunction_data, flags, rng, scale, ("average-wf-test", step))
        hamiltonian = hamiltonian.replace(wavefunction_data=wavefunction)
        paths.append(directory / f"hamiltonian_data_opt_step_{step}.h5")
        hamiltonian.save_to_hdf5(paths[-1])
    return paths


def _write_twist_steps(source, directory, count, *, seed, scale=2.0e-3):
    """Combined steps: shared J1/J2 and per-twist complex MOs, written by the production dump writer."""
    directory.mkdir(parents=True, exist_ok=True)
    n_twist = load_twist_average_input(str(source)).n_twist
    hamiltonians = [load_twist(str(source), index) for index in range(n_twist)]
    rng = np.random.default_rng(seed)
    jastrow_flags = _flags(opt_J1_param=True, opt_J2_param=True)
    geminal_flags = _flags(opt_geminal=True)
    paths = []
    for step in range(1, count + 1):
        token = ("average-wf-twist-test", step)
        shared = [
            _perturbed(block, rng, scale)
            for block in hamiltonians[0].wavefunction_data.get_variational_blocks(state_token=token, **jastrow_flags)
        ]
        for index, hamiltonian in enumerate(hamiltonians):
            wavefunction = hamiltonian.wavefunction_data
            jastrow = wavefunction.jastrow_data
            for block in shared:
                jastrow = jastrow.apply_block_update(block)
            wavefunction = _step(wavefunction.replace(jastrow_data=jastrow), geminal_flags, rng, scale, token)
            hamiltonians[index] = hamiltonian.replace(wavefunction_data=wavefunction)
        checkpoint_dir = directory / f"checkpoint_{step}"
        checkpoint_dir.mkdir()
        for index, hamiltonian in enumerate(hamiltonians):
            with h5py.File(twist_checkpoint_path(str(checkpoint_dir), index), "w") as h5:
                meta = h5.create_group("_meta")
                meta.attrs["twist_index"] = index
                _save_dataclass_to_hdf5(h5.create_group("hamiltonian_data"), hamiltonian)
        save_shared_state(str(checkpoint_dir), hamiltonians[0])
        paths.append(directory / f"hamiltonian_data_opt_step_{step}.h5")
        write_twist_optimized_hamiltonian(str(source), str(checkpoint_dir), str(paths[-1]))
    return paths


@pytest.fixture(scope="module")
def twist_steps(tmp_path_factory, bases):
    """Four combined optimization-step dumps of the 8-twist lithium input (built once per module)."""
    return _write_twist_steps(bases["twist-combined"], tmp_path_factory.mktemp("average_wf_twist_steps"), 4, seed=2749)


def _copy_steps(paths, directory, count):
    """Private copies of the first ``count`` shared step files (tests modify their inputs)."""
    directory.mkdir(parents=True, exist_ok=True)
    copies = [directory / path.name for path in paths[:count]]
    for path, copy_path in zip(paths, copies, strict=False):
        shutil.copyfile(path, copy_path)
    return copies


# --------------------------------------------------------------------------- independent readers


def _read_tree(path):
    """Every dataset (``/a/b``) and attribute (``/a@b``) of an HDF5 file, read with h5py."""
    tree = {}
    with h5py.File(path, "r") as h5:

        def visit(name, obj):
            prefix = "/" + name
            for key, value in obj.attrs.items():
                tree[f"{prefix}@{key}"] = np.asarray(value)
            if isinstance(obj, h5py.Dataset):
                tree[prefix] = np.asarray(obj[()])

        for key, value in h5.attrs.items():
            tree[f"/@{key}"] = np.asarray(value)
        h5.visititems(visit)
    return tree


def _same_bits(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.kind in "biufc":
        return a.tobytes() == b.tobytes()
    return bool(np.array_equal(a, b))


def _without_provenance(tree):
    return {leaf: value for leaf, value in tree.items() if not leaf.startswith(f"/@{_PROVENANCE}")}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _average(inputs, output):
    digests = [_sha256(path) for path in inputs]
    result = _invoke(vmc_app, ["average-wf", *inputs, "-o", output])
    assert [_sha256(path) for path in inputs] == digests, "average-wf modified an input file"
    return result


def _assert_uniform_mean(inputs, output, expected_leaves):
    """Averaged leaves equal the NumPy mean of the inputs; every other leaf keeps the first input's bits."""
    trees = [_without_provenance(_read_tree(path)) for path in inputs]
    out = _read_tree(output)
    differing = {leaf for leaf in trees[0] if any(not _same_bits(trees[0][leaf], tree[leaf]) for tree in trees[1:])}
    assert differing == set(expected_leaves), "the inputs do not change exactly the hand-listed parameters"
    assert set(out[f"/@{_PROVENANCE}leaves"].astype(str)) == differing
    assert _without_provenance(out).keys() == trees[0].keys()
    atol, rtol = get_tolerance_min(_ALL_ZONES, "strict")
    for leaf, first in trees[0].items():
        actual = out[leaf]
        if leaf not in differing:
            assert _same_bits(actual, first), f"invariant leaf {leaf} was not copied bitwise"
            continue
        stack = np.stack([tree[leaf] for tree in trees])
        assert actual.dtype == first.dtype and actual.shape == first.shape
        work = np.complex128 if np.iscomplexobj(stack) else np.float64
        np.testing.assert_allclose(actual.astype(work), np.mean(stack.astype(work), axis=0), atol=atol, rtol=rtol, err_msg=leaf)
        constant = np.all(stack == stack[0], axis=0)
        np.testing.assert_array_equal(actual[constant], first[constant], err_msg=f"{leaf}: constant elements re-rounded")


def _load_every_twist(path):
    from jqmc._twist_average import is_twist_average_input

    if not is_twist_average_input(path):
        return [Hamiltonian_data.load_from_hdf5(path)]
    return [load_twist(str(path), index) for index in range(load_twist_average_input(str(path)).n_twist)]


# --------------------------------------------------------------------------- averaging contract


@pytest.mark.light
@pytest.mark.parametrize("case", sorted(_CASES))
def test_average_wf_equals_numpy_mean_and_keeps_invariant_leaves(tmp_path, bases, case):
    """Four random-walk steps: only the optimized leaves are averaged, all else stays bitwise.

    The expected leaf set is listed by hand per optimized-parameter
    combination (J1/J2/J3, AGP lambda, AGPn MO + lambda_diag, DTN, J3 basis,
    SD MOs; molecular and periodic Gamma). The output reloads through
    ``Hamiltonian_data.load_from_hdf5`` and the inputs are unchanged.
    """
    base, flags, expected = _CASES[case]
    inputs = _write_steps(bases[base], tmp_path / "steps", flags, 4, seed=749)
    output = tmp_path / "averaged.h5"
    result = _average(inputs, output)
    assert result.exit_code == 0, (result.output, result.exception)
    _assert_uniform_mean(inputs, output, expected)
    out = _read_tree(output)
    assert out[f"/@{_PROVENANCE}steps"].tolist() == [1, 2, 3, 4]
    assert out[f"/@{_PROVENANCE}sources"].astype(str).tolist() == [str(path) for path in inputs]
    assert int(out[f"/@{_PROVENANCE}num_sources"]) == 4
    assert str(out[f"/@{_PROVENANCE}weighting"]) == "uniform"
    _load_every_twist(output)[0].sanity_check()


@pytest.mark.light
def test_average_wf_twist_equals_numpy_mean_and_round_trips(tmp_path, twist_steps):
    """Combined opt-step dumps (8 complex twists) average per twist into a combined file that loads for every twist."""
    inputs = _copy_steps(twist_steps, tmp_path / "steps", 4)
    output = tmp_path / "averaged.h5"
    result = _average(inputs, output)
    assert result.exit_code == 0, (result.output, result.exception)
    _assert_uniform_mean(inputs, output, _TWIST_LEAVES)
    meta = load_twist_average_input(str(output))
    source_meta = load_twist_average_input(str(inputs[0]))
    assert meta.n_twist == source_meta.n_twist == 8
    assert meta.k_points == source_meta.k_points
    np.testing.assert_array_equal(meta.weights, source_meta.weights)
    tree = _read_tree(output)
    for index, hamiltonian in enumerate(_load_every_twist(output)):
        geminal = hamiltonian.wavefunction_data.geminal_data
        up = np.asarray(geminal.orb_data_up_spin.mo_coefficients)
        assert np.iscomplexobj(up)
        np.testing.assert_array_equal(up, tree[f"/twists/twist_{index:03d}/mo_coefficients_up"])
        # Time reversal survives the average exactly: mean(conj(C)) == conj(mean(C)).
        np.testing.assert_array_equal(
            np.asarray(geminal.orb_data_dn_spin.mo_coefficients), np.conj(up[: geminal.num_electron_dn])
        )


@pytest.mark.light
@pytest.mark.parametrize("layout", ["molecular-sd", "pbc-gamma-sd", "twist-combined"])
def test_average_wf_single_input_is_bitwise_identity(tmp_path, bases, twist_steps, layout):
    """One input: every dataset and attribute of the output equals the input bitwise; provenance is added."""
    if layout == "twist-combined":
        (source,) = _copy_steps(twist_steps, tmp_path / "steps", 1)
    else:
        (source,) = _write_steps(bases[layout], tmp_path / "steps", _CASES["molecular-sd-j1j2j3-mo"][1], 1, seed=5)
    output = tmp_path / "identity.h5"
    result = _average([source], output)
    assert result.exit_code == 0, (result.output, result.exception)
    expected, actual = _read_tree(source), _read_tree(output)
    assert _without_provenance(actual).keys() == expected.keys()
    for leaf, value in expected.items():
        assert _same_bits(actual[leaf], value), leaf
    assert actual[f"/@{_PROVENANCE}sources"].astype(str).tolist() == [str(source)]
    assert actual[f"/@{_PROVENANCE}steps"].tolist() == [1]
    assert actual[f"/@{_PROVENANCE}leaves"].size == 0
    _load_every_twist(output)


# --------------------------------------------------------------------------- refusals


def _set_dataset(path, leaf, transform):
    with h5py.File(path, "r+") as h5:
        values = h5[leaf][()]
        del h5[leaf]
        h5.create_dataset(leaf, data=transform(values))


def _refusal_inputs(tmp_path, bases, twist_steps, case):
    """Return ``(inputs, output, expected message fragment)`` for one refused input set."""
    sd_flags = _CASES["molecular-sd-j1j2j3-mo"][1]
    output = tmp_path / "averaged.h5"
    if case.startswith("twist"):
        inputs = _copy_steps(twist_steps, tmp_path / "twist", 2)
        if case == "twist-k-point":
            _set_dataset(inputs[1], "/twists/twist_003/k_point", lambda k: k + 0.25)
        elif case == "twist-count":
            with h5py.File(inputs[1], "r+") as h5:
                for index in range(2, 8):
                    del h5[f"twists/twist_{index:03d}"]
                h5["_meta"].attrs["n_twist"] = 2
        elif case == "twist-layout":
            inputs = [inputs[0], *_write_steps(bases["molecular-sd"], tmp_path / "sd", sd_flags, 1, seed=3)]
            return inputs, output, "is a single-twist file but"
        return inputs, output, "the twist set (count, k-points or weights) differs"
    inputs = _write_steps(bases["molecular-sd"], tmp_path / "sd", sd_flags, 3, seed=17)
    if case == "structure":
        _set_dataset(inputs[2], "/structure_data/positions", lambda r: r + 0.01)
        return inputs, output, "the atomic structure differs"
    if case == "basis-shape":
        other = tmp_path / "j3_ao_medium.h5"
        result = _invoke(
            trexio_app,
            ["convert-to", _TREXIO / "H2_ecp_ccpvtz_cart.h5", "-o", other, "-j1", "1.0", "-j2", "1.0", "-j3", "ao-medium"],
        )
        assert result.exit_code == 0, result.output
        return [*inputs, *_write_steps(other, tmp_path / "medium", sd_flags, 1, seed=3, first_step=4)], output, "has shape"
    if case == "basis-values":
        _set_dataset(inputs[1], f"{_GEM}/orb_data_up_spin/aos_data/ao_coefficient_scales", lambda s: s * 1.5)
        return inputs, output, "the basis set differs"
    if case == "frozen-j1-optimization":
        frozen = _write_steps(
            inputs[0],
            tmp_path / "frozen",
            _flags(opt_J2_param=True, opt_J3_param=True, opt_geminal=True),
            1,
            seed=5,
            first_step=4,
        )
        return [*inputs, *frozen], output, "stored optimization state"
    if case == "ansatz":
        agpn = _write_steps(bases["molecular-agpn"], tmp_path / "agpn", _flags(opt_geminal=True), 1, seed=5, first_step=4)
        return [*inputs, *agpn], output, "the wavefunction structure (ansatz"
    if case == "precision-dtype":
        # The other precision mode stores the Jastrow zone in the other float width.
        _set_dataset(inputs[1], _J3, lambda j: j.astype(np.float64 if j.dtype == np.float32 else np.float32))
        return inputs, output, "a different precision mode stores different dtypes"
    if case == "non-finite":
        _set_dataset(inputs[1], _J3, lambda j: np.where(np.arange(j.size).reshape(j.shape) == 0, np.nan, j))
        return inputs, output, "holds a non-finite value"
    if case in ("unchanged-nonfinite-mo", "single-nonfinite-mo"):
        # A supported independent-spin SD (time_reversal=False) whose up-spin MO holds the
        # same NaN in every input; only J3 differs, so the MO leaf itself is unchanged.
        independent = tmp_path / "independent_spin.h5"
        hamiltonian = Hamiltonian_data.load_from_hdf5(bases["molecular-sd"])
        geminal = hamiltonian.wavefunction_data.geminal_data.replace(time_reversal=False)
        hamiltonian.replace(wavefunction_data=hamiltonian.wavefunction_data.replace(geminal_data=geminal)).save_to_hdf5(
            independent
        )
        count = 3 if case == "unchanged-nonfinite-mo" else 1
        inputs = _write_steps(independent, tmp_path / "independent", _flags(opt_J3_param=True), count, seed=13)
        for path in inputs:
            _set_dataset(path, _MO[0], lambda c: np.where(np.arange(c.size).reshape(c.shape) == 0, np.nan, c))
        return inputs, output, "holds a non-finite value"
    if case == "duplicate":
        return [*inputs, inputs[0]], output, "is given more than once"
    if case == "output-is-input":
        return inputs, inputs[1], "is one of the inputs"
    if case == "not-hamiltonian":
        return [*inputs, _TREXIO / "H2_ecp_ccpvtz_cart.h5"], output, "is not a Hamiltonian data file"
    raise AssertionError(case)


_REFUSALS = [
    "structure",
    "basis-shape",
    "basis-values",
    "frozen-j1-optimization",
    "ansatz",
    "precision-dtype",
    "non-finite",
    "unchanged-nonfinite-mo",
    "single-nonfinite-mo",
    "duplicate",
    "output-is-input",
    "not-hamiltonian",
    "twist-k-point",
    "twist-count",
    "twist-layout",
]


@pytest.mark.light
@pytest.mark.parametrize("case", _REFUSALS)
@pytest.mark.parametrize("existing_output", [False, True], ids=["no-existing-output", "existing-output"])
def test_average_wf_refuses_inconsistent_inputs_without_writing(tmp_path, bases, twist_steps, case, existing_output):
    """Inputs that are not steps of one optimization exit 1 with the named cause and write nothing.

    Mixing a J1-optimized with a J1-frozen optimization is refused because the
    optimizer stores a never-updated scalar parameter as an HDF5 attribute;
    the SD -> AGPn ansatz, a float32 (mixed-precision) J3 and every structure,
    basis, k-point or twist-count difference are refused too.
    """
    inputs, output, message = _refusal_inputs(tmp_path, bases, twist_steps, case)
    sentinel = b"existing output must survive a refusal"
    if existing_output and case != "output-is-input":
        output.write_bytes(sentinel)
    before = {path: _sha256(path) for path in inputs}
    result = CliRunner().invoke(vmc_app, ["average-wf", *map(str, inputs), "-o", str(output)])
    assert result.exit_code == 1, (result.output, result.exception)
    assert result.stderr.startswith("Error: ") and message in result.stderr, result.stderr
    assert "Averaged Hamiltonian data is saved" not in result.stdout
    assert {path: _sha256(path) for path in inputs} == before
    if case != "output-is-input":
        assert output.read_bytes() == sentinel if existing_output else not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.tmp")), "a temporary file was left behind"


@pytest.mark.light
@pytest.mark.parametrize("layout", ["molecular-real-sign-flip", "twist-complex-phase-jump"])
def test_average_wf_refuses_an_mo_gauge_jump(tmp_path, bases, twist_steps, layout):
    """A sign / phase jump of one MO row between steps is refused with its row coherence.

    The jumped file alone is a valid input (the row phase is a determinant
    gauge), but the arithmetic mean over the jump is not the mean of one
    continuous optimization. The reported coherence equals the independent
    ``|sum_i c_i| / sum_i |c_i|`` of that row.
    """
    if layout.startswith("twist"):
        inputs = _copy_steps(twist_steps, tmp_path / "steps", 4)
        up, dn, phase = "/twists/twist_002/mo_coefficients_up", "/twists/twist_002/mo_coefficients_dn", 1j
    else:
        inputs = _write_steps(bases["molecular-sd"], tmp_path / "steps", _CASES["molecular-sd-j1j2j3-mo"][1], 4, seed=11)
        up, dn, phase = _MO[0], _MO[1], -1.0
    _set_dataset(inputs[2], up, lambda c: c * phase)
    _set_dataset(inputs[2], dn, lambda c: c * np.conj(phase))
    _load_every_twist(inputs[2])  # still a valid Hamiltonian on its own
    rows = np.stack([_read_tree(path)[up][0] for path in inputs])
    coherence = np.linalg.norm(rows.sum(axis=0)) / np.sum(np.linalg.norm(rows, axis=1))
    output = tmp_path / "averaged.h5"
    result = _average(inputs, output)
    assert result.exit_code == 1, (result.output, result.exception)
    assert "are not coherent across the inputs" in result.stderr and f"{coherence:.6f}" in result.stderr, result.stderr
    assert not output.exists()


@pytest.mark.light
def test_average_wf_warns_when_parameters_do_not_change_together(tmp_path, bases):
    """A block frozen between two steps, and a duplicated state, average with a warning instead of a refusal."""
    inputs = _write_steps(bases["molecular-sd"], tmp_path / "steps", _CASES["molecular-sd-j1j2j3-mo"][1], 4, seed=23)
    with h5py.File(inputs[0], "r") as source:
        frozen = source[_J3][()]
    _set_dataset(inputs[1], _J3, lambda _: frozen)
    duplicate = tmp_path / "steps" / "copy_of_step_4.h5"
    shutil.copyfile(inputs[3], duplicate)
    output = tmp_path / "averaged.h5"
    result = _average([*inputs, duplicate], output)
    assert result.exit_code == 0, (result.output, result.exception)
    assert f"{inputs[3]} and {duplicate} hold identical parameters" in result.stderr
    assert f"{_J3} is identical in {inputs[0]} and {inputs[1]}" in result.stderr
    _assert_uniform_mean([*inputs, duplicate], output, {_J1, _J2, _J3, *_MO})


@pytest.mark.light
@pytest.mark.parametrize("base", ["molecular-sd", "molecular-agp", "molecular-agpn", "molecular-dtn", "pbc-gamma-sd"])
def test_average_wf_accepts_every_leaf_the_optimizer_updates(tmp_path, bases, base):
    """Every leaf written back by a published variational block is averaged, and no other leaf changes.

    For each block of ``get_variational_blocks`` with every optimization flag
    set, a pair of files that differ only by that block's update (through the
    production ``apply_block_update`` owners) must average successfully, and
    the averaged leaves are exactly the leaves the update changed.
    """
    hamiltonian = Hamiltonian_data.load_from_hdf5(bases[base])
    all_flags = dict.fromkeys(_NO_FLAGS, True)
    token = ("average-wf-allowlist", 0)
    wavefunction = hamiltonian.wavefunction_data
    blocks = wavefunction.get_variational_blocks(state_token=token, **all_flags)
    identity = wavefunction
    for block in blocks:  # an unperturbed update fixes every leaf's stored representation
        identity = identity.replace(
            jastrow_data=identity.jastrow_data.apply_block_update(block),
            geminal_data=identity.geminal_data.apply_block_update(block),
        )
    reference = tmp_path / "reference.h5"
    hamiltonian.replace(wavefunction_data=identity).save_to_hdf5(reference)
    rng = np.random.default_rng(3)
    checked = 0
    for index, block in enumerate(blocks):
        updated = _perturbed(block, rng, 1.0e-2)
        moved = identity.replace(
            jastrow_data=identity.jastrow_data.apply_block_update(updated),
            geminal_data=identity.geminal_data.apply_block_update(updated),
        )
        candidate = tmp_path / f"block_{index}_{block.name}.h5"
        hamiltonian.replace(wavefunction_data=moved).save_to_hdf5(candidate)
        first, second = _read_tree(reference), _read_tree(candidate)
        assert first.keys() == second.keys(), block.name
        changed = {leaf for leaf in first if not _same_bits(first[leaf], second[leaf])}
        if not changed:  # e.g. SD / AGPn orbital-basis blocks are published but not written back
            continue
        output = tmp_path / f"average_{index}.h5"
        result = _average([reference, candidate], output)
        assert result.exit_code == 0, (block.name, result.stderr)
        assert set(_read_tree(output)[f"/@{_PROVENANCE}leaves"].astype(str)) == changed, block.name
        checked += 1
    assert checked >= 2


# --------------------------------------------------------------------------- the averaged wavefunction


def _inject_mean(hamiltonian, inputs):
    """The first input's Hamiltonian with NumPy-mean J1/J2/J3 and MO coefficients set via dataclass fields."""
    trees = [_read_tree(path) for path in inputs]

    def mean(leaf):
        stack = np.stack([tree[leaf] for tree in trees])
        return np.mean(stack.astype(np.complex128 if np.iscomplexobj(stack) else np.float64), axis=0).astype(stack.dtype)

    wavefunction = hamiltonian.wavefunction_data
    jastrow, geminal = wavefunction.jastrow_data, wavefunction.geminal_data
    jastrow = jastrow.replace(
        jastrow_one_body_data=jastrow.jastrow_one_body_data.replace(jastrow_1b_param=mean(_J1))
        if jastrow.jastrow_one_body_data is not None
        else None,
        jastrow_two_body_data=jastrow.jastrow_two_body_data.replace(jastrow_2b_param=mean(_J2)),
        jastrow_three_body_data=jastrow.jastrow_three_body_data.replace(j_matrix=mean(_J3)),
    )
    geminal = geminal.replace(
        orb_data_up_spin=geminal.orb_data_up_spin.replace(mo_coefficients=mean(_MO[0])),
        orb_data_dn_spin=geminal.orb_data_dn_spin.replace(mo_coefficients=mean(_MO[1])),
    )
    return wavefunction.replace(jastrow_data=jastrow, geminal_data=geminal)


@pytest.mark.heavy
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
@pytest.mark.parametrize("layout", ["molecular-sd", "pbc-gamma-sd"])
def test_average_wf_output_evaluates_the_mean_parameter_wavefunction(tmp_path, bases, layout):
    """The averaged file reloads and gives a finite ln|Psi| equal to the NumPy-mean-parameter wavefunction.

    The determinant needs only the averaged coefficients, so no
    re-orthonormalization is applied; for a single determinant a row
    re-orthonormalization would only rescale Psi by a constant.
    """
    flags = _flags(opt_J1_param=layout == "molecular-sd", opt_J2_param=True, opt_J3_param=True, opt_geminal=True)
    inputs = _write_steps(bases[layout], tmp_path / "steps", flags, 5, seed=4)
    output = tmp_path / "averaged.h5"
    result = _average(inputs, output)
    assert result.exit_code == 0, (result.output, result.exception)
    averaged = Hamiltonian_data.load_from_hdf5(output)
    reference = _inject_mean(Hamiltonian_data.load_from_hdf5(inputs[0]), inputs)
    rng = np.random.default_rng(749)
    center = np.asarray(averaged.structure_data.positions)
    geminal = averaged.wavefunction_data.geminal_data
    atol, rtol = get_tolerance_min(("ao_eval", "mo_eval", "det_eval", "jastrow_eval", "wf_eval"), "strict")
    for _ in range(3):
        r_up = center[rng.integers(len(center), size=geminal.num_electron_up)] + rng.normal(
            scale=0.7, size=(geminal.num_electron_up, 3)
        )
        r_dn = center[rng.integers(len(center), size=geminal.num_electron_dn)] + rng.normal(
            scale=0.7, size=(geminal.num_electron_dn, 3)
        )
        actual = float(evaluate_ln_wavefunction(averaged.wavefunction_data, r_up, r_dn))
        expected = float(evaluate_ln_wavefunction(reference, r_up, r_dn))
        assert np.isfinite(actual)
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _run_jqmc(root, monkeypatch, payload):
    """Run the ``jqmc`` command line in ``root`` on one TOML input."""
    input_file = root / f"{payload['control']['job_type']}.toml"
    input_file.write_text(toml.dumps(payload), encoding="utf-8")
    logger = logging.getLogger("jqmc")
    handlers, level = list(logger.handlers), logger.level
    try:
        with monkeypatch.context() as patch:
            patch.chdir(root)
            patch.setattr(sys, "argv", ["jqmc", str(input_file)])
            patch.setattr(jqmc_cli, "cli_parameters", copy.deepcopy(jqmc_cli.cli_parameters))
            jqmc_cli._cli()
    finally:
        for handler in logger.handlers:
            if handler not in handlers and isinstance(handler, logging.FileHandler):
                handler.close()
        logger.handlers, logger.level = handlers, level


@pytest.mark.heavy
@pytest.mark.gdd_electron_configuration(mode="required", position_source="production-driver-generated")
def test_average_wf_of_real_optimizer_steps_runs_in_jqmc(tmp_path, bases, monkeypatch):
    """Real ``jqmc`` VMC steps (J1/J2/J3 + SD MOs) average to the NumPy mean and drive a ``jqmc`` MCMC run."""
    run = tmp_path / "vmc"
    run.mkdir()
    shutil.copyfile(bases["molecular-sd"], run / "hamiltonian_data.h5")
    precision = {"mode": "mixed" if is_mixed_precision_enabled() else "full"}
    vmc = copy.deepcopy(jqmc_cli.cli_parameters["vmc"])
    vmc.update(
        num_mcmc_steps=40,
        num_mcmc_per_measurement=10,
        num_mcmc_warmup_steps=5,
        num_mcmc_bin_blocks=5,
        num_opt_steps=4,
        wf_dump_freq=1,
        opt_J1_param=True,
        opt_J2_param=True,
        opt_J3_param=True,
        opt_geminal=True,
    )
    vmc["optimizer_kwargs"] = {**vmc["optimizer_kwargs"], "method": "sr", "delta": 0.05, "epsilon": 0.001}
    control = {"mcmc_seed": 34456, "number_of_walkers": 4, "restart": False, "verbosity": "low"}
    _run_jqmc(
        run,
        monkeypatch,
        {
            "control": {**control, "job_type": "vmc", "hamiltonian_h5": "hamiltonian_data.h5", "restart_chk": "vmc.h5"},
            "precision": precision,
            "vmc": vmc,
        },
    )
    inputs = [run / f"hamiltonian_data_opt_step_{step}.h5" for step in (2, 3, 4)]
    output = run / "hamiltonian_data_averaged.h5"
    result = _average(inputs, output)
    assert result.exit_code == 0, (result.output, result.exception)
    _assert_uniform_mean(inputs, output, {_J1, _J2, _J3, *_MO})
    assert _read_tree(output)[f"/@{_PROVENANCE}steps"].tolist() == [2, 3, 4]
    assert re.search(r"Smallest MO-row coherence .* = 0\.99\d+", result.stdout)
    mcmc = copy.deepcopy(jqmc_cli.cli_parameters["mcmc"])
    mcmc.update(num_mcmc_steps=20, num_mcmc_per_measurement=5, num_mcmc_warmup_steps=2, num_mcmc_bin_blocks=2)
    _run_jqmc(
        run,
        monkeypatch,
        {
            "control": {**control, "job_type": "mcmc", "hamiltonian_h5": output.name, "restart_chk": "mcmc.h5"},
            "precision": precision,
            "mcmc": mcmc,
        },
    )
    observables = load_observables_from_checkpoint(str(run / "mcmc.h5"), ["e_L"])
    local_energies = np.concatenate([np.ravel(rank["e_L"]) for rank in observables])
    assert local_energies.size > 0 and np.all(np.isfinite(local_energies))


@pytest.mark.light
def test_tutorial_helper_average_last_writes_the_numpy_mean_of_the_last_dumps(tmp_path, bases):
    """``examples/copy_optimized_hamiltonian.py --average-last 4`` averages exactly the four largest numeric steps.

    The helper runs as a real subprocess, as in the tutorials, and calls
    ``jqmc-tool vmc average-wf``; it refuses an existing destination and a
    window longer than the available dumps without touching the destination.
    """
    import subprocess

    helper = Path(__file__).resolve().parents[2] / "examples" / "copy_optimized_hamiltonian.py"
    steps = _write_steps(bases["molecular-sd"], tmp_path / "vmc", _CASES["molecular-sd-j1j2j3-mo"][1], 6, seed=91)
    destination = tmp_path / "mcmc" / "hamiltonian_data.h5"
    destination.parent.mkdir()

    def helper_run(*args):
        return subprocess.run([sys.executable, str(helper), *map(str, args)], capture_output=True, text=True)

    short = helper_run("--average-last", 20, tmp_path / "vmc", destination)
    assert short.returncode == 1 and "only 6 dumps" in short.stderr and not destination.exists()
    result = helper_run("--average-last", 4, tmp_path / "vmc", destination)
    assert result.returncode == 0, result.stderr
    assert f"Averaged optimized trial: 4 dumps, steps 3-6 of {tmp_path / 'vmc'} -> {destination}" in result.stdout
    _assert_uniform_mean(steps[2:], destination, {_J1, _J2, _J3, *_MO})
    assert _read_tree(destination)[f"/@{_PROVENANCE}steps"].tolist() == [3, 4, 5, 6]
    digest = _sha256(destination)
    again = helper_run("--average-last", 4, tmp_path / "vmc", destination)
    assert again.returncode == 1 and "already exists" in again.stderr and _sha256(destination) == digest
