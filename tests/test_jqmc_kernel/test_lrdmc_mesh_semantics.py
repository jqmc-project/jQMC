"""Independent contracts for the finite-a LRDMC random-mesh prescription.

These tests arbitrate physical/mathematical mesh conventions directly.  Backend
parity alone cannot detect a convention error shared by XLA and Rust.
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

import ast
from pathlib import Path

import jax
import numpy as np
import pytest

pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

ROOT = Path(__file__).resolve().parents[2]


def _function_node(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    assert len(matches) == 1, f"expected one production function {name!r}, found {len(matches)}"
    return matches[0]


def _call_keyword_values(function: ast.AST, callee: str, keyword: str) -> list[str]:
    values = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        called = ast.unparse(node.func)
        if called.rsplit(".", 1)[-1] != callee:
            continue
        values.extend(ast.unparse(item.value) for item in node.keywords if item.arg == keyword)
    return sorted(values)


def _nv_defaults(tree: ast.AST) -> dict[str, str]:
    defaults = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        positional = [*node.args.posonlyargs, *node.args.args]
        paired = zip(positional[-len(node.args.defaults) :], node.args.defaults, strict=True) if node.args.defaults else ()
        for argument, default in paired:
            if argument.arg == "Nv":
                defaults[node.name] = ast.unparse(default)
        for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
            if argument.arg == "Nv" and default is not None:
                defaults[node.name] = ast.unparse(default)
    return defaults


def _local_call_graph(tree: ast.AST) -> dict[str, set[str]]:
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return {
        name: {
            ast.unparse(call.func).rsplit(".", 1)[-1]
            for call in ast.walk(function)
            if isinstance(call, ast.Call) and ast.unparse(call.func).rsplit(".", 1)[-1] in functions
        }
        for name, function in functions.items()
    }


def _reaches(graph: dict[str, set[str]], start: str, target: str) -> bool:
    pending = [start]
    seen = set()
    while pending:
        current = pending.pop()
        if current == target:
            return True
        if current in seen:
            continue
        seen.add(current)
        pending.extend(graph.get(current, ()))
    return False


def _numpy_haar_rz_ry_rx(uniforms: np.ndarray) -> np.ndarray:
    """Independent NumPy statement of the prescribed Haar Euler map."""
    alpha = 2.0 * np.pi * uniforms[..., 0]
    sin_beta = 2.0 * uniforms[..., 1] - 1.0
    cos_beta = np.sqrt(1.0 - sin_beta * sin_beta)
    gamma = 2.0 * np.pi * uniforms[..., 2]
    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_g, sin_g = np.cos(gamma), np.sin(gamma)
    rows = (
        (cos_beta * cos_g, cos_g * sin_a * sin_beta - cos_a * sin_g, sin_a * sin_g + cos_a * cos_g * sin_beta),
        (cos_beta * sin_g, cos_a * cos_g + sin_a * sin_beta * sin_g, cos_a * sin_beta * sin_g - cos_g * sin_a),
        (-sin_beta, cos_beta * sin_a, cos_a * cos_beta),
    )
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def test_kinetic_and_ecp_rotation_streams_are_distinct_and_deterministic():
    """Kinetic and ECP rotations use distinct deterministic Threefry substreams.

    The kinetic stream is the rotation key itself, so the ECP stream does not
    perturb the kinetic lattice or move-key schedule.  The
    ECP stream is the second child of that key and therefore cannot be silently
    re-coupled to the kinetic orientation.
    """
    from jqmc.jqmc_gfmc import _split_lrdmc_rotation_keys

    rotation_key = jax.random.PRNGKey(34456)
    kinetic_key, ecp_key = _split_lrdmc_rotation_keys(rotation_key)
    expected_ecp_key = jax.random.split(rotation_key, 2)[1]

    np.testing.assert_array_equal(np.asarray(kinetic_key), np.asarray(rotation_key))
    np.testing.assert_array_equal(np.asarray(ecp_key), np.asarray(expected_ecp_key))
    assert not np.array_equal(np.asarray(kinetic_key), np.asarray(ecp_key))


def test_xla_lrdmc_production_wires_kinetic_and_ecp_rotations_separately():
    """Every XLA GFMC production call site receives its designated mesh rotation.

    This source-level implementation contract covers both conventional and
    load-balanced algorithms, including the full-recompute branches behind the
    always-streaming production route. Kinetic kernels must receive ``R``/``RT``;
    non-local ECP kernels and ECP cache builders/advancers must receive the
    independent ``ecp_R``/``ecp_RT``. A swapped variable therefore fails even if
    a currently active cache makes that argument numerically dormant.
    """
    tree = ast.parse((ROOT / "jqmc/jqmc_gfmc.py").read_text(encoding="utf-8"))

    t_core = _function_node(tree, "_projection_t_core")
    assert _call_keyword_values(t_core, "compute_discretized_kinetic_energy_fast_update", "RT") == ["RT"]
    assert _call_keyword_values(t_core, "compute_ecp_non_local_parts_nearest_neighbors_fast_update", "RT") == ["ecp_RT"]

    t_first = _function_node(tree, "_projection_t_streaming_first")
    assert _call_keyword_values(t_first, "_init_discretized_kinetic_mesh_streaming_state", "RT") == ["RT"]
    assert _call_keyword_values(t_first, "_init_ecp_candidate_mesh_streaming_state", "RT") == ["ecp_RT"]
    t_streaming = _function_node(tree, "_projection_t_streaming")
    assert _call_keyword_values(t_streaming, "_advance_ecp_candidate_mesh_streaming_state", "RT") == ["ecp_RT"]

    n_projection = _function_node(tree, "_projection_n")
    assert _call_keyword_values(n_projection, "compute_discretized_kinetic_energy_fast_update", "RT") == ["R.T"]
    assert _call_keyword_values(n_projection, "compute_ecp_non_local_parts_nearest_neighbors_fast_update", "RT") == [
        "ecp_R.T",
        "ecp_R.T",
    ]
    assert _call_keyword_values(n_projection, "_init_discretized_kinetic_mesh_streaming_state", "RT") == ["interval_R.T"]
    assert _call_keyword_values(n_projection, "_init_ecp_candidate_mesh_streaming_state", "RT") == ["interval_ecp_R.T"]
    assert _call_keyword_values(n_projection, "_advance_ecp_candidate_mesh_streaming_state", "RT") == ["interval_ecp_R.T"]


def test_default_ecp_quadrature_is_nv12_icosahedron():
    """All implicit production Nv routes resolve to the 12-point icosahedron.

    Direct ECP kernels bind ``Nv_default`` in their signatures. Derivative
    wrappers may expose ``Nv=None`` but must reach the single resolver in
    ``_ecp_nl_common``. No production call may hide a literal quadrature order;
    literal Nv values are reserved for the four base-mesh declarations.
    """
    from jqmc._setting import (
        ATOL_BIT_EXACT,
        ATOL_LRDMC_ICOSAHEDRON_GEOMETRY,
        Nv_default,
        RTOL_BIT_EXACT,
    )
    from jqmc.jqmc_mcmc import _NV_BASE_MESH

    mesh = _NV_BASE_MESH[Nv_default]
    points = np.asarray(mesh.grid_points)
    weights = np.asarray(mesh.weights)
    assert Nv_default == 12
    assert mesh.Nv == 12
    assert points.shape == (12, 3)
    np.testing.assert_allclose(
        np.linalg.norm(points, axis=1),
        np.ones(12),
        atol=ATOL_LRDMC_ICOSAHEDRON_GEOMETRY,
        rtol=RTOL_BIT_EXACT,
    )
    np.testing.assert_allclose(
        weights,
        np.full(12, 1.0 / 12.0),
        atol=ATOL_BIT_EXACT,
        rtol=RTOL_BIT_EXACT,
    )
    dots = points @ points.T
    off_diagonal = dots[~np.eye(12, dtype=bool)]
    expected_dot_values = np.asarray([-1.0, -1.0 / np.sqrt(5.0), 1.0 / np.sqrt(5.0)])
    for value in off_diagonal:
        assert np.min(np.abs(expected_dot_values - value)) < ATOL_LRDMC_ICOSAHEDRON_GEOMETRY

    coulomb_tree = ast.parse((ROOT / "jqmc/coulomb_potential.py").read_text(encoding="utf-8"))
    coulomb_defaults = _nv_defaults(coulomb_tree)
    assert coulomb_defaults
    assert set(coulomb_defaults.values()) == {"Nv_default"}

    derivative_tree = ast.parse((ROOT / "jqmc/derivatives/ecp_force.py").read_text(encoding="utf-8"))
    derivative_defaults = _nv_defaults(derivative_tree)
    assert derivative_defaults
    assert set(derivative_defaults.values()) == {"None"}
    resolver = _function_node(derivative_tree, "_ecp_nl_common")
    assert any(
        isinstance(node, ast.IfExp) and ast.unparse(node) == "Nv_default if Nv is None else Nv" for node in ast.walk(resolver)
    )
    graph = _local_call_graph(derivative_tree)
    assert all(_reaches(graph, function, "_ecp_nl_common") for function in derivative_defaults)

    literal_calls = []
    for path in sorted((ROOT / "jqmc").rglob("*.py")):
        production_tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in (node for node in ast.walk(production_tree) if isinstance(node, ast.Call)):
            for keyword in call.keywords:
                if keyword.arg == "Nv" and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, int):
                    literal_calls.append((path.relative_to(ROOT).as_posix(), ast.unparse(call.func), keyword.value.value))
    assert literal_calls == [
        ("jqmc/coulomb_potential.py", "_Mesh", 4),
        ("jqmc/coulomb_potential.py", "_Mesh", 6),
        ("jqmc/coulomb_potential.py", "_Mesh", 12),
        ("jqmc/coulomb_potential.py", "_Mesh", 18),
    ]


def test_nv6_mesh_is_confined_to_explicit_quadrature_suites():
    """Rust tests may use Nv=6 only when quadrature order is the test input.

    Production-default parity, invariant, dispatch, and driver suites must use
    the same Nv=12 icosahedron as an implicit XLA production call.  The
    reviewed exceptions exercise the lower-order quadrature API explicitly;
    any new Nv=6 use therefore requires a visible contract review instead of
    silently comparing a Rust Nv=6 result with an XLA Nv=12 reference.
    """
    explicit_quadrature_suites = {
        "tests/test_rs_kernel/test_agpn.py",
        "tests/test_rs_kernel/test_complex_composite.py",
        "tests/test_rs_kernel/test_complex_core.py",
        "tests/test_rs_kernel/test_ecp_local.py",
        "tests/test_rs_kernel/test_ecp_rotated_mesh.py",
        "tests/test_rs_kernel/test_ewald.py",
        # The mixed ECP/AE derivative compares the same explicit Nv=6 rule.
        "tests/test_rs_kernel/test_mixed_ecp_ae_routes.py",
    }

    def requests_nv6(path: Path) -> bool:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.alias) and node.name == "octahedron_sym_mesh_Nv6":
                return True
            if isinstance(node, ast.Name) and node.id == "octahedron_sym_mesh_Nv6":
                return True
            if isinstance(node, ast.Attribute) and node.attr == "octahedron_sym_mesh_Nv6":
                return True
            if isinstance(node, ast.Call) and any(
                keyword.arg == "Nv" and isinstance(keyword.value, ast.Constant) and keyword.value.value == 6
                for keyword in node.keywords
            ):
                return True
        return False

    actual = {
        path.relative_to(ROOT).as_posix() for path in (ROOT / "tests/test_rs_kernel").glob("test_*.py") if requests_nv6(path)
    }
    assert actual == explicit_quadrature_suites


def test_random_mesh_rotation_is_haar_on_so3():
    """MCMC/LRDMC implement the prescribed Haar map on the original fixed keys."""
    from jqmc._setting import (
        ATOL_PYTHON_FP64_LOOSE,
        RTOL_PYTHON_FP64_LOOSE,
    )
    from jqmc.jqmc_gfmc import _sample_lrdmc_rotation_matrix
    from jqmc.jqmc_mcmc import _generate_rotation_matrix

    mcmc_key = jax.random.PRNGKey(34456)
    mcmc_rt = np.asarray(_generate_rotation_matrix(mcmc_key))
    lrdmc_r = np.asarray(_sample_lrdmc_rotation_matrix(jax.random.split(mcmc_key, 2)[1], True))
    np.testing.assert_allclose(
        mcmc_rt,
        lrdmc_r.T,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_LOOSE,
    )

    keys = jax.random.split(jax.random.PRNGKey(20260901), 8192)
    uniforms = np.asarray(jax.vmap(lambda key: jax.random.uniform(key, shape=(3,)))(keys))
    rotations = np.asarray(jax.vmap(lambda key: _sample_lrdmc_rotation_matrix(key, True))(keys))
    independent = _numpy_haar_rz_ry_rx(uniforms[:64])
    np.testing.assert_allclose(
        rotations[:64],
        independent,
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_LOOSE,
    )
    np.testing.assert_allclose(
        rotations[:64, 2, 0],
        -(2.0 * uniforms[:64, 1] - 1.0),
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_LOOSE,
    )

    identities = np.swapaxes(rotations[:64], 1, 2) @ rotations[:64]
    np.testing.assert_allclose(
        identities,
        np.broadcast_to(np.eye(3), identities.shape),
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_LOOSE,
    )
    np.testing.assert_allclose(
        np.linalg.det(rotations[:64]),
        np.ones(64),
        atol=ATOL_PYTHON_FP64_LOOSE,
        rtol=RTOL_PYTHON_FP64_LOOSE,
    )


def _fixed_rotation_grid(monkeypatch, driver, n_angle):
    """Feed product quadrature nodes through the actual driver sampling body."""
    from itertools import product

    import jax.numpy as jnp

    from jqmc import jqmc_gfmc, jqmc_mcmc

    # Equal angular nodes integrate Fourier modes through degree two exactly;
    # two-point Gauss--Legendre integrates the required powers of sin(beta).
    # This is finite quadrature, not a random empirical-moment acceptance band.
    angles = np.arange(n_angle) / n_angle
    sine_nodes = (1.0 + np.asarray([-1.0, 1.0]) / np.sqrt(3.0)) / 2.0
    uniforms = np.asarray(list(product(angles, sine_nodes, angles)))
    rotations = []
    key = jax.random.PRNGKey(0)
    with monkeypatch.context() as local:
        for uniform in uniforms:

            def fixed_uniform(key, shape, minval, maxval):
                assert shape == (3,) and minval == 0.0 and maxval == 1.0
                return jnp.asarray(uniform)

            local.setattr(jax.random, "uniform", fixed_uniform)
            if driver == "mcmc":
                # Keep the production body, including key split and transpose;
                # bypass only the JIT cache, which would capture a previous node.
                matrix = jqmc_mcmc._generate_rotation_matrix.__wrapped__(key).T
            else:
                matrix = jqmc_gfmc._sample_lrdmc_rotation_matrix(key, True)
            rotations.append(np.asarray(matrix))
    return uniforms, np.asarray(rotations)


def _compare_rotation_grid(uniforms, rotations):
    """Compare the full mapping and exact first/second moments on a finite grid."""
    from jqmc._precision import get_tolerance_min

    atol, rtol = get_tolerance_min(("coulomb",), "strict")
    axis = rotations[:, :, 0]
    values = [
        ("mapping", rotations, _numpy_haar_rz_ry_rx(uniforms)),
        ("axis-mean", axis.mean(axis=0), np.zeros(3)),
        ("axis-second-moment", axis.T @ axis / len(axis), np.eye(3) / 3.0),
    ]
    for quantity, actual, reference in values:
        np.testing.assert_allclose(actual, reference, atol=atol, rtol=rtol, equal_nan=False, err_msg=quantity)


@pytest.mark.parametrize("driver", ["mcmc", "gfmc"])
@pytest.mark.parametrize("n_angle", [4, 8])
def test_mesh_rotation_exact_product_quadrature(monkeypatch, driver, n_angle):
    """Both sampling bodies obey the Haar map and exact fixed-grid isotropy moments."""
    uniforms, rotations = _fixed_rotation_grid(monkeypatch, driver, n_angle)
    _compare_rotation_grid(uniforms, rotations)


@pytest.mark.parametrize("driver", ["mcmc", "gfmc"])
def test_mesh_rotation_rejects_actual_uniform_pitch(monkeypatch, driver):
    """The same fixed-grid reference rejects uniform pitch in the production body."""
    import inspect

    from jqmc import jqmc_gfmc, jqmc_mcmc

    uniforms, rotations = _fixed_rotation_grid(monkeypatch, driver, 4)
    _compare_rotation_grid(uniforms, rotations)
    module = jqmc_mcmc if driver == "mcmc" else jqmc_gfmc
    original = module._haar_rotation_matrix_from_uniform
    source = inspect.getsource(original)
    needle = "jnp.arcsin(2 * uniforms[1] - 1)"
    assert source.count(needle) == 1
    namespace = dict(original.__globals__)
    exec(compile(source.replace(needle, "4 * jnp.pi * uniforms[1] - 2 * jnp.pi"), "<uniform-pitch-fault>", "exec"), namespace)
    monkeypatch.setattr(module, original.__name__, namespace[original.__name__])
    uniforms, rotations = _fixed_rotation_grid(monkeypatch, driver, 4)
    with pytest.raises(AssertionError):
        _compare_rotation_grid(uniforms, rotations)
