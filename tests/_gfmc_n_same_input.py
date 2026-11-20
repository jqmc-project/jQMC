"""Test-only observations and same-input references for the load-balanced driver.

The observer wraps actual calls; it does not execute an AST copy of production.
No trace hook replaces GDD's own tracing. All patches are scoped by the caller.
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

import ast
import copy
from functools import partial, wraps
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from jqmc import jqmc_gfmc as implementation
from jqmc.coulomb_potential import (
    compute_coulomb_potential_ee_dispatch,
    compute_coulomb_potential_ei_dispatch,
    compute_coulomb_potential_ei_discretized_dispatch,
    compute_coulomb_potential_ii_dispatch,
    compute_ecp_local_parts_all_pairs,
    compute_ecp_non_local_parts_nearest_neighbors,
)
from jqmc.determinant import compute_geminal_all_elements
from jqmc.wavefunction import compute_Jastrow_part, compute_discretized_kinetic_energy, compute_kinetic_energy_all_elements


_CORE = "GFMC_n.run.<locals>._projection_n.<locals>._body_step_core"
_HOP_FIELDS = (
    "i",
    "r_up_carts",
    "r_dn_carts",
    "A_old_inv",
    "move_key",
    "random_value",
    "cdf",
    "k",
    "p_list",
    "b_x",
    "w_L",
    "diagonal_sum_hamiltonian",
    "non_diagonal_sum_hamiltonian",
    "RT",
    "ecp_RT",
    "next_up",
    "next_dn",
)


class ObserveGFMCN:
    """Record all real projection substeps and pre-reconfiguration observables.

    Missing/changed caller bindings fail instead of producing empty evidence.
    The science test separately compares the instrumented and uninstrumented
    driver outputs (``observer_neutrality_split``): trajectories, keys, the
    inverse and bookkeeping exactly, floating observables within policy; an
    observer must not change the tested trajectory.
    """

    def __init__(self, driver):
        self.driver = driver
        # The public setter creates a derivative-mask copy during construction.
        # Pin that initialized object before run(), not its caller's object id.
        self.hamiltonian = driver.hamiltonian_data
        self.intervals, self.branches, self.pending, self.reconfigurations = [], [], [], []
        self.initial = tuple(
            np.array(value, copy=True)
            for value in (
                driver._latest_r_up_carts,
                driver._latest_r_dn_carts,
                driver._jax_PRNG_key_list,
            )
        )
        self.branch_rng_state = copy.deepcopy(driver._branching_rng.get_state())
        tree = ast.parse(Path(implementation.__file__).read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "GFMC_n")
        run = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        sites = [
            node.lineno
            for node in ast.walk(run)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "bcast"
            and node.value.args
            and isinstance(node.value.args[0], ast.Name)
            and node.value.args[0].id == "zeta"
        ]
        if len(sites) != 1:
            raise AssertionError("the GFMC_n branching observation must bind exactly once")
        self.branch_line = sites[0]

    def _receive_hop(self, *values):
        self.pending.append({name: np.array(value, copy=True) for name, value in zip(_HOP_FIELDS, values, strict=True)})

    def install(self, monkeypatch):
        original_sort, original_jit, original_numpy_sort = jnp.searchsorted, implementation.jit, np.searchsorted
        observer = self

        def searchsorted(cdf, value, *args, **kwargs):
            selected = original_sort(cdf, value, *args, **kwargs)
            frame = sys._getframe(1)
            if frame.f_code.co_qualname == _CORE and frame.f_code.co_filename == implementation.__file__:
                local = frame.f_locals
                data = dict(
                    local,
                    cdf=cdf,
                    random_value=value,
                    k=selected,
                    RT=local["R"].T,
                    ecp_RT=local["ecp_R"].T,
                    next_up=local["non_diagonal_move_mesh_r_up_carts"][selected],
                    next_dn=local["non_diagonal_move_mesh_r_dn_carts"][selected],
                )
                jax.debug.callback(observer._receive_hop, *(data[name] for name in _HOP_FIELDS))
            return selected

        def branching_searchsorted(cdf, value, *args, **kwargs):
            selected = original_numpy_sort(cdf, value, *args, **kwargs)
            frame = sys._getframe(1)
            if frame.f_code is implementation.GFMC_n.run.__code__:
                local = frame.f_locals
                assert local["self"] is observer.driver and cdf is local["global_cumprob"]
                observer.reconfigurations.append(
                    {
                        "step": local["i_mcmc_step"],
                        "cdf": np.array(cdf, copy=True),
                        "draws": np.array(value, copy=True),
                        "parents": np.array(selected, copy=True),
                    }
                )
            return selected

        def traced_jit(fun=None, **kwargs):
            if fun is None:
                return partial(traced_jit, **kwargs)
            compiled = original_jit(fun, **kwargs)
            if getattr(fun, "__name__", None) != "_projection_n":
                return compiled

            @wraps(compiled)
            def projection(*args, **kwargs):
                frame = sys._getframe(1)
                if frame.f_code is not implementation.GFMC_n.run.__code__:
                    return compiled(*args, **kwargs)
                assert frame.f_locals["self"] is observer.driver
                jax.effects_barrier()
                assert not observer.pending, "unclaimed projection observations"
                result = compiled(*args, **kwargs)
                jax.block_until_ready(result)
                jax.effects_barrier()
                hops, observer.pending = observer.pending, []
                if "i_mcmc_step" in frame.f_locals:
                    observer.intervals.append(
                        {
                            "step": frame.f_locals["i_mcmc_step"],
                            "inputs": jax.tree_util.tree_map(lambda x: np.array(x, copy=True), args[:6]),
                            "settings": args[6:10],
                            "ham": args[10],
                            "hops": hops,
                            "outputs": jax.tree_util.tree_map(lambda x: np.array(x, copy=True), result[:9]),
                        }
                    )
                else:
                    assert len(hops) == len(args[0]) * args[6], "warmup observer did not execute every hop"
                return result

            return projection

        class Communication:
            def bcast(self, value, *args, **kwargs):
                result = comm.bcast(value, *args, **kwargs)
                frame = sys._getframe(1)
                if frame.f_code is implementation.GFMC_n.run.__code__ and frame.f_lineno == observer.branch_line:
                    local = frame.f_locals
                    assert local["self"] is observer.driver
                    names = (
                        "w_L_list",
                        "V_diag_list",
                        "e_L_list",
                        "latest_RTs",
                        "latest_kinetic_RTs",
                        "grad_e_L_R",
                        "grad_e_L_r_up",
                        "grad_e_L_r_dn",
                        "grad_ln_Psi_dR",
                        "grad_ln_Psi_r_up",
                        "grad_ln_Psi_r_dn",
                    )
                    row = {name: np.array(local[name], copy=True) for name in names}
                    row.update(
                        step=local["i_mcmc_step"],
                        zeta=result,
                        E_scf=observer.driver._E_scf,
                        up=np.array(observer.driver._latest_r_up_carts, copy=True),
                        down=np.array(observer.driver._latest_r_dn_carts, copy=True),
                    )
                    observer.branches.append(row)
                return result

            def __getattr__(self, name):
                return getattr(comm, name)

        comm = self.driver._mpi_comm
        monkeypatch.setattr(jnp, "searchsorted", searchsorted)
        monkeypatch.setattr(np, "searchsorted", branching_searchsorted)
        monkeypatch.setattr(implementation, "jit", traced_jit)
        monkeypatch.setattr(self.driver, "_mpi_comm", Communication())

    def population_payload(self, assembled):
        """Gather only population inputs/outputs, not Hamiltonians or hop tapes."""
        return {
            "initial": self.initial,
            "branch_rng_state": self.branch_rng_state,
            "intervals": [{name: row[name] for name in ("step", "inputs", "outputs")} for row in self.intervals],
            "branches": self.branches,
            "reconfigurations": self.reconfigurations,
            "assembled": assembled,
            "driver_fields": driver_state(self.driver),
        }


def driver_state(driver):
    """All observable buffers and persistent numerical sampler state, not timers."""
    names = [name for name in vars(driver) if name.startswith("_stored_")]
    names.extend(
        (
            "_latest_r_up_carts",
            "_latest_r_dn_carts",
            "_latest_gfmc_pot_updates",
            "_jax_PRNG_key_list",
            "_GFMC_n__latest_A_old_inv",
            "_G_L",
            "_G_e_L",
            "_E_scf",
            "_mcmc_counter",
            "_epsilon_PW",
            "_use_swct",
            "_num_gfmc_collect_steps",
        )
    )
    return {name: np.array(getattr(driver, name), copy=True) for name in names}


# Observer neutrality scope.
# The instrumented driver compiles different HLO (a debug callback inside the
# traced projection), and XLA from jax 0.7.0 then rounds observable
# accumulators differently at the last bit. Walker trajectories, PRNG keys, the
# carried inverse and the integer/flag bookkeeping must stay bit-identical; the
# floating observable accumulators are judged with the node's existing
# within-mode energy and force policies. Every driver_state field must belong to
# exactly one set, so a new buffer cannot silently escape the check.
OBSERVER_BITWISE_FIELDS = frozenset(
    (
        "_latest_r_up_carts",
        "_latest_r_dn_carts",
        "_jax_PRNG_key_list",
        "_GFMC_n__latest_A_old_inv",
        "_latest_gfmc_pot_updates",
        "_stored_average_projection_counter",
        "_mcmc_counter",
        "_epsilon_PW",
        "_use_swct",
        "_num_gfmc_collect_steps",
    )
)
OBSERVER_ENERGY_FIELDS = frozenset(("_stored_w_L", "_stored_e_L", "_stored_e_L2", "_stored_mpc_dV", "_G_L", "_G_e_L", "_E_scf"))
OBSERVER_FORCE_FIELDS = frozenset(
    (
        "_stored_force_HF",
        "_stored_force_PP",
        "_stored_E_L_force_PP",
        "_stored_stress_HF",
        "_stored_stress_PP",
        "_stored_E_L_stress_PP",
    )
)


def observer_neutrality_split(actual_state, baseline_state):
    """Split observed/unobserved driver states into bitwise names and observable pairs.

    Returns ``(bitwise_names, (energy_observed, energy_unobserved),
    (force_observed, force_unobserved))``. Observable pairs are the fields'
    flattened concatenations in sorted name order. A field outside the three
    reviewed sets, or a shape change between the two runs, raises.
    """
    if actual_state.keys() != baseline_state.keys():
        raise ValueError("observed and unobserved driver states expose different fields")
    names = set(actual_state)
    unknown = names - OBSERVER_BITWISE_FIELDS - OBSERVER_ENERGY_FIELDS - OBSERVER_FORCE_FIELDS
    if unknown:
        raise ValueError(f"driver_state fields without an observer-neutrality category: {sorted(unknown)}")

    def pair(selected):
        present = sorted(names & selected)
        for name in present:
            if np.shape(actual_state[name]) != np.shape(baseline_state[name]):
                raise ValueError(f"observer changed the shape of {name}")
        return tuple(
            np.concatenate(
                [np.ravel(np.asarray(state[name], dtype=np.result_type(state[name], np.float64))) for name in present]
            )
            for state in (actual_state, baseline_state)
        )

    return sorted(names & OBSERVER_BITWISE_FIELDS), pair(OBSERVER_ENERGY_FIELDS), pair(OBSERVER_FORCE_FIELDS)


def history_comparisons(driver):
    """Compare each public history with its independently sliced stored input."""
    fields = driver_state(driver)
    collect = driver._num_gfmc_collect_steps
    bare = fields["_stored_w_L"][:, 0]
    weights = np.array([np.prod(bare[i - collect : i]) for i in range(collect, len(bare))])[:, None]
    return {
        "weights": (np.asarray(driver.w_L), weights),
        **{
            name: (np.asarray(getattr(driver, name)), fields["_stored_" + name][collect:])
            for name in ("e_L", "e_L2", "force_HF", "force_PP", "E_L_force_PP")
        },
    }


@partial(jax.jit, static_argnames=("non_local_move",))
def leaf_reference(ham, up, down, kinetic_rt, ecp_rt, alat, *, non_local_move):
    """Reassemble full-recompute leaves without the projection/update kernels.

    This is independent of streaming caches and projection assembly, not an
    independent implementation of every Hamiltonian primitive. AE cutoffs and
    ECP tmove/dltmove are kept distinct, as are propagation and measurement.
    """
    wf, cd = ham.wavefunction_data, ham.coulomb_potential_data
    mesh_up, mesh_dn, hops = compute_discretized_kinetic_energy(alat, wf, up, down, kinetic_rt)
    hops = hops.reshape(-1, 6)
    continuum = jnp.concatenate(compute_kinetic_energy_all_elements(wf, up, down))
    bare = jnp.concatenate(compute_coulomb_potential_ei_dispatch(cd, up, down))
    lower = bare if cd.ecp_flag else jnp.concatenate(compute_coulomb_potential_ei_discretized_dispatch(cd, up, down, alat))
    regularized = bare + continuum - jnp.sum(hops + 1 / (2 * alat**2), axis=1)
    regularized = jnp.where(jnp.any(hops >= 0, axis=1), jnp.maximum(regularized, lower), regularized)
    pair = compute_coulomb_potential_ee_dispatch(cd, up, down) + compute_coulomb_potential_ii_dispatch(cd)
    diagonal = 3 * (len(up) + len(down)) / alat**2 + pair + jnp.sum(regularized) + jnp.sum(jnp.maximum(hops, 0))
    measurement = pair + jnp.sum(bare + continuum)
    off_diagonal = jnp.minimum(hops, 0).ravel()
    if cd.ecp_flag:
        local = compute_ecp_local_parts_all_pairs(cd, up, down)
        extra_up, extra_dn, nonlocal_values, _ = compute_ecp_non_local_parts_nearest_neighbors(
            cd, wf, up, down, ecp_rt, flag_determinant_only=non_local_move == "dltmove"
        )
        positive, negative = jnp.sum(jnp.maximum(nonlocal_values, 0)), jnp.minimum(nonlocal_values, 0).ravel()
        if non_local_move == "dltmove":
            initial = compute_Jastrow_part(wf.jastrow_data, up, down)
            proposed = jax.vmap(compute_Jastrow_part, in_axes=(None, 0, 0))(wf.jastrow_data, extra_up, extra_dn)
            negative = negative * jnp.exp(proposed - initial)
        elif non_local_move != "tmove":
            raise ValueError("unsupported nonlocal move in same-input reference")
        diagonal = diagonal + local + positive
        measurement = measurement + local + positive + jnp.sum(negative)
        off_diagonal = jnp.concatenate((off_diagonal, negative))
        mesh_up, mesh_dn = jnp.concatenate((mesh_up, extra_up)), jnp.concatenate((mesh_dn, extra_dn))
    probabilities = jnp.real(off_diagonal) / jnp.sum(jnp.real(off_diagonal))
    matrix = compute_geminal_all_elements(wf.geminal_data, up, down)
    return mesh_up, mesh_dn, jnp.real(off_diagonal), jnp.real(diagonal), measurement, probabilities, matrix


def ordered_hops(interval):
    """Match actual records to prescribed key schedules, rejecting omissions/duplicates."""
    count = interval["settings"][0]
    inputs = interval["inputs"]
    records = {}
    for row in interval["hops"]:
        token = tuple(int(value) for value in row["move_key"])
        if token in records:
            raise AssertionError("duplicate GFMC projection hop")
        records[token] = row
    result = []
    for walker, initial_key in enumerate(inputs[4]):
        key = jnp.asarray(initial_key)
        rows = []
        for step in range(count):
            key, rotation_key = jax.random.split(key)
            key, move_key = jax.random.split(key)
            token = tuple(int(value) for value in move_key)
            if token not in records:
                raise AssertionError("missing GFMC projection hop")
            row = records.pop(token)
            if int(row["i"]) != step:
                raise AssertionError("GFMC projection hop index changed")
            rows.append((row, np.array(key), np.array(rotation_key)))
        result.append(rows)
    if records:
        raise AssertionError("unexpected GFMC projection hop")
    return result


@partial(jax.jit, static_argnames=("non_local_move",))
def _batch_leaves(ham, up, down, kinetic_rt, ecp_rt, alat, *, non_local_move):
    return jax.vmap(lambda ru, rd, kr, er: leaf_reference(ham, ru, rd, kr, er, alat, non_local_move=non_local_move))(
        up, down, kinetic_rt, ecp_rt
    )


def _flat(values):
    return np.concatenate([np.asarray(value).ravel() for value in values])


@partial(jax.jit, static_argnames=("non_local_move", "use_swct"))
def _force_leaves(ham, up, down, rt, alat, *, non_local_move, use_swct):
    """Use the existing debug derivative backend, with fresh inverses per input.

    The derivative formulas are covered by their independent derivative
    guarantees. This driver test checks their inputs, batching, cached-inverse
    use and aggregation; it does not claim a second independent derivative.
    """
    from jqmc import derivatives
    from jqmc.derivatives import gfmc as gfmc_derivatives
    from jqmc.swct import evaluate_swct_domega, evaluate_swct_omega

    if not implementation._ANALYTIC_GRAD_ENABLED:
        raise AssertionError("same-input GFMC force reference requires the declared analytic driver backend")

    def walker(ru, rd, rotation):
        hf, hf_up, hf_dn = gfmc_derivatives.grad_e_L_n_wrt_positions(ham, ru, rd, rotation, alat, non_local_move=non_local_move)
        score = derivatives.compute_grad_ln_Psi(ham, ru, rd, {"opt_r": True, "opt_R": True}, analytic=True)
        raw = (hf, hf_up, hf_dn, score["R"], score["r"][0], score["r"][1])
        pp = score["R"]
        if use_swct:
            structure = ham.structure_data
            omega_up, omega_dn = evaluate_swct_omega(structure, ru), evaluate_swct_omega(structure, rd)
            hf = hf + omega_up @ hf_up + omega_dn @ hf_dn
            pp = (
                pp
                + omega_up @ score["r"][0]
                + omega_dn @ score["r"][1]
                + (evaluate_swct_domega(structure, ru) + evaluate_swct_domega(structure, rd)) / 2
            )
        distance = 1 / jnp.sqrt(jnp.sum(score["r"][0] ** 2) + jnp.sum(score["r"][1] ** 2))
        return raw, hf, pp, distance

    return jax.vmap(walker)(up, down, rt)


def projection_comparisons(
    intervals,
    *,
    expected_intervals,
    expected_walkers,
    expected_hops,
    expected_random_mesh,
    expected_non_local_move,
    expected_alat,
    expected_hamiltonian,
):
    """Return separate actual/reference arrays; no tolerance decision is hidden here.

    Each reference is evaluated at that real hop's input, not a separately
    advancing walk. Conditional inputs (the running weight and inverse) have
    their own independent checks and exact handoff requirements.
    """
    if len(intervals) != expected_intervals:
        raise AssertionError("missing or unexpected GFMC projection interval")
    pairs = {name: ([], []) for name in ("candidates", "cdf", "propagation", "weights", "inverse", "positions")}
    exact = []

    def add(name, actual, reference):
        pairs[name][0].append(np.asarray(actual))
        pairs[name][1].append(np.asarray(reference))

    for index, interval in enumerate(intervals):
        count, random_mesh, non_local, alat = interval["settings"]
        weight, up, down, inverse, keys, energy_shift = interval["inputs"]
        output = interval["outputs"]
        exact.extend(
            (
                interval["step"] == index,
                count == expected_hops,
                len(up) == expected_walkers,
                np.array_equal(weight, np.ones(expected_walkers)),
            )
        )
        if (
            count != expected_hops
            or len(up) != expected_walkers
            or random_mesh != expected_random_mesh
            or non_local != expected_non_local_move
            or alat != expected_alat
            or interval["ham"] is not expected_hamiltonian
        ):
            raise AssertionError("GFMC projection input domain was reduced")
        ordered = ordered_hops(interval)
        flat_rows = [row for walker in ordered for row, _, _ in walker]
        arguments = [np.asarray([row[name] for row in flat_rows]) for name in ("r_up_carts", "r_dn_carts", "RT", "ecp_RT")]
        reference = tuple(
            np.asarray(value) for value in _batch_leaves(interval["ham"], *arguments, alat, non_local_move=non_local)
        )
        mesh_up, mesh_down, candidates, diagonal, _, probabilities, matrices = reference
        for walker, rows in enumerate(ordered):
            previous_up, previous_down, previous_inverse, previous_weight = (
                up[walker],
                down[walker],
                inverse[walker],
                weight[walker],
            )
            for step, (row, next_key, rotation_key) in enumerate(rows):
                offset = walker * count + step
                exact.extend((np.array_equal(row["r_up_carts"], previous_up), np.array_equal(row["r_dn_carts"], previous_down)))
                if step == 0:
                    exact.append(np.array_equal(row["A_old_inv"], previous_inverse))
                    kinetic_key, ecp_key = implementation._split_lrdmc_rotation_keys(jnp.asarray(rotation_key))
                    expected_rotations = (
                        np.asarray(implementation._sample_lrdmc_rotation_matrix(kinetic_key, random_mesh).T),
                        np.asarray(implementation._sample_lrdmc_rotation_matrix(ecp_key, random_mesh).T),
                    )
                for observed, expected in zip((row["RT"], row["ecp_RT"]), expected_rotations, strict=True):
                    add("positions", observed, expected)
                actual_probability = row["p_list"] / np.sum(row["p_list"])
                add("candidates", row["p_list"], candidates[offset])
                add("cdf", row["cdf"], np.cumsum(probabilities[offset]))
                add("cdf", row["cdf"], np.cumsum(actual_probability))
                add("propagation", np.real(row["diagonal_sum_hamiltonian"]), diagonal[offset])
                add("propagation", np.real(row["non_diagonal_sum_hamiltonian"]), candidates[offset].sum())
                expected_factor = -candidates[offset].sum() / (diagonal[offset] - energy_shift)
                add("weights", row["b_x"], expected_factor)
                add("weights", row["w_L"], previous_weight * expected_factor)
                # This checks an independently rebuilt matrix, not the identity
                # of an inverse with itself. Failures are not conditioning-skipped.
                add("inverse", row["A_old_inv"], np.linalg.inv(matrices[offset]))
                expected_uniform = np.asarray(jax.random.uniform(jnp.asarray(row["move_key"]), minval=0.0, maxval=1.0))
                selected = int(np.searchsorted(row["cdf"], row["random_value"]))
                exact.extend(
                    (
                        np.array_equal(row["random_value"], expected_uniform),
                        int(row["k"]) == selected,
                        bool(np.all(np.isfinite(row["cdf"]))),
                        bool(np.all(np.diff(row["cdf"]) >= 0)),
                    )
                )
                if not 0 <= selected < len(mesh_up[offset]):
                    raise AssertionError("invalid GFMC categorical selection")
                add("positions", row["next_up"], mesh_up[offset, selected])
                add("positions", row["next_dn"], mesh_down[offset, selected])
                previous_up, previous_down, previous_weight = row["next_up"], row["next_dn"], row["w_L"]
            exact.extend(
                (
                    np.array_equal(output[0][walker], previous_weight),
                    np.array_equal(output[1][walker], previous_up),
                    np.array_equal(output[2][walker], previous_down),
                    np.array_equal(output[4][walker], next_key),
                )
            )
        for rows in (flat_rows,):
            exact.append(len(rows) == expected_walkers * expected_hops)
    result = {name: tuple(_flat(side) for side in sides) for name, sides in pairs.items()}
    result["exact"] = (np.asarray(exact, dtype=bool), np.ones(len(exact), dtype=bool))
    return result


_GRADIENT_FIELDS = (
    "grad_e_L_R",
    "grad_e_L_r_up",
    "grad_e_L_r_dn",
    "grad_ln_Psi_dR",
    "grad_ln_Psi_r_up",
    "grad_ln_Psi_r_dn",
)


def assemble_observed_forces(ham, branch, *, use_swct):
    """Assemble checked derivative leaves without the driver's fused SWCT dot.

    These are conditional-input references for aggregation, not replacements
    for the separately required fresh-inverse derivative comparisons.
    """
    from jqmc.swct import evaluate_swct_domega, evaluate_swct_omega

    hf, hf_up, hf_dn, pp, pp_up, pp_dn = (branch[name] for name in _GRADIENT_FIELDS)
    hf, pp = hf.copy(), pp.copy()
    if use_swct:
        for walker, (up, down) in enumerate(zip(branch["up"], branch["down"], strict=True)):
            for positions, hf_r, pp_r in ((up, hf_up[walker], pp_up[walker]), (down, hf_dn[walker], pp_dn[walker])):
                omega = np.asarray(evaluate_swct_omega(ham.structure_data, jnp.asarray(positions)))
                hf[walker] += omega @ hf_r
                pp[walker] += omega @ pp_r + np.asarray(evaluate_swct_domega(ham.structure_data, jnp.asarray(positions))) / 2
    distance = 1 / np.sqrt(np.sum(pp_up**2, axis=(1, 2)) + np.sum(pp_dn**2, axis=(1, 2)))
    return hf, pp, distance


def measurement_comparisons(intervals, branches, *, use_swct):
    """Check measurement inputs and leaves separately from their weighted sums."""
    if not intervals or len(intervals) != len(branches):
        raise AssertionError("missing GFMC measurement interval")
    pairs = {
        name: ([], [])
        for name in ("measurement_diagonal", "measurement_energy", "measurement_inverse", "force_leaves", "force_assembly")
    }
    exact, assembled = [], []
    for interval, branch in zip(intervals, branches, strict=True):
        outputs, ham = interval["outputs"], interval["ham"]
        _, _, non_local, alat = interval["settings"]
        exact.extend((branch["step"] == interval["step"], np.array_equal(branch["E_scf"], interval["inputs"][5])))
        for observed, reference in (
            (branch["w_L_list"], outputs[0]),
            (branch["up"], outputs[1]),
            (branch["down"], outputs[2]),
            (branch["latest_kinetic_RTs"], outputs[5]),
            (branch["latest_RTs"], outputs[6]),
        ):
            exact.append(np.array_equal(observed, reference))
        leaves = _batch_leaves(
            ham,
            branch["up"],
            branch["down"],
            branch["latest_kinetic_RTs"],
            branch["latest_RTs"],
            alat,
            non_local_move=non_local,
        )
        for name, actual, reference in (
            ("measurement_diagonal", np.real(branch["V_diag_list"]), leaves[3]),
            ("measurement_energy", branch["e_L_list"], leaves[4]),
            ("measurement_inverse", outputs[3], np.linalg.inv(np.asarray(leaves[6]))),
        ):
            pairs[name][0].append(actual)
            pairs[name][1].append(np.asarray(reference))
        raw, hf, pp, distance = _force_leaves(
            ham, branch["up"], branch["down"], branch["latest_RTs"], alat, non_local_move=non_local, use_swct=use_swct
        )
        for field, reference in zip(_GRADIENT_FIELDS, raw, strict=True):
            pairs["force_leaves"][0].append(branch[field])
            pairs["force_leaves"][1].append(np.asarray(reference))
        conditional = assemble_observed_forces(ham, branch, use_swct=use_swct)
        for actual, reference in zip(conditional, (hf, pp, distance), strict=True):
            pairs["force_assembly"][0].append(actual)
            pairs["force_assembly"][1].append(np.asarray(reference))
        assembled.append(conditional)
    result = {name: tuple(_flat(side) for side in sides) for name, sides in pairs.items()}
    result["exact"] = (np.asarray(exact, dtype=bool), np.ones(len(exact), dtype=bool))
    return result, assembled


def finite_sample_estimates(bare_weights, energy, energy_squared, hf, pp, energy_pp, *, collect, warmup, bins):
    """Direct leave-one-block-out arithmetic on fixed samples, not a confidence test.

    Retained indices are explicitly selected for each omitted block rather
    than using the production sum-minus-block and MPI reduction algorithms.
    """
    bare_weights = np.asarray(bare_weights).reshape(-1)
    weights = np.array([np.prod(bare_weights[i - collect : i]) for i in range(collect, len(bare_weights))])
    values = [np.asarray(x)[collect + warmup :] for x in (energy, energy_squared, hf, pp, energy_pp)]
    weights = weights[warmup:]
    if not 1 < bins <= len(weights) or any(len(x) != len(weights) for x in values):
        raise AssertionError("invalid fixed-sample jackknife domain")
    blocks = np.array_split(np.arange(len(weights)), bins)
    samples = []
    for omitted in range(bins):
        keep = np.concatenate([block for index, block in enumerate(blocks) if index != omitted])
        normalized = weights[keep] / weights[keep].sum()
        e, e2, fh, fp, ep = [np.tensordot(normalized, x[keep], axes=(0, 0)) for x in values]
        # Variance of a complex replicate uses the modulus square |<e_L>|^2
        # (doc/notes/error_analysis.md, "Complex observables"), never (Re<e_L>)^2.
        samples.append((np.real(e), np.real(e2) - np.abs(e) ** 2, np.real(-fh - 2 * (ep - e * fp))))
    e, variance, force = (np.asarray([row[i] for row in samples]) for i in range(3))

    def estimate(x):
        mean = x.mean(axis=0)
        return mean, np.sqrt((bins - 1) * np.mean((x - mean) ** 2, axis=0))

    return (*estimate(e), *estimate(variance)), estimate(force)


def population_comparisons(payloads, *, expected_intervals, expected_walkers, initial_energy_shift):
    """Check rank-wide branching, carries, moments and energy-shift feedback.

    Stored arrays are compared to independently reduced actual leaves. Those
    leaves must also pass ``measurement_comparisons``; neither check suffices
    by itself. Realized CDF decisions are exact, never proximity/5-sigma tests.
    """
    if not payloads:
        raise AssertionError("no GFMC ranks observed")
    for payload in payloads:
        if any(len(payload[name]) != expected_intervals for name in ("intervals", "branches", "reconfigurations", "assembled")):
            raise AssertionError("missing GFMC population observation")
    pairs = {
        name: ([], [])
        for name in ("branching_cdf", "moments_energy", "moments_energy_squared", "moments_force", "weights", "energy_shift")
    }
    exact, shift = [], initial_energy_shift
    rng = np.random.RandomState()
    rng.set_state(payloads[0]["branch_rng_state"])
    root = payloads[0]["driver_fields"]
    total = len(payloads) * expected_walkers
    for step in range(expected_intervals):
        branches = [p["branches"][step] for p in payloads]
        bare = np.concatenate([b["w_L_list"] for b in branches])
        up, down = (np.concatenate([b[name] for b in branches]) for name in ("up", "down"))
        if len(bare) != total:
            raise AssertionError("GFMC population size changed")
        normalized = np.cumsum(bare / bare.sum())
        zeta = rng.random()
        for rank, payload in enumerate(payloads):
            interval, branch, selection = (payload[name][step] for name in ("intervals", "branches", "reconfigurations"))
            exact.extend((branch["step"] == step, selection["step"] == step, branch["zeta"] == zeta))
            if step == 0:
                exact.extend(
                    np.array_equal(a, b)
                    for a, b in zip(
                        (interval["inputs"][1], interval["inputs"][2], interval["inputs"][4]), payload["initial"], strict=True
                    )
                )
            draws = (np.arange(rank * expected_walkers, (rank + 1) * expected_walkers) + zeta) / total
            parents = np.searchsorted(selection["cdf"], selection["draws"])
            exact.extend(
                (
                    np.array_equal(selection["draws"], draws),
                    np.array_equal(selection["parents"], parents),
                    bool(np.all(np.isfinite(selection["cdf"]))),
                    bool(np.all(np.diff(selection["cdf"]) >= 0)),
                )
            )
            pairs["branching_cdf"][0].append(selection["cdf"])
            pairs["branching_cdf"][1].append(normalized)
            if np.any(parents < 0) or np.any(parents >= total):
                raise AssertionError("invalid population parent")
            if step + 1 < expected_intervals:
                following = payload["intervals"][step + 1]["inputs"]
                next_up, next_down, next_keys = following[1], following[2], following[4]
            else:
                final = payload["driver_fields"]
                next_up, next_down, next_keys = (
                    final["_latest_r_up_carts"],
                    final["_latest_r_dn_carts"],
                    final["_jax_PRNG_key_list"],
                )
            exact.extend(
                (
                    np.array_equal(next_up, up[parents]),
                    np.array_equal(next_down, down[parents]),
                    np.array_equal(next_keys, interval["outputs"][4]),
                )
            )
            pairs["energy_shift"][0].append(branch["E_scf"])
            pairs["energy_shift"][1].append(shift)
        denominator = np.concatenate([np.real(b["V_diag_list"]) - b["E_scf"] for b in branches])
        energy = np.concatenate([b["e_L_list"] for b in branches])
        weights = bare / denominator
        norm = weights.sum()
        hf, pp, distance = (np.concatenate([p["assembled"][step][i] for p in payloads]) for i in range(3))
        epsilons = np.concatenate([np.full(expected_walkers, p["driver_fields"]["_epsilon_PW"]) for p in payloads])
        factor = np.ones_like(distance)
        use = epsilons > 0
        t = distance[use] / epsilons[use]
        factor[use] = np.where(t < 1, 7 * t**6 - 15 * t**4 + 9 * t**2, 1)
        expected = {
            "_stored_w_L": bare.mean(),
            "_stored_e_L": np.dot(weights, energy) / norm,
            "_stored_e_L2": np.dot(weights, np.abs(energy) ** 2) / norm,
            "_stored_force_HF": np.tensordot(weights * factor, hf, axes=(0, 0)) / norm,
            "_stored_force_PP": np.tensordot(weights * factor, pp, axes=(0, 0)) / norm,
            "_stored_E_L_force_PP": np.tensordot(weights * factor * energy, pp, axes=(0, 0)) / norm,
        }
        for name, reference in expected.items():
            group = (
                "moments_force"
                if "force" in name
                else {"_stored_w_L": "weights", "_stored_e_L": "moments_energy", "_stored_e_L2": "moments_energy_squared"}[name]
            )
            pairs[group][0].append(root[name][step])
            pairs[group][1].append(reference)
        # Fixed on-the-fly feedback protocol (distinct from accessor collect).
        collect, bins = implementation.GFMC_ON_THE_FLY_COLLECT_STEPS, implementation.GFMC_ON_THE_FLY_BIN_BLOCKS
        count = step - collect + 1
        cadence = max(expected_intervals // 100, 1)
        if count >= bins and (step < cadence or (step + 1) % cadence == 0):
            warmup = min(collect + bins, max(count - bins, 0))
            bare_history = np.asarray(root["_stored_w_L"])[: step + 1].reshape(-1)
            energy_history = np.asarray(root["_stored_e_L"])[: step + 1].reshape(-1)
            products = np.array([np.prod(bare_history[i - collect : i]) for i in range(collect, step + 1)])[warmup:]
            energies = energy_history[collect + warmup :]
            blocks = np.array_split(np.arange(len(products)), bins)
            estimates = []
            for omit in range(bins):
                keep = np.concatenate([x for index, x in enumerate(blocks) if index != omit])
                estimates.append(np.dot(products[keep], energies[keep]) / products[keep].sum())
            shift = float(np.real(np.mean(estimates)))
    for payload in payloads:
        pairs["energy_shift"][0].append(payload["driver_fields"]["_E_scf"])
        pairs["energy_shift"][1].append(shift)
    result = {name: tuple(_flat(side) for side in sides) for name, sides in pairs.items()}
    result["exact"] = (np.asarray(exact, dtype=bool), np.ones(len(exact), dtype=bool))
    return result
