"""Exact reviewed classifications of Rust boolean route predicates.

The source detector deliberately treats every ``*_ok`` boolean function as a
potential route gate.  Capability/performance routes therefore need a witness
for the slower-route behavior, while numerical-safety predicates are recorded
separately so their names cannot silently exempt them from review.
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

from typing import TypeAlias


PredicateKey: TypeAlias = tuple[str, str, str]
CounterWitness: TypeAlias = int | tuple[str, int]
RouteReview: TypeAlias = tuple[CounterWitness | None, str]

_GFMC = "jqmc/rs/src/gfmc.rs"

RUST_ROUTE_PREDICATES: dict[PredicateKey, RouteReview] = {
    ("jqmc/rs/src/atomic_orbital.rs", "fn", "kernel_sphe_soa_ok"): (
        None,
        "performance: point-major Sphe AO value kernel, bitwise identical to the shell loop it replaces "
        "(kernel_sphe_soa_matches_shell_eval_bitwise); decline keeps that shell loop",
    ),
    ("jqmc/rs/src/atomic_orbital.rs", "fn", "kernel_cart_soa_ok"): (
        None,
        "performance: point-major Cart AO value kernel, bitwise identical to the shell loop it replaces "
        "(kernel_cart_soa_matches_shell_loop_bitwise); decline keeps that shell loop",
    ),
    ("jqmc/rs/src/complex_stress.rs", "fn", "moved_group_ok"): (
        None,
        "performance: fused moved-electron pass of the complex ECP mesh force / strain candidates "
        "(mesh_vgs_multi_matches_derivative_kernel_tables; the force / stress parity tests of "
        "IMPL-RS-CPLX-ECP-FORCE-RANK1-001 and the complex stress guarantees); decline keeps the "
        "per-candidate rank-1 path",
    ),
    ("jqmc/rs/src/forces.rs", "fn", "kin_sweep_ok"): (
        None,
        "performance: fused orbital-side kinetic force sweep, equal to the bundle contraction it replaces to "
        "reduction order (kin_sweep_matches_bundle_contraction); decline keeps that bundle contraction",
    ),
    ("jqmc/rs/src/coulomb_potential.rs", "fn", "ecp_wdot_ok"): (
        ("predicate", 1),
        "decline takes the batched orb-mesh ECP path (route-predicate counter 1; 0 counts builds)",
    ),
    ("jqmc/rs/src/coulomb_potential.rs", "fn", "ecp_j3_wdot_ok"): (
        ("predicate", 3),
        "decline takes the batched J3 orb-mesh ECP path (route-predicate counter 3; 2 counts builds)",
    ),
    (_GFMC, "fn", "fast_mol_kinetic_ok_cell"): (1, "decline takes the reduced interval state (counter 2)"),
    (_GFMC, "fn", "fast_mol_kinetic_ok"): (1, "wrapper of fast_mol_kinetic_ok_cell"),
    (_GFMC, "fn", "molecular_candidate_ok"): (
        11,
        "only on the uncached per-group det-ratio route (random mesh, or after a counted [11] decline)",
    ),
    (_GFMC, "fn", "candidate_basis_class"): (11, "Unsupported class is the counted candidate slow route"),
    (_GFMC, "fn", "det_candidate_class"): (11, "Unsupported class is the counted candidate slow route"),
    (_GFMC, "fn", "j3_candidate_class"): (12, "Unsupported J3 class is the counted J3 slow fallback"),
    (_GFMC, "fn", "fast_generic_det_gradlap_cell"): (13, "basis-generic det grad/lap cache build counter"),
    (_GFMC, "fn", "fast_det_gradlap_ok_cell"): (41, "decline is the counted real det grad/lap slow route"),
    (_GFMC, "fn", "fast_det_gradlap_ok"): (41, "wrapper of fast_det_gradlap_ok_cell"),
    (_GFMC, "fn", "complex_candidate_class"): (21, "Unsupported class is the counted complex candidate slow route"),
    (_GFMC, "fn", "complex_det_gradlap_ok"): (30, "decline is the counted complex det derivative slow route"),
    (_GFMC, "fn", "complex_jaspart_gradlap_ok"): (
        37,
        "decline is the counted complex J1/J2 (33 for DTN) slow route",
    ),
    (_GFMC, "jas_ratios", "j3_fast"): (
        12,
        "only on the uncached per-group Jastrow-ratio route (random mesh, or after a counted [12] decline)",
    ),
    (_GFMC, "mesh_fast_state", "j3_state_ok"): (
        12,
        "no J3 streaming state is counted at the consumer (J3 slow); the MO-wrapped J3 decline is the "
        "declared slow route of test_gfmc_projection.py::test_declared_slow_route_is_taken_and_matches_xla",
    ),
    (_GFMC, "compute_discretized_kinetic_elements_fast", "det_up_fast"): (11, "counted candidate slow route"),
    (_GFMC, "compute_discretized_kinetic_elements_fast", "det_dn_fast"): (11, "counted candidate slow route"),
    (_GFMC, "compute_discretized_kinetic_elements_fast", "j3_fast"): (12, "counted J3 slow fallback"),
    (_GFMC, "compute_discretized_kinetic_elements_fast", "cache_ok"): (
        11,
        "conjunction of det/J3 above + fixed mesh",
    ),
    (_GFMC, "compute_ecp_nonlocal_mesh_fast", "det_up_fast"): (11, "counted candidate slow route"),
    (_GFMC, "compute_ecp_nonlocal_mesh_fast", "det_dn_fast"): (11, "counted candidate slow route"),
    (_GFMC, "compute_ecp_nonlocal_mesh_fast", "j3_fast"): (12, "counted J3 slow fallback"),
    (_GFMC, "compute_ecp_nonlocal_mesh_fast", "cache_ok"): (11, "conjunction of det/J3 above + fixed mesh"),
    (_GFMC, "discretized_kinetic_elements_complex_fast", "det_up_fast"): (
        21,
        "counted complex candidate slow route",
    ),
    (_GFMC, "discretized_kinetic_elements_complex_fast", "det_dn_fast"): (
        21,
        "counted complex candidate slow route",
    ),
    (_GFMC, "discretized_kinetic_elements_complex_fast", "j3_fast"): (22, "counted complex J3 slow fallback"),
    (_GFMC, "discretized_kinetic_elements_complex_fast", "cache_ok"): (
        21,
        "conjunction of det/J3 above + fixed mesh",
    ),
    (_GFMC, "compute_ecp_nonlocal_mesh_complex_fast", "det_up_fast"): (
        21,
        "counted complex candidate slow route",
    ),
    (_GFMC, "compute_ecp_nonlocal_mesh_complex_fast", "det_dn_fast"): (
        21,
        "counted complex candidate slow route",
    ),
    (_GFMC, "compute_ecp_nonlocal_mesh_complex_fast", "j3_fast"): (22, "counted complex J3 slow fallback"),
    (_GFMC, "compute_ecp_nonlocal_mesh_complex_fast", "cache_ok"): (
        21,
        "conjunction of det/J3 above + fixed mesh",
    ),
    (_GFMC, "compute_ecp_nonlocal_mesh_complex_fast", "det_up_wdot_ok"): (
        ("predicate", 17),
        "decline keeps the orbital-rank complex candidate table for the up spin "
        "(route-predicate counter 17; 16 counts AO-space w-dot spins)",
    ),
    (_GFMC, "compute_ecp_nonlocal_mesh_complex_fast", "det_dn_wdot_ok"): (
        ("predicate", 17),
        "decline keeps the orbital-rank complex candidate table for the dn spin "
        "(route-predicate counter 17; 16 counts AO-space w-dot spins)",
    ),
    ("jqmc/rs/src/atomic_orbital.rs", "fn", "bloch_mesh_dot_ok"): (
        ("predicate", 17),
        "gto-block coverage of the complex Bloch mesh dot, consumed only by complex_det::mesh_wdot_ok: decline keeps "
        "the orbital-rank candidate table (route-predicate counter 17; 16 counts engaged spins)",
    ),
    ("jqmc/rs/src/complex_det.rs", "fn", "mesh_wdot_ok"): (
        ("predicate", 17),
        "complex leaf gate of the FAST ECP mesh AO-space w-dot: decline keeps the orbital-rank candidate table "
        "(route-predicate counter 17; 16 counts engaged spins)",
    ),
    ("jqmc/rs/src/jastrow_factor.rs", "fn", "j12_candidate_cache_ok"): (
        51,
        "J1/J2 kinetic table fallback counter",
    ),
    ("jqmc/rs/src/jastrow_factor.rs", "fn", "j12_generic_cache_ok"): (
        42,
        "decline is the counted J1/J2 grad/lap slow route",
    ),
    ("jqmc/rs/src/lib.rs", "run_energy_chunk_real_body", "j3_carry_ok"): (
        ("predicate", 7),
        "decline rebuilds the MCMC chunk J3 ratio state per step (route-predicate counter 7; 6 counts carries)",
    ),
    ("jqmc/rs/src/param_grad_ecp.rs", "fn", "sm_ratio_ok"): (
        ("predicate", 9),
        "decline takes the fresh per-point rebuild of an ECP parameter-gradient mesh candidate instead of the SM "
        "increment (route-predicate counter 9; 8 counts SM increments)",
    ),
    ("jqmc/rs/src/lib.rs", "run_energy_chunk_real_body", "det_carry_ok"): (
        ("predicate", 5),
        "decline rebuilds the MCMC chunk determinant state per step (route-predicate counter 5; 4 counts carries)",
    ),
    ("jqmc/rs/src/lib.rs", "run_energy_chunk_complex_body", "kin_carry_ok"): (
        ("predicate", 23),
        "decline keeps the fresh per-e_L complex kinetic (compute_kinetic_withj_complex_with_inv) instead of the "
        "chunk-carried derivative tables (route-predicate counter 23; 22 counts carried e_L)",
    ),
    ("jqmc/rs/src/lib.rs", "run_energy_chunk_complex_body", "det_base_carry_ok"): (
        ("predicate", 28),
        "decline rebuilds the complex chunk's geminal, inverse and DetRatioStateC every step (route-predicate "
        "counter 28; 27 counts carried steps)",
    ),
    (_GFMC, "screen", "kin_ginv_carry_ok"): (
        ("predicate", 26),
        "Per-hop screen of the carried complex kinetic tables' own inverse: decline re-seeds every table at the "
        "current positions (route-predicate counter 26 = re-seeds; 24 counts all seeds)",
    ),
    ("jqmc/rs/src/determinant.rs", "fn", "interval_ginv_carry_ok"): (
        ("predicate", 11),
        "Per-hop interval-inverse screen: decline rebuilds the carried G^-1 (and the winv tables) from "
        "scratch at the post-hop positions (route-predicate counter 11 = refreshes; 10 counts kept carries)",
    ),
}

RUST_NUMERIC_SAFETY_PREDICATES: dict[PredicateKey, str] = {
    ("jqmc/rs/src/determinant.rs", "fn", "as_linear_ok"): (
        "numerical-safety: selects direct linear arithmetic only while it is finite and well-scaled; otherwise the "
        "same AS factor is evaluated by the native scaled/log-domain formula"
    )
}
