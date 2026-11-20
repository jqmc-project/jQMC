"""Unified analytic-production guard: IMPLEMENTED, and implemented WITHOUT generic autodiff.

WHAT THIS GUARDS (two failure modes, one sweep)
-----------------------------------------------
The production derivative contract is: every
supported (combination, quantity, axis) runs ANALYTIC kernels; an unsupported one
raises ``NotImplementedError`` at trace time; there is NO silent ``_auto`` fallback.
That contract can break two ways:

  1. a supported cell starts RAISING ``NotImplementedError`` (coverage regression), or
  2. a cell keeps running but quietly routes through ``jax.grad`` / ``hessian`` /
     ``jacrev`` -- "analytic" in name only. This mode is invisible to raise-detection,
     and the analytic-vs-``_auto`` correctness matrix cannot see it either: when the
     "analytic" route itself dispatches to ``_auto``, both sides of the comparison run
     the identical code and ``a == b`` holds trivially.

This module probes the full cartesian product

    (AO family) x (geminal) x (Jastrow) x (BC) x (potential) x (quantity) x (axis)

with tiny fixture-free DUMMY Hamiltonians and classifies every cell:

  * ``IMPLEMENTED``      -- runs clean, no generic-AD transform fired inside jqmc
  * ``SILENT_AUTODIFF``  -- runs, but a generic-AD transform fired (file:line reported)
  * ``NOT_IMPLEMENTED``  -- raises ``NotImplementedError``
  * ``OTHER_ERROR``      -- any other exception (a dummy-builder bug, not coverage data)

Only ``IMPLEMENTED`` passes. There are NO xfail escapes: a known offender is a
failing test. The Rust closed forms are the reference implementations for the
analytic forms: ``jqmc/rs/src/jastrow_factor.rs``
(``smooth_dist`` / ``smooth_dist_hess``) and ``gfmc_forces::swct_domega``.

HOW AD IS DETECTED MECHANICALLY
-------------------------------
``grad``/``hessian``/``jacrev``/... are trace-time Python calls: by the time a jaxpr
exists the differentiation has been expanded into ordinary primitives (verified --
jaxpr inspection cannot work). But the transform call itself always executes during
tracing, so we spy-wrap the public AD transforms on the ``jax`` module AND rebind
every ``from jax import grad``-style alias in every loaded ``jqmc.*`` module, then
run the probe and require the spy never fired from inside ``jqmc``. Deliberate
``custom_vjp`` / ``custom_jvp`` rules are exempt by construction (only the generic
transforms are wrapped); ``_auto`` oracles merely existing in a module never trigger
it -- only an executed dispatch does. ``jax.clear_caches()`` runs once per CELL so a
jit-cache hit from an earlier test with the same signature can never skip the trace
and blind the spy (within a cell every (quantity, axis) signature is distinct).

BEYOND THE CARTESIAN PRODUCT
----------------------------
The BC axis of the product is molecular-only until the periodic dummy builders land
(EXPANSION HOOKS below), so the known periodic offenders are probed directly:
PBC J1/J2 grad/lap on the e_L kinetic path (``compute_local_energy_fast ->
compute_kinetic_energy_all_elements_fast_update -> compute_grads_and_laplacian_
Jastrow_part`` dispatches to ``_auto`` when ``pbc_flag``) and ``evaluate_swct_domega``
(``jacrev`` inside the analytic HF-force assembly). These are plain RED until fixed.

=====================================================================================
EXPANSION HOOKS -- what each future axis needs
=====================================================================================
  * BC ``pbc_gamma`` / ``twist`` (``_build_structure`` / ``_build_coulomb``): periodic
    ``Structure_data.build`` + periodic AO leaves (pgto / complex-Bloch); Ewald in the
    Coulomb block. Landing this axis absorbs the direct PBC probes below into the
    product.
  * AO family ``plane_wave`` / ``bspline`` / ``composite`` (``_build_aos``): dummy
    G-vector table / knot grid instead of the GTO ``_from_full_basis`` path.
  * geminal ``AGP`` / ``AGPn`` (``_build_geminal``); potential ``ECP``
    (``_build_coulomb`` with dummy z_cores / ang_moms / exponents / powers).
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

import itertools
import sys
import traceback
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Fixture-free (dummy hams in memory, no committed h5/QE/trexio read): no CWD isolation.
pytestmark = [pytest.mark.light, pytest.mark.without_chdir]

from jqmc import derivatives
from jqmc.atomic_orbital import AOs_cart_gto_data, AOs_sphe_gto_data
from jqmc.coulomb_potential import Coulomb_potential_data
from jqmc.determinant import SD_data
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import (
    Jastrow_data,
    Jastrow_DTN_data,
    Jastrow_one_body_data,
    Jastrow_three_body_data,
    Jastrow_two_body_data,
)
from jqmc.jastrow_factor import (
    compute_grads_and_laplacian_Jastrow_one_body,
    compute_grads_and_laplacian_Jastrow_two_body,
)
from jqmc.molecular_orbital import MOs_data
from jqmc.structure import Structure_data
from jqmc.swct import evaluate_swct_domega, evaluate_swct_omega
from jqmc.wavefunction import Wavefunction_data

# --------------------------------------------------------------------------- #
# The AD spy.
# --------------------------------------------------------------------------- #
#: Public generic-AD transforms that must never fire while a production path runs.
#: ``custom_vjp`` / ``custom_jvp`` are deliberately NOT listed: registering an
#: analytic rule is the sanctioned mechanism, not a violation.
_AD_TRANSFORMS = ("grad", "value_and_grad", "jacrev", "jacfwd", "hessian", "vjp", "jvp", "linearize")


class _ADSpy:
    """Record every generic-AD transform call originating inside ``jqmc``."""

    def __init__(self):
        self.calls: list[str] = []

    def install(self, monkeypatch):
        originals = {n: getattr(jax, n) for n in _AD_TRANSFORMS}

        def make_spy(name, orig):
            def spy(*args, **kwargs):
                for fr in reversed(traceback.extract_stack()[:-1]):
                    if "/jqmc/" in fr.filename and "site-packages" not in fr.filename:
                        rel = fr.filename.split("/jqmc/")[-1]
                        self.calls.append(f"{name} @ jqmc/{rel}:{fr.lineno} in {fr.name}")
                        break
                return orig(*args, **kwargs)

            return spy

        spies = {n: make_spy(n, o) for n, o in originals.items()}
        # Rebind ``from jax import grad``-style aliases in every loaded jqmc module
        # (patching ``jax.grad`` alone would miss them), then the jax module itself
        # (covers ``jax.grad`` attribute-style call sites).
        for mod_name, mod in list(sys.modules.items()):
            if not (mod_name == "jqmc" or mod_name.startswith("jqmc.")):
                continue
            if not isinstance(mod, types.ModuleType):
                continue
            for n in _AD_TRANSFORMS:
                if getattr(mod, n, None) is originals[n]:
                    monkeypatch.setattr(mod, n, spies[n])
        for n in _AD_TRANSFORMS:
            monkeypatch.setattr(jax, n, spies[n])
        return self

    def assert_clean(self, label: str):
        offenders = sorted(set(self.calls))
        assert not offenders, (
            f"generic autodiff fired inside a PRODUCTION path [{label}]:\n  "
            + "\n  ".join(offenders)
            + "\nProduction kernels must be analytic; "
            "port the analytic form (the Rust closed forms in jqmc/rs/src are the reference)."
        )


@pytest.fixture
def ad_spy(monkeypatch):
    return _ADSpy().install(monkeypatch)


# --------------------------------------------------------------------------- #
# Prototype universe axes. Extend these lists (and the matching builder branch)
# per the EXPANSION HOOKS in the module docstring.
# --------------------------------------------------------------------------- #
_AO_FAMILIES = ("cart_gto", "sphe_gto")  # HOOK: + plane_wave / bspline / composite (periodic leaves)
_GEMINALS = ("SD",)  # HOOK: + AGP (dense-AO-AGP) / AGPn
_JASTROWS = ("none", "J1", "J2", "J3poly", "DTN")
_BCS = ("mol",)  # HOOK: + pbc_gamma / twist
_POTENTIALS = ("AE",)  # HOOK: + ECP

_QUANTITIES = ("ln_Psi", "e_L")
_AXES = ("c", "r", "R", "eps")

# c-axis mask: every parameter block ON except the two geminal radial-basis blocks
# (``opt_lambda_basis_*``), which ``with_param_grad_mask`` makes mutually exclusive with
# ``opt_geminal`` (freezing the MO coefficients). Mirrors ``_C_MASK`` in the 8-set harness.
_C_MASK = {k: ("lambda_basis" not in k) for k in derivatives._PARAM_FLAG_KEYS}


def _flags_for_axis(axis: str) -> dict:
    """Unified ``opt_flags`` selecting exactly one target axis."""
    if axis == "c":
        return dict(_C_MASK)
    return {f"opt_{axis}": True}


# --------------------------------------------------------------------------- #
# Fixture-free DUMMY builders. Values are arbitrary (physicality / conditioning are
# IRRELEVANT -- there is no _auto compare). Only SHAPES must be consistent so the
# analytic call reaches the type-dispatch point.
# --------------------------------------------------------------------------- #
_N_ELEC_UP = 1
_N_ELEC_DN = 1


def _build_structure(bc: str) -> Structure_data:
    """A minimal molecular ``Structure_data`` (2 distinct H-like centers)."""
    if bc != "mol":  # pragma: no cover - expansion hook
        raise NotImplementedError(f"BC {bc!r} not in prototype scope (see EXPANSION HOOKS)")
    return Structure_data(
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]),
        pbc_flag=False,
        atomic_numbers=(1, 1),
        element_symbols=("H", "H"),
        atomic_labels=("H1", "H2"),
    )


def _build_aos(ao_family: str, structure: Structure_data):
    """A 2-AO (one s-orbital per atom) contracted GTO leaf, cart or sphe."""
    common = dict(
        structure_data=structure,
        nucleus_index=(0, 1),
        num_ao=2,
        num_ao_prim=2,
        orbital_indices=(0, 1),
        exponents=np.array([1.2, 0.9], dtype=np.float64),
        coefficients=np.array([1.0, 1.0], dtype=np.float64),
        angular_momentums=(0, 0),
    )
    if ao_family == "cart_gto":
        aos = AOs_cart_gto_data._from_full_basis(
            polynominal_order_x=(0, 0), polynominal_order_y=(0, 0), polynominal_order_z=(0, 0), **common
        )
    elif ao_family == "sphe_gto":
        aos = AOs_sphe_gto_data._from_full_basis(magnetic_quantum_numbers=(0, 0), **common)
    else:  # pragma: no cover - expansion hook
        raise NotImplementedError(f"AO family {ao_family!r} not in prototype scope (see EXPANSION HOOKS)")
    aos.sanity_check()
    return aos


def _build_geminal(geminal: str, aos):
    """An occupied-MO ``SD_data`` over a small deterministic MO block."""
    if geminal != "SD":  # pragma: no cover - expansion hook
        raise NotImplementedError(f"geminal {geminal!r} not in prototype scope (see EXPANSION HOOKS)")
    mos = MOs_data(num_mo=2, aos_data=aos, mo_coefficients=np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64))
    mos.sanity_check()
    gem = SD_data.build(num_electron_up=_N_ELEC_UP, num_electron_dn=_N_ELEC_DN, mos_up=mos)  # time_reversal auto
    gem.sanity_check()
    return gem


def _build_jastrow(jastrow: str, structure: Structure_data, aos) -> Jastrow_data:
    """The requested ISOLATED Jastrow channel; ``none`` = all channels absent."""
    j1 = j2 = j3 = dtn = None
    if jastrow == "none":
        pass
    elif jastrow == "J1":
        j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure,
            # Zero is the CORRECT value here, not the hardcoded placeholder that
            # tests/_ecp_core_electrons.py exists to replace: this dummy ham is built
            # fixture-free and _build_coulomb hardcodes ecp_flag=False, so there is no
            # pseudopotential and no core to remove. There is nothing to read it from.
            core_electrons=tuple([0.0] * len(structure.atomic_numbers)),
            jastrow_1b_type="exp",
        )
    elif jastrow == "J2":
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=1.0, jastrow_2b_type="exp", structure_data=structure
        )
    elif jastrow == "J3poly":
        j3 = _make_j3(aos)
    elif jastrow == "DTN":
        dtn = _make_dtn(structure)
    else:  # pragma: no cover - guard
        raise NotImplementedError(f"Jastrow {jastrow!r} not in prototype scope")
    return Jastrow_data(jastrow_one_body_data=j1, jastrow_two_body_data=j2, jastrow_three_body_data=j3, jastrow_dtn_data=dtn)


def _make_j3(orb_data) -> Jastrow_three_body_data:
    """An ACTIVE three-body Jastrow on the GTO AO leaf (dummy symmetric coupling)."""
    rng = np.random.default_rng(5)
    n = int(orb_data._num_orb)
    jm = rng.normal(size=(n, n + 1)) * 0.1
    jm[:, :-1] = 0.5 * (jm[:, :-1] + jm[:, :-1].T)  # bosonic coupling block must be symmetric
    return Jastrow_three_body_data(orb_data=orb_data, j_matrix=jm)


def _make_dtn(structure: Structure_data) -> Jastrow_DTN_data:
    """An ACTIVE DTN polynomial Jastrow (u/chi/f, small orders)."""
    return Jastrow_DTN_data.build(
        structure_data=structure,
        # DTN cusps require explicit effective charges.
        z_eff_per_species=tuple(float(z) for z in np.unique(structure.atomic_numbers)),
        pseudized_per_species=tuple(False for _ in np.unique(structure.atomic_numbers)),
        use_u=True,
        use_chi=True,
        use_f=True,
        cutoff_smoothness=4,
        n_order_u=2,
        n_order_chi=2,
        n_order_eN=2,
        n_order_ee=2,
        L_u=None,
        L_chi=None,
        L_f=None,
        random_init=True,
        seed=13,
    )


def _build_coulomb(potential: str, structure: Structure_data) -> Coulomb_potential_data:
    """The Coulomb potential: ``AE`` = bare sum-of-Z/r (``ecp_flag=False``)."""
    if potential != "AE":  # pragma: no cover - expansion hook
        raise NotImplementedError(f"potential {potential!r} not in prototype scope (see EXPANSION HOOKS)")
    return Coulomb_potential_data(structure_data=structure, ecp_flag=False)


def build_dummy_ham(ao_family: str, geminal: str, jastrow: str, bc: str, potential: str) -> Hamiltonian_data:
    """Assemble a fixture-free dummy ``Hamiltonian_data`` for one universe cell."""
    structure = _build_structure(bc)
    aos = _build_aos(ao_family, structure)
    gem = _build_geminal(geminal, aos)
    wf = Wavefunction_data(jastrow_data=_build_jastrow(jastrow, structure, aos), geminal_data=gem)
    return Hamiltonian_data(
        structure_data=structure, coulomb_potential_data=_build_coulomb(potential, structure), wavefunction_data=wf
    )


# --------------------------------------------------------------------------- #
# The probe: run the analytic-ONLY derivative under the spy and classify.
# --------------------------------------------------------------------------- #
def _draw(seed: int = 0):
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(scale=0.7, size=(_N_ELEC_UP, 3))),
        jnp.asarray(rng.normal(scale=0.7, size=(_N_ELEC_DN, 3))),
    )


def probe(ham: Hamiltonian_data, quantity: str, axis: str, spy: _ADSpy) -> tuple[str, str]:
    """Run ``derivatives.compute_grad_<quantity>(..., analytic=True)[axis]`` and classify.

    Returns ``(status, detail)`` with status in IMPLEMENTED / SILENT_AUTODIFF /
    NOT_IMPLEMENTED / OTHER_ERROR.
    """
    spy.calls.clear()
    r_up, r_dn = _draw()
    flags = _flags_for_axis(axis)
    try:
        if quantity == "ln_Psi":
            derivatives.compute_grad_ln_Psi(ham, r_up, r_dn, flags, analytic=True)[axis]
        else:
            derivatives.compute_grad_e_L(ham, r_up, r_dn, jnp.eye(3), flags, analytic=True)[axis]
    except NotImplementedError as e:
        return "NOT_IMPLEMENTED", str(e).splitlines()[0]
    except Exception as e:
        return "OTHER_ERROR", f"{type(e).__name__}: {str(e).splitlines()[0]}"
    if spy.calls:
        return "SILENT_AUTODIFF", "; ".join(sorted(set(spy.calls)))
    return "IMPLEMENTED", ""


# --------------------------------------------------------------------------- #
# Parametrization over the full cartesian product of the prototype universe.
# --------------------------------------------------------------------------- #
def _cells() -> list[tuple[str, str, str, str, str]]:
    return list(itertools.product(_AO_FAMILIES, _GEMINALS, _JASTROWS, _BCS, _POTENTIALS))


def _cell_id(cell: tuple[str, ...]) -> str:
    return "__".join(cell)


_PARAMS = [
    pytest.param(cell, quantity, axis, id=f"{_cell_id(cell)}-{quantity}-{axis}")
    for cell in _cells()
    for quantity in _QUANTITIES
    for axis in _AXES
]

_LAST_CLEARED_CELL: tuple[str, ...] | None = None

# Session inventory: every classification, summarised at teardown.
_INVENTORY: list[tuple[tuple[str, ...], str, str, str, str]] = []


@pytest.fixture(scope="session", autouse=True)
def _sweep_report():
    yield
    counts = {
        s: sum(1 for *_r, st, _d in _INVENTORY if st == s)
        for s in ("IMPLEMENTED", "SILENT_AUTODIFF", "NOT_IMPLEMENTED", "OTHER_ERROR")
    }
    lines = [
        "",
        "=" * 80,
        "analytic-production sweep inventory (this xdist worker):",
        "-" * 80,
        f"  probed {len(_INVENTORY)} (cell, quantity, axis) combinations over "
        f"{len(_AO_FAMILIES)} AO x {len(_GEMINALS)} geminal x {len(_JASTROWS)} Jastrow x "
        f"{len(_BCS)} BC x {len(_POTENTIALS)} potential",
        "  " + "   ".join(f"{k} = {v}" for k, v in counts.items()),
        "-" * 80,
    ]
    for status in ("SILENT_AUTODIFF", "NOT_IMPLEMENTED", "OTHER_ERROR"):
        lines.append(f"{status} combinations [{counts[status]}]:")
        for cell, q, ax, st, detail in sorted(_INVENTORY):
            if st == status:
                lines.append(f"  {status}: {_cell_id(cell):28s} {q:7s} {ax:4s} :: {detail}")
    lines.append("=" * 80)
    print("\n".join(lines))


@pytest.mark.parametrize("cell, quantity, axis", _PARAMS)
def test_analytic_and_ad_free(cell, quantity, axis, ad_spy):
    """One (cell, quantity, axis): must be IMPLEMENTED -- runs clean AND fires no generic AD.

    ``NOT_IMPLEMENTED`` is a coverage regression, ``SILENT_AUTODIFF`` is a quiet ``_auto``
    dispatch (invisible to raise-detection and to the analytic-vs-_auto matrix), and
    ``OTHER_ERROR`` is a dummy-builder bug. All three are RED.
    """
    # Clear the jit cache once per CELL: a cache hit from an earlier session/test with
    # the same signature would skip tracing and blind the spy. Within a cell every
    # (quantity, axis) has a distinct signature (flags/static differ), so each probe
    # still traces at least the top level; per-probe clearing would only improve
    # offender ATTRIBUTION (cell -> exact probe) at ~8x the runtime.
    global _LAST_CLEARED_CELL
    if cell != _LAST_CLEARED_CELL:
        jax.clear_caches()
        _LAST_CLEARED_CELL = cell
    ham = build_dummy_ham(*cell)
    status, detail = probe(ham, quantity, axis, ad_spy)
    _INVENTORY.append((tuple(cell), quantity, axis, status, detail))
    assert status == "IMPLEMENTED", f"{_cell_id(cell)} {quantity}/{axis}: expected IMPLEMENTED, got {status} :: {detail}"


# --------------------------------------------------------------------------- #
# Direct periodic probes -- cells the cartesian product cannot build yet (BC axis is
# molecular-only until the pbc dummy builders land). Plain RED until the analytic
# ports of the Rust closed forms land (no xfail escapes).
# --------------------------------------------------------------------------- #
_L_BOX = 10.0
_NA_PBC = 4
_POS_PBC = np.random.default_rng(0).random((_NA_PBC, 3)) * _L_BOX
_R_PBC = jnp.full((6, 3), 2.0)


def _pbc_structure() -> Structure_data:
    return Structure_data.build(
        positions=_POS_PBC,
        pbc_flag=True,
        vec_a=(_L_BOX, 0.0, 0.0),
        vec_b=(0.0, _L_BOX, 0.0),
        vec_c=(0.0, 0.0, _L_BOX),
        atomic_numbers=(6,) * _NA_PBC,
        element_symbols=("C",) * _NA_PBC,
        atomic_labels=("C",) * _NA_PBC,
    )


class TestPeriodicKineticJastrowIsAnalytic:
    """PBC J1/J2 grad/lap sit on the production e_L kinetic path; they must be analytic.

    Reference implementations for the ports: ``smooth_dist`` (value/grad/|g|^2/lap,
    closed form) in ``jqmc/rs/src/jastrow_factor.rs``.
    """

    @pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
    def test_j1_grad_lap_pbc(self, ad_spy):
        """J1 grad lap PBC."""
        j1 = Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0, structure_data=_pbc_structure(), core_electrons=(0.0,) * _NA_PBC
        )
        jax.make_jaxpr(lambda a, b: compute_grads_and_laplacian_Jastrow_one_body(j1, a, b))(_R_PBC, _R_PBC)
        ad_spy.assert_clean("PBC J1 grad/lap (e_L kinetic path)")

    @pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
    def test_j2_grad_lap_pbc(self, ad_spy):
        """J2 grad lap PBC."""
        j2 = Jastrow_two_body_data.init_jastrow_two_body_data(jastrow_2b_param=1.0, structure_data=_pbc_structure())
        jax.make_jaxpr(lambda a, b: compute_grads_and_laplacian_Jastrow_two_body(j2, a, b))(_R_PBC, _R_PBC)
        ad_spy.assert_clean("PBC J2 grad/lap (e_L kinetic path)")


class TestSwctIsAnalytic:
    """SWCT terms of the analytic HF force; reference port: rs ``gfmc_forces::swct_domega``."""

    @pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
    def test_omega(self, ad_spy):
        """Omega."""
        jax.make_jaxpr(lambda r: evaluate_swct_omega(_pbc_structure(), r))(_R_PBC)
        ad_spy.assert_clean("swct omega")

    @pytest.mark.gdd_electron_configuration(mode="required", position_source="explicit")
    def test_domega(self, ad_spy):
        """Domega."""
        jax.make_jaxpr(lambda r: evaluate_swct_domega(_pbc_structure(), r))(_R_PBC)
        ad_spy.assert_clean("swct domega")


class TestSpyMechanism:
    """The guard itself must be able to see a violation (no silent-green failure mode)."""

    def test_spy_detects_planted_ad(self, ad_spy):
        """A grad call executed from a jqmc-side frame must be caught.

        Calls from OUTSIDE jqmc (e.g. this test file, oracles in tests/) are
        deliberately ignored by the spy, so the plant must run in a frame whose
        filename carries a ``/jqmc/`` path segment. ``compile`` with a synthetic
        filename gives exactly that without touching the real package.
        """
        code = compile(
            "spy_result = jax.grad(lambda x: (x**2).sum())(x0)",
            "/jqmc/_spy_self_check.py",
            "exec",
        )
        namespace = {"jax": jax, "x0": jnp.ones(3)}
        exec(code, namespace)
        assert any("grad" in c and "_spy_self_check" in c for c in ad_spy.calls), (
            "spy failed to observe a jax.grad call from a jqmc-side frame -- the guard is blind"
        )
