r"""Mixed precision configuration for jQMC.

================================================================================
DESIGN PRINCIPLES (READ THIS FIRST)
================================================================================

The mixed-precision implementation rests on **three** principles. Principle 3
is the most important in practice; almost every precision bug we have seen is
a violation of 3a or 3b.

------------------------------------------------------------
Principle 1 -- One Precision Zone is owned by exactly one module.
------------------------------------------------------------
A zone (e.g. ``ao_eval``, ``coulomb``) is *defined and consumed* in a single
module. The mapping zone <-> owning module is one-to-one and is documented in
the table below (and enforced by convention in ``_FULL_PRECISION`` /
``_MIXED_PRECISION``).

------------------------------------------------------------
Principle 2 -- A module may own multiple Precision Zones.
------------------------------------------------------------
Different code paths in the same module legitimately need different precisions
(e.g. ``ao_eval`` vs ``ao_grad_lap``, or ``det_eval`` vs ``det_ratio``). Each
zone is named for its *purpose*, not for its dtype.

------------------------------------------------------------
Principle 3 -- Cast responsibility lies with the function that does
              arithmetic on the value, never with passthrough wrappers.
------------------------------------------------------------

Definition. *arithmetic* means consuming a value as an operand of a numerical
operation (``+ - * /``, ``jnp.linalg.norm``, ``jnp.dot``, ``@``, ``jnp.exp``,
...) **or** as an input to ``jax.grad`` / ``jax.jacrev`` / ``jax.hessian``.
Operations that do *not* count as arithmetic and therefore do *not* trigger a
cast: ``len(x)``, ``x.shape``, ``x[i]`` (index lookup), the *target* of
``.at[i].set(y)``, and forwarding ``x`` as an argument to another function.

Principle 3a -- Arguments are frozen.
    Function arguments are treated as **frozen**, in the same sense as the
    attributes of a ``@dataclass(frozen=True)``: the name introduced by the
    parameter list **must not be rebound** for the entire body of the
    function. In particular, ``arg = jnp.asarray(arg, dtype=...)`` at the
    top of a function is forbidden -- it silently coerces the argument for
    every later use, including forwarding.

    Consequences (not extra rules -- direct corollaries of "frozen"):
      * Forwarding neutrality. A value forwarded to a callee transits in
        the dtype it was received in; the callee is responsible for casting
        it to *its* own zone (Principle 3b).
      * Cast at the use site. When the function consumes ``arg`` as an
        operand of its own arithmetic, the cast appears **inside the
        expression** (``arg.astype(dtype)``). Do *not* preemptively
        introduce a local alias just to hold the cast -- only do so when
        the cast result is reused multiple times, in which case introduce
        a *new* local variable with a different name (e.g.
        ``arg_local = arg.astype(dtype)``). The original ``arg`` always
        remains frozen.

Principle 3b -- Local cast at the point of arithmetic.
    A function casts a value to its own zone's dtype **immediately before**
    consuming it as an operand. Inputs and outputs of the function's
    arithmetic both live in its zone. Intermediate computations may use a
    higher precision when needed for numerical reasons (the canonical case
    being ``r - R``: reconstruct the difference in the **dtype the value
    was received in** -- i.e. the precision chosen by the upper layer --
    to avoid catastrophic cancellation, then down-cast the result back to
    the function's own zone). In jQMC the upstream (mcmc walker state) is
    always fp64, so in practice the reconstruction happens in fp64; the
    *principle*, however, is "use the caller-supplied precision," not
    "hardcode fp64." Concretely: do not write ``jnp.float64`` as a
    literal, and avoid pinning the reconstruction to a specific zone
    name when the incoming value's own dtype already carries the right
    precision.

Worked example (the ECP -> AO bug this design prevents)::

    # WRONG: rebinding `r_carts` at the top of compute_coulomb forwards a
    # fp32-truncated array to compute_AOs, even though `ao_eval` is fp64.
    def compute_coulomb(r_carts, R_carts):
        dtype_jnp = get_dtype_jnp("coulomb")
        r_carts = jnp.asarray(r_carts, dtype=dtype_jnp)  # 3a violation
        R_carts = jnp.asarray(R_carts, dtype=dtype_jnp)
        ao = compute_AOs(..., r_carts, R_carts)          # downstream sees fp32
        diff = r_carts - R_carts
        ...

    # RIGHT: forwarding stays in the caller's dtype; the local arithmetic
    # reconstructs the difference in the dtype the values were received in
    # (the upper-layer precision -- fp64 in jQMC because mcmc walker state
    # is fp64) and casts the result back to the function's own zone.
    def compute_coulomb(r_carts, R_carts):
        ao = compute_AOs(..., r_carts, R_carts)          # 3a: forward as-is
        dtype_jnp = get_dtype_jnp("coulomb")
        # reconstruct in the caller-supplied precision, then down-cast
        diff = (r_carts - R_carts).astype(dtype_jnp)     # 3b
        ...

Verification procedure.
    To verify a module:
      * (3a) Search for ``arg = jnp.asarray(arg, dtype=...)`` at the top of
        any public function. Each occurrence is a 3a candidate violation --
        the rebind silently coerces the argument for any subsequent
        forwarding too.
      * (3b) For each arithmetic expression, check that all operands have
        been cast to the function's zone in the immediately preceding lines.
      * (catastrophic cancellation) For each ``r - R`` style difference of
        coordinates, check the reconstruct-in-caller-precision-then-downcast
        pattern (in jQMC this is fp64 in practice because the upstream is
        always fp64, but the rule is "use the dtype the value was received
        in," not "hardcode fp64").

No hardcoded dtypes inside selectable-precision modules.
    Inside any module that owns one or more selectable-precision zones
    (i.e. any zone whose dtype can differ between ``"full"`` and
    ``"mixed"`` mode), **never hardcode** ``jnp.float64`` / ``np.float64``
    / ``jnp.float32`` / ``np.float32`` as a literal dtype for arrays the
    module produces or consumes. Always go through the accessors
    ``get_dtype_jnp("<zone>")`` / ``get_dtype_np("<zone>")`` so the dtype
    follows the active mode automatically. The only legitimate exception
    is a module whose owned data is **always fp64 by construction**,
    independent of any selectable zone:

      * ``mcmc`` / ``gfmc`` (MCMC and GFMC walker state, always fp64).
      * I/O modules that load and store external numerical data
        (``structure``, ``trexio_wrapper``, ``_driver_helpers``,
        ``jqmc_tool``, and the ``_load_dataclass_from_hdf5`` /
        ``_save_dataclass_to_hdf5`` helpers in ``hamiltonians``):
        on-disk numerical data (AO exponents/coefficients, nuclear
        coordinates, geminal coefficients, etc.) is always fp64
        because fp32 storage would silently lose precision that no
        downstream fp64 upcast can recover.
      * The closed-form analytic derivative kernels (``jqmc/derivatives/``).
        The package is fp64 by construction, uniformly (every dtype
        literal in it is float64; zero float32): the analytic kernels are
        the PRODUCTION gradient integrands, and the mixed-precision
        gradient contract is that estimator integrands are evaluated at
        full precision regardless of mode -- the sampler's fp32 zones
        perturb only the sampled measure (an O(eps32) relative effect,
        far below statistical error), never the derivative arithmetic.
        Mode-dependence still reaches these kernels lawfully through the
        VALUES they consume from the owning modules (``compute_AOs`` ->
        ``ao_eval``, ``compute_grads_and_laplacian_Jastrow_part`` ->
        ``jastrow_grad_lap``): the owning module casts per its own zone
        (Principle 3b) and the derivative kernel receives the result
        as-is. Principle 1 (zone <-> owning module is 1:1) is also why
        these kernels do not consult ``get_dtype_jnp("ao_grad_lap")``
        themselves -- that zone belongs to ``atomic_orbital.py``.
      * Basis-data storage accessors. ``_*_jnp`` properties on
        selectable-precision dataclasses whose underlying storage
        field is typed ``npt.NDArray[np.float64]`` are *lift-only*
        adapters (numpy -> ``jax.Array``), not arithmetic. The dtype
        is fp64 by construction: storage is loaded from
        HDF5/TREXIO/optimizer output, and downcasting at the
        accessor would silently
        lose precision that no downstream upcast can recover. The
        consumer is responsible for casting the lifted array to its
        own zone at the use site (Principle 3b). Concretely this
        covers the ``_*_jnp`` accessors for AO exponents/coefficients
        and normalization-factorial-ratio caches in
        ``atomic_orbital``, ``_mo_coefficients_jnp`` in
        ``molecular_orbital``, ``_lambda_matrix_jnp`` in
        ``determinant`` and ``_j_matrix_jnp`` in ``jastrow_factor``.
      * Jastrow variational-parameter storage
        (``jastrow_factor._JASTROW_PARAMETER_STORAGE_DTYPE``): the J1 / J2
        parameters, the J3 ``j_matrix`` and the DTN independent coefficients
        are stored in fp64 at construction, basis conversion, optimizer update
        and checkpoint load, in both modes.  Storing them in a zone dtype
        rounded them for every zone that consumes them (issue #781).

    These modules may use ``jnp.float64`` / ``np.float64`` directly
    because the dtype is not a function of ``mode``. Verify with::

        grep -nE 'jnp\.float(32|64)|np\.float(32|64)' jqmc/<module>.py

    Each hit inside a selectable-precision module is a candidate violation.

================================================================================

Users choose one of two modes:

- ``"full"``  (default) -- all zones float64 (backward compatible).
- ``"mixed"`` -- recommended mixed precision; low-risk zones become
  float32 while numerically sensitive zones stay float64.

Individual zone assignments are **not** user-configurable; they are
defined in ``_FULL_PRECISION`` and ``_MIXED_PRECISION`` below.
Developers who need to tweak per-zone dtypes should edit those dicts
directly in this file.

Precision Zones
---------------

==================  =================================  =========  ========  =====  =========
Zone                Owning module                      Default    Mixed     risk   E_L path
==================  =================================  =========  ========  =====  =========
``ao_eval``         atomic_orbital.py (forward)        float64    float32   low    core
``ao_bloch_eval``   atomic_orbital.py (Bloch GTO)      float64    float64   high   core
``ao_grad_lap``     atomic_orbital.py (grad/Lap)       float64    float64   high+  core
``mo_eval``         molecular_orbital.py (forward)     float64    float64   high*  core
``mo_grad``         molecular_orbital.py (gradient)    float64    float64   high   core
``mo_lap``          molecular_orbital.py (Laplacian)   float64    float64   high   core
``tiling_eval``     orbital_tiling.py (forward)       float64    float64   high   core
``tiling_grad``     orbital_tiling.py (gradient)      float64    float64   high   core
``tiling_lap``      orbital_tiling.py (Laplacian)      float64    float64   high   core
``jastrow_eval``    jastrow_factor.py (forward)        float64    float32   low    core*
``jastrow_grad_lap`` jastrow_factor.py (grad/lap)      float64    float32   low#   core
``jastrow_grad_lap_carry`` jastrow_factor.py (DTN carry) float64  float64   high## core
``jastrow_ratio``   jastrow_factor.py (ratio update)   float64    float64   high   indirect**
``det_eval``        determinant.py (geminal + log-det) float64    float64   high   core
``det_grad_lap``    determinant.py (grad/lap of lnDet) float64    float64   high   core
``det_ratio``       determinant.py (SM ratio update)   float64    float64   high   indirect**
``coulomb``         coulomb_potential.py               float64    float64   low-med core
``ewald``           coulomb_potential.py (Ewald)       float64    float64   high   core
``wf_eval``         wavefunction.py (Psi, ln Psi)      float64    float64   high   core*
``wf_kinetic``      wavefunction.py (T_L assembly)     float64    float64   high   core
``wf_ratio``        wavefunction.py (Psi(R')/Psi(R))   float64    float64   high   no
``local_energy``    hamiltonians.py (T + V assembly)   float64    float64   high   core
``swct``            swct.py                            float64    float64   high   no
==================  =================================  =========  ========  =====  =========

\\* ``mo_eval`` is a high-risk zone even though the consumed AO values are
fp32: the small ``mo_coefficients @ aos`` matmul is run in this zone, and
its output feeds the determinant matrix, where fp32 round-off is
amplified by log|det|.

* ``jastrow_eval`` and ``wf_eval`` are on the E_L core path but their
forward values (J and ln|Psi|) do not enter the E_L formula directly
(E_L depends on *derivatives* of ln|Psi|), so fp32 in these zones alone
has no E_L impact.

+ ``ao_grad_lap`` is fp64 even in mixed mode because the analytic
Laplacian kernel for spherical AOs contains catastrophic cancellation
(``4 Z^2 r^2 - 6 Z`` and ``(safe_div - 2 Z*base)^2 - safe_div^2 - 2 Z``
terms) that fp32 cannot resolve for tight Gaussians. ``ao_grad`` is
merged with ``ao_lap`` into this single zone because the fused
``compute_AOs_value_grad_lap`` kernel shares one heavy expression
(``exp / pow / phi / S_l_m``) across grad and lap. Running that shared
kernel at fp32 would break the lap path, so the unified zone is fp64
always -- a small extra cost on the standalone ``compute_AOs_grad``
(which is not on the per-step hot path) in exchange for a single
source of truth for the shared kernel dtype.

# ``jastrow_grad_lap`` remains an fp32 mixed zone.  The DTN f-channel pins the
contraction order of its small polynomial reduction to ``n -> m -> l``.  The
default ``l -> m -> n`` order is cancellation-sensitive in higher derivatives
of the final stress tensor.  No input, intermediate, or public grad/Laplacian
array is promoted by this stabilization.

## ``jastrow_grad_lap_carry`` is the carried DTN grad/Laplacian state of the
GFMC/LRDMC projection (XLA ``Jastrow_DTN_grad_lap_streaming_state``, Rust
``DtnGradLapCache``): its seed, its per-hop moved-electron delta and its
accumulator.  The carried state lives for a whole branching step or projection
interval, so a rounding error of the seed is not re-drawn at each hop as a
fresh ``jastrow_grad_lap`` evaluation is; it persists until the next seed.  It
is therefore fp64 in both modes.  Its DTN coefficients are inflated in this
zone too (XLA ``_dtn_carry_coefficient_dtype``, Rust ``JastrowDtnData::zone``),
not in the ``jastrow_grad_lap`` dtype of the shared native coefficient set,
which the carry does not consume (issue #781), so the two backends carry the
same quantity.

Every Jastrow ratio ``exp(J(R') - J(R))`` is ``jastrow_ratio`` arithmetic, also
where a reference / fallback / debug route forms it from two whole values: both
values are then evaluated in ``jastrow_ratio`` (XLA
``compute_Jastrow_part_ratio_zone``, Rust
``coulomb_potential::jastrow_value_for_ratio``), never in ``jastrow_eval``.  A
state shared by a ratio kernel and a grad/lap kernel (the XLA J3 streaming
state) keeps the tables the ratio reads in the higher precision of the two
zones (issue #781).

** ``det_ratio`` and ``jastrow_ratio`` affect E_L **indirectly** through
the ECP non-local potential, which evaluates Psi(R')/Psi(R) on a
quadrature grid via rank-1 ratio updates (see
``coulomb_potential.compute_ecp_non_local_parts_nearest_neighbors_fast_update``).
In non-ECP systems these zones have no E_L impact.

Usage::

    from jqmc._precision import get_dtype_jnp

    def compute_AOs(aos_data, r_carts):
        # Forwarding-only path: do NOT rebind r_carts here (Principle 3a).
        return _compute_AOs_kernel(aos_data, r_carts)

    def _compute_AOs_kernel(aos_data, r_carts):
        # This function performs arithmetic on r_carts, so it casts at the
        # use site (Principle 3b). The (r - R) reconstruction is done in
        # the dtype the values were received in (caller-supplied
        # precision: fp64 in jQMC because mcmc walker state is fp64 and
        # the atomic centers are loaded from disk as fp64). The result
        # is then down-cast to this function's own zone (``ao_eval``).
        # NOTE: never reach for another module's zone (e.g.
        # ``get_dtype_jnp("local_energy")``) here -- that violates
        # Principle 1 (zone <-> owning module is 1:1). atomic_orbital.py
        # may only consult ao_eval / ao_grad_lap.
        dtype_jnp = get_dtype_jnp("ao_eval")
        R_carts = aos_data._atomic_center_carts_jnp
        diff = (r_carts - R_carts).astype(dtype_jnp)
        ...
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

import logging
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax import typing as jnpt

logger = logging.getLogger(__name__)

# Every jax-using jqmc module imports this one, so it is where jqmc first meets
# jax. Report (once, never raising, changing nothing) when the installed jaxlib
# can crash silently loading a large executable from an enabled JAX persistent
# compilation cache (jaxlib 0.9.1/0.9.2).
from ._setting import warn_if_persistent_cache_load_can_overflow  # noqa: E402

warn_if_persistent_cache_load_can_overflow()

# --- mode="full" (all float64, backward compatible) ---
# Zones are listed grouped by owning module for readability.
_FULL_PRECISION: dict[str, str] = {
    # atomic_orbital.py
    "ao_eval": "float64",  # AO forward evaluation
    "ao_bloch_eval": "float64",  # cancellation-sensitive complex periodic GTO values
    "ao_grad_lap": "float64",  # AO gradient + Laplacian (unified for fused kernel)
    # molecular_orbital.py
    "mo_eval": "float64",  # MO forward evaluation (mo_coef @ AO)
    "mo_grad": "float64",  # MO gradient
    "mo_lap": "float64",  # MO Laplacian
    # orbital_tiling.py: local block contractions and final normalization
    "tiling_eval": "float64",
    "tiling_grad": "float64",
    "tiling_lap": "float64",
    # jastrow_factor.py
    "jastrow_eval": "float64",  # Jastrow factor (J1/J2/J3)
    "jastrow_grad_lap": "float64",  # Jastrow gradient / Laplacian
    "jastrow_grad_lap_carry": "float64",  # carried DTN gradient / Laplacian (GFMC projection state)
    "jastrow_ratio": "float64",  # Jastrow ratio (rank-1 update)
    # determinant.py
    "det_eval": "float64",  # geminal matrix elements + log-det / SVD / AS reg
    "det_grad_lap": "float64",  # gradient / Laplacian of ln|Det|
    "det_ratio": "float64",  # |Det(R')|/|Det(R)| Sherman-Morrison rank-1 update
    # coulomb_potential.py
    "coulomb": "float64",  # Coulomb + ECP potential
    "ewald": "float64",  # Ewald lattice sum (PBC); reciprocal-space cancellation is fp32-fragile
    # wavefunction.py
    "wf_eval": "float64",  # Psi, ln Psi evaluators
    "wf_kinetic": "float64",  # T_L = -1/2 (lap_lnPsi + |grad_lnPsi|^2) assembly
    "wf_ratio": "float64",  # Psi(R')/Psi(R) = exp(J' - J) * det'/det (LRDMC discretized)
    # hamiltonians.py
    "local_energy": "float64",  # E_L = T + V assembly
    # swct.py
    "swct": "float64",  # SWCT omega / domega
}

# --- mode="mixed" (recommended mixed precision) ---
# Three "low risk" zones drop to float32:
#
#   ao_eval          - real GTO, plane-wave and spline value kernels.
#                      Sensitive downstream contractions use their own fp64
#                      zones. Complex pGTO values have the separate guarded
#                      ao_bloch_eval policy below; upcasting already-rounded
#                      Bloch values cannot recover the lost precision.
#   jastrow_eval     - smooth correlation function value (pre-exp).
#   jastrow_grad_lap - nabla J, nabla^2 J; the bulk smooth Jastrow work
#                      stays fp32. Kept as a single zone (no grad/lap split)
#                      because Jastrow grad/lap functions compute both
#                      together (``compute_grads_and_laplacian_*``). The DTN
#                      f-channel pins its fp32 polynomial contraction order to
#                      avoid a cancellation-sensitive planner choice; all its
#                      inputs and outputs remain in-zone.
# All other zones stay fp64 because fp32 in those zones produces
# unacceptable bias on E_L, OR the kernel is cheap enough that fp32
# is not worth the bias:
#
#   ao_grad_lap   - analytic gradient + Laplacian kernel for spherical/
#                   Cartesian AOs.  Lap arithmetic contains catastrophic
#                   cancellation (``4 Z^2 r^2 - 6 Z`` and
#                   ``(safe_div - 2 Z*base)^2 - safe_div^2 - 2 Z``).
#                   fp64 mandatory.  This zone merges grad and lap
#                   because the fused ``compute_AOs_value_grad_lap``
#                   kernel evaluates ``exp / pow / phi / S_l_m`` once
#                   and reuses it across grad and lap; running the
#                   shared path at fp32 would break the lap output.
#   jastrow_grad_lap_carry
#                 - carried DTN grad/lap of the GFMC projection (seed,
#                   per-hop delta, accumulator). A seed rounding error is
#                   carried unchanged until the next seed instead of being
#                   re-drawn per evaluation; the per-hop delta is
#                   O(N_atom N_e), so fp64 costs little.
#   coulomb       - sum of 1/r + ECP spherical quadrature.  Cheap
#                   (O(N_e^2) el-el + O(N_e * N_nuc) el-ion, vs
#                   O(N_e * N_ao) AO eval); cost/benefit favors fp64.
#   ewald         - Ewald lattice sum (PBC).  Reciprocal-space sum
#                   contains 1/G^2 in the denominator and cos(G * r_ij)
#                   cancellations between G and -G; both are fp32-fragile.
#                   Real-space sum uses erfc(eta r) / r which can
#                   underflow at fp32 for r near r_real_cut. fp64 in
#                   both modes -- the kernel is cheap (image / G lists
#                   pre-enumerated) so cost/benefit strongly favors fp64.
#
#   ao_bloch_eval - complex periodic GTO values and Bloch image sums. Real QE
#                   stress comparisons expose cancellation after local MO and
#                   inverse contractions even with a moderate Slater condition
#                   number. Rounding only the completed Bloch AO is insufficient.
#   mo_eval       - mo_coef @ AO matmul feeds the determinant matrix;
#                   fp32 here amplifies into log|det| errors of O(1).
#   det_eval      - geminal matrix + log(det) + SVD; cancellation in
#                   log(det), SVD 1/s near-singular entries produce
#                   O(1) log|det| error.
#   mo_grad / mo_lap / det_grad_lap
#                 - second derivatives of ln|Psi|; cancellation-sensitive
#                   on the determinant side (the AO-side fp32 is absorbed
#                   by the fp64 mo_coef matmul). jastrow_grad_lap is the
#                   zone-level exception: its bulk work is smooth fp32, with
#                   the DTN f-channel contraction protected by the explicit
#                   fp32 reduction order documented above.
#                   det_grad_lap is kept as a single zone (no grad/lap
#                   split) for symmetry with jastrow_grad_lap and because
#                   the determinant grad/lap functions naturally compute
#                   both quantities together (``compute_grads_and_laplacian_*``).
#   wf_kinetic    - sum (lap_J + lap_lnD) + |grad_J + grad_lnD|^2; cancellation.
#   local_energy  - T + V assembly; small differences between large terms.
#   det_ratio     - SM rank-1 ratio used by MCMC accept/reject AND
#                   ECP non-local Psi(R')/Psi(R) quadrature.
#   jastrow_ratio - J(R')-J(R) subtracts nearby accumulated Jastrow
#                   values. fp32 cancellation is amplified by the LRDMC
#                   finite-difference kinetic estimator, so this stays fp64.
#   swct          - geometric SWCT correction, derivative-sensitive.
_MIXED_PRECISION: dict[str, str] = {
    # atomic_orbital.py
    "ao_eval": "float32",  # real GTO, PW and spline values
    "ao_bloch_eval": "float64",  # Bloch image/local-MO cancellation amplifies value roundoff in stress
    "ao_grad_lap": "float64",  # high risk (catastrophic cancellation in 4Z^2r^2-6Z terms;
    # unified zone -- grad and lap share the fused compute_AOs_value_grad_lap kernel,
    # which must run at fp64 to protect the lap path)
    # molecular_orbital.py
    "mo_eval": "float64",  # high risk (feeds det_eval)
    "mo_grad": "float64",  # high risk
    "mo_lap": "float64",  # high risk
    # orbital_tiling.py: AO values arrive in ao_eval; contractions remain high precision
    "tiling_eval": "float64",
    "tiling_grad": "float64",
    "tiling_lap": "float64",
    # jastrow_factor.py
    "jastrow_eval": "float32",  # low risk
    "jastrow_grad_lap": "float32",  # DTN f contraction pins a stable fp32 reduction order
    "jastrow_grad_lap_carry": "float64",  # high risk (a seed rounding persists over the carried interval)
    "jastrow_ratio": "float64",  # high risk (LRDMC finite-difference amplification)
    # determinant.py
    "det_eval": "float64",  # high risk (LU/det / SVD)
    "det_grad_lap": "float64",  # high risk (kept unsplit for symmetry with jastrow)
    "det_ratio": "float64",  # high risk (SM update error + ECP non-local ratio)
    # coulomb_potential.py
    "coulomb": "float64",  # cheap kernel; cost/benefit favors fp64
    "ewald": "float64",  # reciprocal-sum cancellation in cos(G * r_ij) and 1/G^2 is fp32-fragile
    # wavefunction.py
    "wf_eval": "float64",  # high risk
    "wf_kinetic": "float64",  # high risk
    "wf_ratio": "float64",  # high risk (exp(J'-J)*det'/det in LRDMC)
    # hamiltonians.py
    "local_energy": "float64",  # high risk
    # swct.py
    "swct": "float64",  # high risk
}

ALL_ZONES = frozenset(_FULL_PRECISION.keys())
# Stable wire order for the Python -> Rust precision-policy snapshot.  The
# dictionaries above remain the single source of truth; this tuple only fixes
# serialization order and is asserted against the Rust schema by the native
# precision-contract tests.
PRECISION_ZONE_ORDER = tuple(_FULL_PRECISION.keys())

# Runtime zone -> dtype string mapping ("float32" or "float64").
# Strings are stored (not numpy/jax dtype types) so the str -> jnp.* / np.*
# conversion lives inside the per-flavor accessors below.  This keeps the
# concrete dtype flavor (jnp vs np) cleanly separated at the API boundary.
#
# This module-level dict is the **single source of truth** for the precision
# state within a Python process: ``configure(mode)`` clears and refills it,
# and ``get_dtype_jnp`` / ``get_dtype_np`` read from it.  No other variable
# (class attribute, environment variable, etc.) holds the active dtype
# mapping -- this dict is the only place to consult or mutate.
_zone_dtypes: dict[str, str] = {}


def configure(mode: str = "full") -> None:
    """Activate a precision mode.

    Args:
        mode: ``"full"`` (default, all float64) or ``"mixed"``
            (recommended mixed precision).

    Raises:
        ValueError: If *mode* is not ``"full"`` or ``"mixed"``.
    """
    if mode == "full":
        base = _FULL_PRECISION
    elif mode == "mixed":
        base = _MIXED_PRECISION
    else:
        raise ValueError(f"Unknown precision mode '{mode}'. Must be 'full' or 'mixed'.")

    # fp32 zones and their tolerances assume true fp32 arithmetic.  With the
    # JAX matmul precision unset, XLA lowers fp32 dot products at DEFAULT
    # precision, which Ampere-class GPUs execute at TF32-class accuracy
    # (~1e-3 relative); fp64 dots and CPU runs are unaffected.  Request HIGHEST
    # unless the user chose a precision explicitly (e.g. via
    # JAX_DEFAULT_MATMUL_PRECISION).  Set before the idempotency check below so
    # a process that configures the same mode again still gets it.
    if jax.config.jax_default_matmul_precision is None:
        jax.config.update("jax_default_matmul_precision", "highest")

    new_policy: dict[str, str] = {}
    for zone, dtype_str in base.items():
        if dtype_str not in ("float32", "float64"):
            raise ValueError(f"Invalid dtype '{dtype_str}'. Must be 'float32' or 'float64'.")
        new_policy[zone] = dtype_str
    current_policy = {zone: _get_zone_str(zone) for zone in ALL_ZONES}
    if current_policy == new_policy:
        return

    # Precision accessors are evaluated while a jitted function is traced;
    # the module-level policy itself is not part of JAX's compilation-cache
    # key.  Reusing an executable after changing this policy would therefore
    # run the previous mode's dtypes.  Invalidate compiled executables before
    # publishing the new policy.  Reconfiguring the already-active mode is a
    # no-op so per-test and per-driver setup does not force recompilation.
    jax.clear_caches()
    _zone_dtypes.clear()
    _zone_dtypes.update(new_policy)


def _get_zone_str(zone: str) -> str:
    """Return the stored dtype string for *zone* (``"float32"`` or ``"float64"``).

    ``"jastrow_two_body"`` is the component-level name used by weighted
    position-derivative consumers; it follows the owning
    ``"jastrow_grad_lap"`` zone. The canonical precision-zone policy and Rust wire
    schema contain only the owning zone.

    When :func:`configure` has not been called, ``"float64"`` is returned
    for any zone name (backward compatible).  After :func:`configure` has
    been called, an unknown *zone* raises :class:`ValueError`.
    """
    if zone == "jastrow_two_body":
        zone = "jastrow_grad_lap"
    if _zone_dtypes:
        if zone not in ALL_ZONES:
            raise ValueError(f"Unknown precision zone '{zone}'. Available zones: {sorted(ALL_ZONES)}")
    return _zone_dtypes.get(zone, "float64")


def precision_policy_snapshot() -> dict[str, str]:
    """Return an immutable-by-convention snapshot for a native run context.

    A fresh dict is returned on every call, so later :func:`configure` calls
    cannot mutate a policy already handed to Rust. Rust validates the complete
    zone schema at the PyO3 boundary.
    """
    return {zone: _get_zone_str(zone) for zone in PRECISION_ZONE_ORDER}


def get_dtype_jnp(zone: str) -> type:
    """Return the JAX dtype for a Precision Zone.

    Args:
        zone: Precision Zone name (e.g. ``"ao_eval"``, ``"wf_kinetic"``).

    Returns:
        ``jnp.float32`` or ``jnp.float64``.
    """
    return jnp.float32 if _get_zone_str(zone) == "float32" else jnp.float64


def get_dtype_np(zone: str) -> type:
    """Return the numpy dtype for a Precision Zone.

    Convenience helper for numpy-only code paths (e.g. ``_debug`` reference
    implementations) where importing or branching on ``jnp`` is awkward.

    Args:
        zone: Precision Zone name (e.g. ``"ao_eval"``, ``"wf_kinetic"``).

    Returns:
        ``np.float32`` or ``np.float64``.
    """
    return np.float32 if _get_zone_str(zone) == "float32" else np.float64


def get_complex_dtype_jnp(zone: str) -> type:
    """Return the JAX complex dtype with the zone's component precision.

    A float32 zone uses complex64 (two float32 components); a float64 zone
    uses complex128 (two float64 components).  Model/storage arrays may be
    wider, but zone-owning arithmetic must select its local dtype through this
    accessor rather than inherit the storage width through ``result_type``.
    """
    return jnp.complex64 if _get_zone_str(zone) == "float32" else jnp.complex128


def get_complex_dtype_np(zone: str) -> type:
    """NumPy counterpart of :func:`get_complex_dtype_jnp`."""
    return np.complex64 if _get_zone_str(zone) == "float32" else np.complex128


def get_loosest_zone_dtype_np(zones: Iterable[str]) -> type:
    """Return the numpy dtype of the lowest-precision zone in ``zones``.

    The dtype analogue of :func:`get_tolerance_min`: when a value is produced
    by a computation that crosses several zones, the accuracy it can actually
    carry is bounded by the weakest zone on that path -- even if the value is
    finally *stored* in a wider dtype. Numerical guards must therefore build
    their machine-epsilon from this dtype, not from the storage dtype, or they
    impose an fp64-tight bound on fp32-accurate data and fire spuriously under
    ``--precision-mode=mixed``.

    Args:
        zones: Iterable of Precision Zone names the value's computation crossed.

    Returns:
        ``np.float32`` if any listed zone is float32, else ``np.float64``.
    """
    return np.float32 if any(_get_zone_str(z) == "float32" for z in zones) else np.float64


def cast_to_zone(x: jnpt.ArrayLike, zone: str, *extra_dtypes: npt.DTypeLike) -> jax.Array:
    r"""Cast ``x`` to a Precision Zone dtype without dropping complex data.

    A 'pre-positioned promotion' pattern --
    ``dtype = jnp.result_type(get_dtype_jnp(zone), reference.dtype)``
    followed by ``jnp.asarray(x, dtype=dtype)`` -- silently drops the
    imaginary part of ``x`` whenever ``x`` itself is complex but the
    reference (and therefore ``dtype``) is real. The canonical example
    is the complex Bloch path: ``geminal.lambda`` is real
    on an RHF singlet, but the AO leaves are complex at twist
    ``k != 0`` so any quantity built from them (det, det_ratio,
    grad/lap, ratio) is complex too. The pre-positioned cast then
    silently zeroes the imaginary part at the wrong place, and the
    ``_strict_complex_cast_guard`` autouse fixture turns the resulting
    ``ComplexWarning`` into a test failure.

    This helper realises zone-owned component precision:

    .. code-block:: python

       # pre-positioned promotion (drops Im on complex Bloch):
       dtype_jnp = jnp.result_type(get_dtype_jnp("zone"), lambda.dtype)
       X = jnp.asarray(X, dtype=dtype_jnp)
       Y = jnp.asarray(Y, dtype=dtype_jnp)

       # canonical (correct on real AND complex paths):
       X = cast_to_zone(X, "zone", lambda.dtype)
       Y = cast_to_zone(Y, "zone", lambda.dtype)

    ``x.dtype`` and ``extra_dtypes`` determine whether the operation is real
    or complex; the zone determines the component width. Therefore an f32
    zone maps real arithmetic to float32 and complex arithmetic to complex64,
    while an f64 zone maps them to float64 and complex128. Wider model/storage
    inputs are intentionally converted at this local arithmetic boundary.

    Args:
        x: The array to cast (anything ``jnp.asarray`` accepts).
        zone: Precision Zone name (e.g. ``"det_ratio"``, ``"wf_kinetic"``).
        *extra_dtypes: Optional reference dtypes used to preserve whether an
            upstream path is complex (e.g. ``geminal._lambda_matrix_jnp.dtype``).
            Their component width does not override the zone. ``x.dtype`` is
            included automatically; do not pass it again.

    Returns:
        ``jnp.asarray(x, dtype=target)`` where ``target`` is float32/complex64
        for an f32 zone or float64/complex128 for an f64 zone.
    """
    x_arr = jnp.asarray(x)
    input_dtypes = (x_arr.dtype, *extra_dtypes)
    is_complex = any(np.issubdtype(np.dtype(dtype), np.complexfloating) for dtype in input_dtypes)
    target = get_complex_dtype_jnp(zone) if is_complex else get_dtype_jnp(zone)
    return jnp.asarray(x_arr, dtype=target)


def is_mixed_precision_enabled() -> bool:
    """Return ``True`` if at least one zone is set to float32."""
    return any(s == "float32" for s in _zone_dtypes.values())


def get_tolerance(zone: str, level: str = "strict") -> tuple[float, float]:
    """Return test tolerances ``(atol, rtol)`` scaled by the zone's dtype.

    Args:
        zone: Precision Zone name.
        level: ``"strict"`` or ``"medium"``.

    Returns:
        ``(atol, rtol)`` tuple appropriate for the zone's current dtype.
    """
    from jqmc._setting import _TOLERANCE

    return _TOLERANCE[level][_get_zone_str(zone)]


def get_evaluation_kappa_max(zone: str, level: str, degree: int) -> float:
    """Return the central evaluation-condition-number cap for a zone comparison.

    The cap is ``rtol / ((4 * degree + 4) * eps)`` where ``rtol`` is the
    unchanged :func:`get_tolerance` value of ``(zone, level)`` and ``eps`` is the
    machine epsilon of the zone's current dtype; see
    :func:`jqmc._setting.evaluation_kappa_max` for the forward-error derivation.

    Args:
        zone: Precision Zone name whose dtype evaluates the quantity.
        level: ``"strict"`` or ``"medium"`` (the comparison's level).
        degree: Total polynomial degree of the evaluated quantity.

    Returns:
        The cap for the active precision mode.
    """
    from jqmc._setting import evaluation_kappa_max

    _atol, rtol = get_tolerance(zone, level)
    return evaluation_kappa_max(rtol, float(np.finfo(get_dtype_np(zone)).eps), degree)


def get_tolerance_squared_observable(atol: float, rtol: float) -> tuple[float, float]:
    """Widen a base quantity's ``(atol, rtol)`` for comparing its SQUARE.

    For ``y = x**2`` the first-order error image is ``dy/y = 2 dx/x``: two
    implementations that agree on ``x`` to ``rtol`` can only be expected to
    agree on ``x**2`` to ``2 * rtol``. Judging both against the same envelope
    makes the squared quantity fail at half the base quantity's headroom on
    every fixture -- a property of the algebra, not of any implementation.
    Only ``rtol`` is amplified: the first-order absolute image is
    ``2 |x| * atol``, which is TIGHTER than ``atol`` whenever ``|x| < 0.5``,
    so keeping the base ``atol`` is conservative.

    Args:
        atol: The base quantity's absolute tolerance (kept unchanged).
        rtol: The base quantity's relative tolerance.

    Returns:
        ``(atol, RTOL_AMPLIFICATION_SQUARED_OBSERVABLE * rtol)``.
    """
    from jqmc._setting import RTOL_AMPLIFICATION_SQUARED_OBSERVABLE

    return atol, RTOL_AMPLIFICATION_SQUARED_OBSERVABLE * rtol


def get_tolerance_cancelling_mean(
    atol: float,
    rtol: float,
    reference_scale: float | np.floating,
) -> tuple[float, float]:
    """Rebase ``(atol, rtol)`` onto the SCALE of the terms a mean is built from.

    For a weighted mean ``m = sum(w_i x_i) / sum(w_i)`` over MIXED-SIGN terms,
    ``|m|`` carries no information about how accurately the two routes agree.
    Each summand is computed to ``rtol`` of its own magnitude, so the absolute
    error of the ratio is set by ``max|x_i|``, not by ``|m|``. When the sum
    cancels, ``rtol * |m|`` collapses with it and the pin silently tightens by
    the cancellation factor -- at no point because the arithmetic got better.

    Measured on GFMC_n / ``Li_ae_ccpvdz_cart.h5`` under ``mpirun -np 2``: the
    per-step ``e_L`` array has median ``|e_L| = 9.22`` and a single element that
    cancels to ``0.0315``, i.e. 293x below scale. A plain ``assert_allclose``
    demands ``1e-8 + 1e-6 * 0.0315 = 4.15e-8`` at that one element against
    ``1e-8 + 1e-6 * 9.22 = 9.2e-6`` at its neighbours -- a 222x tighter
    absolute pin on the SAME array, produced by the cancellation and nothing
    else. The observed 4.29e-8 disagreement is 4.6e-9 of the array's own scale,
    three orders INSIDE the strict fp64 ``rtol``, and it fails that pin.

    The result is one UNIFORM absolute envelope for the whole array, so no
    element is privileged by how much its own mean happened to cancel. This is
    a rebasing, not a widening: on the same measured array it moves the
    detection threshold of the UNCANCELLED elements from ``9.2e-6`` to
    ``1.18e-5`` (a factor 1.28, because ``max|e_L|`` exceeds the median
    ``|e_L|`` by that much) while removing the cancelled element's spurious
    222x tightening. Regression sensitivity on the array as a whole is
    therefore preserved, without the accidental hair trigger.

    Args:
        atol: The base absolute tolerance, kept as an additive floor.
        rtol: The base relative tolerance, reinterpreted against
            ``reference_scale`` instead of against each element.
        reference_scale: The magnitude of the terms the mean is built from --
            in practice ``max(abs(reference_array))`` of the quantity being
            compared.

    Returns:
        ``(atol + rtol * reference_scale, 0.0)`` -- an ``assert_allclose`` pair
        whose relative component is deliberately zero, because the relative
        budget has already been spent, once, on the array's own scale.
    """
    return atol + rtol * float(reference_scale), 0.0


def get_tolerance_min(zones: Iterable[str], level: str = "strict") -> tuple[float, float]:
    """Return ``(atol, rtol)`` loose enough for the lowest-precision zone.

    Use for comparing two exact computations whose path crosses
    multiple zones; the achievable agreement is bounded by the
    weakest (largest tolerance) zone on the path.

    Args:
        zones: Iterable of Precision Zone names.
        level: ``"strict"`` or ``"medium"``.

    Returns:
        ``(atol, rtol)`` tuple using the maximum of each component.
    """
    atols, rtols = zip(*(get_tolerance(z, level) for z in zones))
    return max(atols), max(rtols)


def get_tolerance_reduction_order(zones: Iterable[str], level: str = "medium") -> tuple[float, float]:
    """``(atol, rtol)`` for two routes differing only in reduction order.

    In FULL precision both routes run the same fp64 arithmetic, so the only
    difference is summation order: the machine-epsilon-class pair
    ``(ATOL_REDUCTION_ORDER, RTOL_REDUCTION_ORDER)`` applies and a real
    regression cannot hide behind a physical-zone budget. In MIXED
    precision the dense reference may run in an fp32 forward zone while the
    optimized route keeps fp64, so the ordinary zone budget still governs.

    Args:
        zones: Zone name or iterable of zone names crossed by the quantity.
        level: Zone level used only in mixed precision.

    Returns:
        The ``(atol, rtol)`` pair to use for this comparison.
    """
    from ._setting import ATOL_REDUCTION_ORDER, RTOL_REDUCTION_ORDER

    if is_mixed_precision_enabled():
        return get_tolerance_min(zones, level)
    return (ATOL_REDUCTION_ORDER, RTOL_REDUCTION_ORDER)


def get_tolerance_xla_rust_consistency(zones: str | Iterable[str], level: str = "strict") -> tuple[float, float]:
    """Return the dedicated XLA-vs-Rust backend-consistency tolerance.

    The canonical table and implementation live in :mod:`jqmc._setting`; this
    forwarding accessor keeps zone-related test APIs available from one module.
    """
    from jqmc._setting import get_tolerance_xla_rust_consistency as _get

    return _get(zones, level=level)


def mode_label() -> str:
    """Return a short label for the active precision mode.

    Returns:
        ``"Full Precision (FP64)"`` or ``"Mixed Precision (FP32 + FP64)"``.
    """
    if is_mixed_precision_enabled():
        return "Mixed Precision (FP32 + FP64)"
    return "Full Precision (FP64)"


def zone_detail() -> str:
    """Return a per-zone detail string of the current precision configuration."""
    lines = []
    for zone in sorted(ALL_ZONES):
        lines.append(f"  {zone}: {_get_zone_str(zone)}")
    return "\n".join(lines)
