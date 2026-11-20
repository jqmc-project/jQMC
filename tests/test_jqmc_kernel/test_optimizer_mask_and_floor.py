"""Real-wavefunction mask invariance, SR diagonal-floor freeze and SR-recompute faults.

Three groups of deterministic checks, all on configurations drawn by the
production sampler (``MCMC``) for H2 all-electron cc-pVDZ:

1. Mask invariance. Wavefunctions J1 (Pade 1.0) + J2 (Pade 0.5) + J3 (AO,
   seeded) + geminal, the geminal being either the dense AGP ``lambda_matrix``
   (``agp_lambda``) or the time-reversal SD ``mo_coefficients_up`` block
   (``sd_mo``). The production sampler is run with every block active and,
   from the same seed, with exactly one block active. The walk does not depend
   on the mask (positions are bit-identical) and the derivative source the
   driver publishes for the active block must equal, exactly, the one it
   publishes for that block under the all-active mask. Both c-axis providers
   are covered: the default analytic kernels and the masked-autograd twin
   (``analytic=False``) in which ``with_param_grad_mask`` applies
   ``stop_gradient`` inside the differentiated function. The all-active
   publication is anchored to the independent per-coordinate derivative of
   ``ln Psi`` (``_optimizer_step_reference.coordinate_columns``).
2. Automatic SR diagonal-floor freeze on a real ``run_optimize`` step. With
   J2 = 100 the J2 log-derivative is nearly constant over the sample, so its
   centred variance lies below ``min_S_diag_abs`` (but is not zero), while J1
   stays above it. The recomputed ``diag S`` decides the frozen set; the
   production frozen groups, metric diagonal, update (``q_J2 = 0``, ``q_J1``
   the J1-only regularized solve) and applied change must match it.
3. Faults the NumPy SR recompute of ``test_optimizer_real_batch_update`` must
   detect in both precision modes: a dropped centring of O in the production
   SR sample builder, and a ``stop_gradient`` leaking from a frozen Jastrow
   block onto the shared Jastrow container inside the mask path (masked
   autograd provider, J2 frozen).

No sampling-error band is used anywhere.
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

import contextlib
import os
from dataclasses import dataclass

import jax
import numpy as np
import pytest

import jqmc.derivatives.bridge as bridge_module
import jqmc.jqmc_mcmc as mcmc_module
import jqmc.jqmc_optimizer as optimizer_module
from jqmc._diff_mask import DiffMask, apply_diff_mask
from jqmc._precision import get_tolerance_min, is_mixed_precision_enabled
from jqmc._setting import ATOL_PYTHON_FP64_LOOSE, RTOL_PYTHON_FP64_LOOSE, min_S_diag_abs
from jqmc.determinant import to_AGP
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.jastrow_factor import Jastrow_data, Jastrow_one_body_data, Jastrow_three_body_data, Jastrow_two_body_data
from jqmc.jqmc_mcmc import MCMC
from jqmc.jqmc_optimizer import ParameterSelection
from jqmc.trexio_wrapper import read_trexio_file
from jqmc.wavefunction import Wavefunction_data
from tests.test_jqmc_kernel import test_optimizer_real_batch_update as real_batch
from tests.test_jqmc_kernel._optimizer_step_reference import (
    LN_PSI_ZONES,
    LOCAL_ENERGY_ZONES,
    StepCapture,
    coordinate_columns,
    independent_local_energies,
    record_one_optimize_step,
    reference_sr_update,
)

pytestmark = [pytest.mark.with_chdir]

jax.config.update("jax_enable_x64", True)

_TREXIO_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trexio_example_files", "H2_ae_ccpvdz_cart.h5")
_SEED = 20260928
_MAX_TIME = 86400

# --------------------------------------------------------------------------- #
# Shared builders
# --------------------------------------------------------------------------- #


def _jastrow(structure_data, aos, *, j2_param, with_j3):
    return Jastrow_data(
        jastrow_one_body_data=Jastrow_one_body_data.init_jastrow_one_body_data(
            jastrow_1b_param=1.0,
            structure_data=structure_data,
            core_electrons=tuple(0.0 for _ in structure_data.atomic_numbers),
            jastrow_1b_type="pade",
        ),
        jastrow_two_body_data=Jastrow_two_body_data.init_jastrow_two_body_data(
            jastrow_2b_param=j2_param, jastrow_2b_type="pade"
        ),
        jastrow_three_body_data=(
            Jastrow_three_body_data.init_jastrow_three_body_data(orb_data=aos, random_init=True, random_scale=0.05, seed=1)
            if with_j3
            else None
        ),
    )


def _build_hamiltonian(*, geminal: str, j2_param: float, with_j3: bool) -> Hamiltonian_data:
    structure_data, aos, _mos_up, _mos_dn, geminal_data, coulomb_potential_data = read_trexio_file(
        trexio_file=_TREXIO_FILE, store_tuple=True
    )
    if geminal == "agp_lambda":
        geminal_data = to_AGP(geminal_data)
    else:
        assert geminal == "sd_mo"
    return Hamiltonian_data(
        structure_data=structure_data,
        coulomb_potential_data=coulomb_potential_data,
        wavefunction_data=Wavefunction_data(
            jastrow_data=_jastrow(structure_data, aos, j2_param=j2_param, with_j3=with_j3),
            geminal_data=geminal_data,
        ),
    )


@contextlib.contextmanager
def _c_axis_provider(path: str):
    """Select the driver's c-axis provider: analytic kernels or the masked-autograd twin."""
    assert path in ("analytic", "autograd")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mcmc_module, "_ANALYTIC_GRAD_ENABLED", path == "analytic")
        yield


_JASTROW_FLAGS = ("opt_J1_param", "opt_J2_param", "opt_J3_param")


@contextlib.contextmanager
def _leaking_jastrow_mask():
    """Production mutation: a frozen Jastrow block stop-gradients the SHARED Jastrow container.

    The patched ``with_param_grad_mask`` first applies the production mask and
    then, when any Jastrow block is frozen, applies ``stop_gradient`` to every
    parameter leaf of the whole ``Jastrow_data`` (the intermediate shared by
    J1/J2/J3), so the still-active Jastrow blocks lose their derivative. The
    compiled c-axis programs are dropped before and after, because the mask
    runs at trace time.
    """
    production_mask = Wavefunction_data.with_param_grad_mask

    def leaking_mask(self, **flags):
        masked = production_mask(self, **flags)
        if masked.jastrow_data is not None and not all(flags.get(flag, True) for flag in _JASTROW_FLAGS):
            masked = masked.replace(jastrow_data=apply_diff_mask(masked.jastrow_data, DiffMask(params=False, coords=True)))
        return masked

    bridge_module.cache_clear()
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Wavefunction_data, "with_param_grad_mask", leaking_mask)
            yield
    finally:
        bridge_module.cache_clear()


# --------------------------------------------------------------------------- #
# 1. Mask invariance of the published block derivative
# --------------------------------------------------------------------------- #

_MASK_FLAGS = ("opt_J1_param", "opt_J2_param", "opt_J3_param", "opt_geminal")
_MASK_WAVEFUNCTIONS = ("agp_lambda", "sd_mo")
_MASK_PATHS = ("analytic", "autograd")
_MASK_WALKERS = 4
_MASK_STEPS = 5
_MASK_BIN_BLOCKS = 5
_GEMINAL_BLOCK = {"agp_lambda": "lambda_matrix", "sd_mo": "mo_coefficients_up"}


def _block_name(wavefunction: str, flag: str) -> str:
    return {"opt_J1_param": "j1_param", "opt_J2_param": "j2_param", "opt_J3_param": "j3_matrix"}.get(
        flag, _GEMINAL_BLOCK[wavefunction]
    )


@dataclass(frozen=True)
class _MaskRun:
    hamiltonian: Hamiltonian_data
    r_up: np.ndarray
    r_dn: np.ndarray
    coordinate_map: object
    sources: dict


_MASK_CACHE: dict[tuple, _MaskRun] = {}


def _sample_with_mask(wavefunction: str, path: str, active: tuple[str, ...]) -> _MaskRun:
    """Production sampler with only ``active`` blocks enabled; the driver's published derivative sources."""
    hamiltonian = _build_hamiltonian(geminal=wavefunction, j2_param=0.5, with_j3=True)
    selection = ParameterSelection(**{flag: flag in active for flag in _MASK_FLAGS})
    positions_up, positions_dn = [], []
    with _c_axis_provider(path):
        mcmc = MCMC(
            hamiltonian_data=hamiltonian,
            Dt=2.0,
            mcmc_seed=_SEED,
            num_walkers=_MASK_WALKERS,
            comput_position_deriv=False,
            comput_log_WF_param_deriv=True,
            comput_e_L_param_deriv=False,
        )
        mcmc.select_parameters(selection.gradient_flag_kwargs())
        for _ in range(_MASK_STEPS):
            mcmc.sample(num_steps=1, max_time=_MAX_TIME)
            positions_up.append(np.asarray(mcmc._MCMC__latest_r_up_carts))
            positions_dn.append(np.asarray(mcmc._MCMC__latest_r_dn_carts))
        blocks = mcmc.variational_blocks(selection)
        batch = mcmc.optimization_batch(
            num_warmup_steps=0, num_bin_blocks=_MASK_BIN_BLOCKS, blocks=blocks, need_local_energy_derivatives=False
        )
    n_samples = _MASK_STEPS * _MASK_WALKERS
    return _MaskRun(
        hamiltonian=hamiltonian,
        r_up=np.stack(positions_up).reshape(n_samples, *positions_up[0].shape[1:]),
        r_dn=np.stack(positions_dn).reshape(n_samples, *positions_dn[0].shape[1:]),
        coordinate_map=optimizer_module.OptimizationCoordinateMap.from_blocks(blocks),
        sources={source.name: source for source in batch.log_derivative_sources},
    )


def _mask_run(wavefunction: str, path: str, active: tuple[str, ...]) -> _MaskRun:
    key = (wavefunction, path, active, is_mixed_precision_enabled())
    if key not in _MASK_CACHE:
        _MASK_CACHE[key] = _sample_with_mask(wavefunction, path, active)
    return _MASK_CACHE[key]


def _assert_mask_invariant(all_on: _MaskRun, single: _MaskRun, block_name: str) -> None:
    """Same walk; the single-active publication of ``block_name`` equals the all-active one exactly."""
    np.testing.assert_array_equal(single.r_up, all_on.r_up)
    np.testing.assert_array_equal(single.r_dn, all_on.r_dn)
    assert list(single.sources) == [block_name]
    for component in ("direct", "d_p", "d_pbar"):
        expected = getattr(all_on.sources[block_name], component)
        actual = getattr(single.sources[block_name], component)
        assert (actual is None) == (expected is None), f"{block_name}.{component} presence depends on the mask"
        if expected is not None:
            np.testing.assert_array_equal(
                np.asarray(actual),
                np.asarray(expected),
                err_msg=f"{block_name}.{component} derivative depends on which other blocks are active",
            )


@pytest.mark.light
@pytest.mark.parametrize("path", _MASK_PATHS)
@pytest.mark.parametrize("wavefunction", _MASK_WAVEFUNCTIONS)
def test_all_active_block_derivatives_match_independent_coordinate_derivatives(wavefunction, path):
    """With every block active, each published block derivative equals the independent d ln Psi / dq."""
    run = _mask_run(wavefunction, path, _MASK_FLAGS)
    names = [block.name for block in run.coordinate_map.blocks]
    assert names == [_block_name(wavefunction, flag) for flag in _MASK_FLAGS]
    production = np.asarray(run.coordinate_map.transform_derivatives([run.sources[name] for name in names]))
    reference = coordinate_columns(run.hamiltonian, run.coordinate_map, run.r_up, run.r_dn, "ln_psi")
    atol, rtol = get_tolerance_min(LN_PSI_ZONES, "strict")
    np.testing.assert_allclose(production, reference, rtol=rtol, atol=atol)
    # Non-vacuity: every block's derivative is non-zero on the sample.
    for block_slice, name in zip(run.coordinate_map.block_coordinate_slices, names, strict=True):
        assert np.max(np.abs(reference[:, block_slice])) > atol, f"{name} derivative vanishes on the sample"


@pytest.mark.light
@pytest.mark.parametrize("flag", _MASK_FLAGS)
@pytest.mark.parametrize("path", _MASK_PATHS)
@pytest.mark.parametrize("wavefunction", _MASK_WAVEFUNCTIONS)
def test_block_log_derivative_is_mask_invariant(wavefunction, path, flag):
    """d ln Psi / d(block) published with only that block active equals the all-active publication, exactly."""
    all_on = _mask_run(wavefunction, path, _MASK_FLAGS)
    single = _mask_run(wavefunction, path, (flag,))
    _assert_mask_invariant(all_on, single, _block_name(wavefunction, flag))


@pytest.mark.light
@pytest.mark.parametrize("flag", _JASTROW_FLAGS)
def test_negative_control_leaking_jastrow_stop_gradient_breaks_mask_invariance(flag):
    """A stop_gradient leaking onto the shared Jastrow container in the mask path fails the invariance verdict."""
    all_on = _mask_run("sd_mo", "autograd", _MASK_FLAGS)
    with _leaking_jastrow_mask():
        faulty = _sample_with_mask("sd_mo", "autograd", (flag,))
    with pytest.raises(AssertionError, match="derivative depends on which other blocks are active"):
        _assert_mask_invariant(all_on, faulty, _block_name("sd_mo", flag))


# --------------------------------------------------------------------------- #
# 2. Automatic SR diagonal-floor freeze on a real run_optimize step
# --------------------------------------------------------------------------- #

# J2 = 100 makes d u_2 / d b ~ -1 / (2 b^2) almost configuration independent, so the
# centred J2 variance is O(1e-13): below min_S_diag_abs = 1e-10 but not zero.
_FLOOR_J2 = 100.0
_FLOOR_EPSILON = 1.0e-3
_FLOOR_DELTA = 0.05
_FLOOR_CACHE: dict[bool, StepCapture] = {}


def _floor_capture() -> StepCapture:
    key = is_mixed_precision_enabled()
    if key not in _FLOOR_CACHE:
        _FLOOR_CACHE[key] = record_one_optimize_step(
            _build_hamiltonian(geminal="sd_mo", j2_param=_FLOOR_J2, with_j3=False),
            optimizer_kwargs={"method": "sr", "delta": _FLOOR_DELTA, "epsilon": _FLOOR_EPSILON, "cg_flag": False},
            selection={"opt_J1_param": True, "opt_J2_param": True, "opt_J3_param": False, "opt_geminal": False},
            use_lm=False,
            seed=_SEED,
            num_walkers=real_batch._NUM_WALKERS,
            num_steps=real_batch._NUM_STEPS,
            num_bin_blocks=real_batch._NUM_BIN_BLOCKS,
        )
    return _FLOOR_CACHE[key]


def _assert_floor_freeze(capture: StepCapture, target) -> None:
    """The production frozen set, metric diagonal and update equal the recomputed floor decision."""
    q_ref, diagonal, active = reference_sr_update(
        capture.log_derivatives, capture.local_energies, capture.weights, _FLOOR_EPSILON, min_S_diag_abs
    )
    expected_frozen = tuple((index,) for index in np.flatnonzero(~active))
    assert target.frozen_coordinate_groups == expected_frozen, (
        f"production frozen groups {target.frozen_coordinate_groups} != recomputed floor decision {expected_frozen}"
    )
    np.testing.assert_allclose(
        np.asarray(target.metric_diagonal),
        diagonal,
        rtol=RTOL_PYTHON_FP64_LOOSE,
        atol=ATOL_PYTHON_FP64_LOOSE,
        err_msg="production metric diagonal differs from the recomputed diag S",
    )
    real_batch._assert_update_matches_reference(target.q, q_ref, "floor-frozen SR")


@pytest.mark.light
def test_sr_diagonal_floor_freezes_the_low_variance_block():
    """Real SR step: J2 (0 < diag S < floor) is frozen, J1 solves the restricted system, J2 is untouched."""
    capture = _floor_capture()
    assert [block.name for block in capture.coordinate_map.blocks] == ["j1_param", "j2_param"]
    atol, rtol = get_tolerance_min(LOCAL_ENERGY_ZONES, "strict")
    np.testing.assert_allclose(
        capture.local_energies,
        independent_local_energies(capture.hamiltonian_before, capture.r_up, capture.r_dn),
        rtol=rtol,
        atol=atol,
    )
    ln_atol, ln_rtol = get_tolerance_min(LN_PSI_ZONES, "strict")
    np.testing.assert_allclose(
        capture.log_derivatives,
        coordinate_columns(capture.hamiltonian_before, capture.coordinate_map, capture.r_up, capture.r_dn, "ln_psi"),
        rtol=ln_rtol,
        atol=ln_atol,
    )
    _q_ref, diagonal, active = reference_sr_update(
        capture.log_derivatives, capture.local_energies, capture.weights, _FLOOR_EPSILON, min_S_diag_abs
    )
    # Non-vacuity: exactly J2 is below the floor, and not as a structural zero.
    assert active.tolist() == [True, False]
    assert 0.0 < diagonal[1] <= min_S_diag_abs < diagonal[0]
    _assert_floor_freeze(capture, capture.target)

    before = real_batch._jastrow_parameters(capture.hamiltonian_before)
    after = real_batch._jastrow_parameters(capture.hamiltonian_after)
    real_batch._assert_frozen_parameters_unchanged(before, after, [0])
    param_atol, param_rtol = real_batch._PARAMETER_TOLERANCE
    np.testing.assert_allclose(
        after[0] - before[0], _FLOOR_DELTA * np.asarray(capture.target.q)[0], rtol=param_rtol, atol=param_atol
    )


def _replay(capture, **overrides):
    kwargs = dict(capture.target_kwargs)
    kwargs["warm_start"] = None
    kwargs.update(overrides)
    return optimizer_module.compute_real_coordinate_optimization_target(capture.batch, capture.coordinate_map, **kwargs)


@pytest.mark.light
def test_negative_control_disabled_diagonal_floor_fails_freeze_check():
    """Replaying the production target without the diagonal floor un-freezes J2 and fails the verdict."""
    capture = _floor_capture()
    _assert_floor_freeze(capture, _replay(capture))
    with pytest.raises(AssertionError, match="production frozen groups"):
        _assert_floor_freeze(capture, _replay(capture, diagonal_floor=0.0))


@contextlib.contextmanager
def _uncentred_sr_samples():
    """Production mutation: the SR sample builder drops the centring of O (O_bar = 0)."""
    production_build = optimizer_module.build_sr_coordinate_samples

    def uncentred(batch, coordinate_map, *, total_weight=None, total_weighted_log_derivatives=None, total_weighted_energy=None):
        if total_weight is None:
            weights = np.asarray(batch.weights)
            total_weight = np.sum(weights)
            total_weighted_energy = np.dot(weights, np.asarray(batch.local_energies))
        return production_build(
            batch,
            coordinate_map,
            total_weight=total_weight,
            total_weighted_log_derivatives=np.zeros(coordinate_map.coordinate_size),
            total_weighted_energy=total_weighted_energy,
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(optimizer_module, "build_sr_coordinate_samples", uncentred)
        yield


@pytest.mark.light
def test_negative_control_dropped_centring_fails_freeze_check():
    """Without the centring of O, the J2 second moment exceeds the floor and the freeze verdict fails."""
    capture = _floor_capture()
    with _uncentred_sr_samples():
        mutated = _replay(capture)
    with pytest.raises(AssertionError, match="production frozen groups"):
        _assert_floor_freeze(capture, mutated)


# --------------------------------------------------------------------------- #
# 3. Faults the real-batch NumPy SR recompute must detect
# --------------------------------------------------------------------------- #


@pytest.mark.light
def test_negative_control_dropped_centring_fails_real_batch_sr_recompute():
    """The dense SR recompute rejects a production SR builder that does not centre O."""
    capture = real_batch._capture("sr_eps_small_host")
    reference_q = real_batch._reference_sr_update(
        capture.log_derivatives, capture.local_energies, capture.weights, capture.case.epsilon
    )
    real_batch._assert_update_matches_reference(real_batch._replay_target(capture).q, reference_q, "SR")
    with _uncentred_sr_samples():
        mutated = real_batch._replay_target(capture)
    with pytest.raises(AssertionError, match="production SR update differs from the dense reference"):
        real_batch._assert_update_matches_reference(mutated.q, reference_q, "SR")


_AUTOGRAD_CACHE: dict[bool, StepCapture] = {}


def _record_j2_frozen_autograd() -> StepCapture:
    with _c_axis_provider("autograd"):
        return record_one_optimize_step(
            real_batch._build_hamiltonian(),
            optimizer_kwargs={"method": "sr", "delta": real_batch._DELTA, "epsilon": 0.2, "cg_flag": False},
            selection={"opt_J1_param": True, "opt_J2_param": False, "opt_J3_param": False, "opt_geminal": False},
            use_lm=False,
            seed=real_batch._SEED,
            num_walkers=real_batch._NUM_WALKERS,
            num_steps=real_batch._NUM_STEPS,
            num_bin_blocks=real_batch._NUM_BIN_BLOCKS,
        )


def _j2_frozen_autograd_capture() -> StepCapture:
    key = is_mixed_precision_enabled()
    if key not in _AUTOGRAD_CACHE:
        _AUTOGRAD_CACHE[key] = _record_j2_frozen_autograd()
    return _AUTOGRAD_CACHE[key]


def _assert_real_batch_sr_chain(capture: StepCapture, epsilon: float) -> None:
    """Consumed batch == independent evaluation, then target.q == dense SR solve; J2 bit-identical."""
    evaluate = real_batch._independent_evaluator(capture)
    before = real_batch._jastrow_parameters(capture.hamiltonian_before)
    atol, rtol = get_tolerance_min(real_batch._SOURCE_ZONES, "strict")
    np.testing.assert_allclose(capture.local_energies, evaluate("local_energy", before), rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        np.real(capture.log_derivatives),
        evaluate("d_ln_psi", before)[:, [0]],
        rtol=rtol,
        atol=atol,
        err_msg="consumed log derivatives differ from the independent evaluation",
    )
    reference_q = real_batch._reference_sr_update(capture.log_derivatives, capture.local_energies, capture.weights, epsilon)
    real_batch._assert_update_matches_reference(capture.target.q, reference_q, "SR")
    real_batch._assert_frozen_parameters_unchanged(before, real_batch._jastrow_parameters(capture.hamiltonian_after), [0])


@pytest.mark.light
def test_masked_autograd_j2_frozen_step_matches_sr_recompute():
    """Masked-autograd provider, J2 frozen: consumed J1 derivative, SR update and frozen J2 pass the recompute chain."""
    capture = _j2_frozen_autograd_capture()
    assert [block.name for block in capture.coordinate_map.blocks] == ["j1_param"]
    assert capture.target.frozen_coordinate_groups == ()
    _assert_real_batch_sr_chain(capture, 0.2)


@pytest.mark.light
def test_negative_control_leaking_jastrow_stop_gradient_fails_sr_recompute():
    """A stop_gradient leaking onto the shared Jastrow container (J2 frozen) fails the SR recompute chain."""
    with _leaking_jastrow_mask():
        faulty = _record_j2_frozen_autograd()
    with pytest.raises(AssertionError, match="consumed log derivatives differ from the independent evaluation"):
        _assert_real_batch_sr_chain(faulty, 0.2)
