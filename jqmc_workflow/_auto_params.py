"""Workflow-side ``'auto'`` resolution for execution parameters.

Policy lives here (jqmc-workflow), not in the driver: ``num_mcmc_per_measurement
= 'auto'`` resolves to ``4 * (num_electron_up + num_electron_dn)``, read from the
input Hamiltonian h5 via the jqmc *fact* helper
:func:`jqmc.hamiltonians.get_num_electrons`. The driver only ever receives a
concrete integer (see the CLI guard in ``jqmc.jqmc_cli``).
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
from logging import getLogger

logger = getLogger("jqmc-workflow").getChild(__name__)

# turboRVB-style default: each electron is proposed ~this many times per
# measurement, so num_mcmc_per_measurement scales with the electron count.
NUM_MCMC_PER_MEASUREMENT_ELECTRON_FACTOR = 4


def num_mcmc_per_measurement_from_electrons(num_electron_up: int, num_electron_dn: int) -> int:
    """Return ``'auto'`` num_mcmc_per_measurement = 4 * total electrons.

    Pure function (no I/O) so the policy is unit-testable without an h5.
    """
    total = int(num_electron_up) + int(num_electron_dn)
    return NUM_MCMC_PER_MEASUREMENT_ELECTRON_FACTOR * total


def resolve_num_mcmc_per_measurement(hamiltonian_h5_path: str) -> int:
    """Resolve ``'auto'`` by reading the electron count from a Hamiltonian h5.

    Args:
        hamiltonian_h5_path: path to the (single-twist) input ``Hamiltonian_data``
            HDF5 the run will sample.

    Returns:
        ``4 * (num_electron_up + num_electron_dn)`` as an int.
    """
    from jqmc.hamiltonians import get_num_electrons

    num_electron_up, num_electron_dn = get_num_electrons(hamiltonian_h5_path)
    return num_mcmc_per_measurement_from_electrons(num_electron_up, num_electron_dn)


def fit_Dt_for_acceptance(dts, acceptances, target=0.5):
    """Fit acceptance vs ``1/Dt`` and solve for the Dt hitting ``target``.

    Acceptance is ~inversely proportional to the step size (``A ~= k/Dt``), so --
    unlike LRDMC's plain-linear ``fit_num_projection_per_measurement`` -- the fit
    is done in ``x = 1/Dt``, where the relation is locally linear::

        A = a * (1/Dt) + b   ->   Dt* = 1 / ((target - b) / a)

    Args:
        dts: calibration step sizes (bohr); >= 2 distinct positive values.
        acceptances: measured acceptance ratios in [0, 1], one per Dt.
        target: target acceptance ratio (default 0.5).

    Returns:
        float: the Dt (bohr) predicted to hit ``target``.

    Raises:
        ValueError: fewer than 2 points, or mismatched lengths.
        RuntimeError: degenerate (identical Dt) or non-physical (flat slope /
            non-positive Dt) fit.
    """
    if len(dts) != len(acceptances):
        raise ValueError(f"dts and acceptances length mismatch: {len(dts)} vs {len(acceptances)}")
    if len(dts) < 2:
        raise ValueError(f"Need at least 2 data points, got {len(dts)}")

    xs = [1.0 / d for d in dts]
    ys = list(acceptances)
    n = len(xs)
    sx = sum(xs)
    sx2 = sum(x * x for x in xs)
    sy = sum(ys)
    sxy = sum(x * y for x, y in zip(xs, ys))

    denom = n * sx2 - sx * sx
    if abs(denom) < 1e-30:
        raise RuntimeError("Degenerate fit: all Dt values are identical.")

    a = (n * sxy - sx * sy) / denom
    b = (sy * sx2 - sx * sxy) / denom
    if abs(a) < 1e-15:
        raise RuntimeError(f"Fit slope ~0 (a={a:.6g}); cannot solve for target acceptance {target}.")

    x_target = (target - b) / a  # = 1 / Dt*
    if x_target <= 0:
        raise RuntimeError(
            f"Fit gives non-positive 1/Dt = {x_target:.6g} for target acceptance {target} "
            f"(a={a:.6g}, b={b:.6g}); widen the calibration Dt range."
        )
    # Round to 1 decimal place: the fit is approximate (short pilots, ~1/Dt
    # model), so extra digits are spurious precision and a clean value keeps the
    # continuation TOML / checkpoint comparison human-readable.
    return round(1.0 / x_target, 1)


# trial step sizes for the Dt='auto' calibration pilot (pilot_a). A modest
# spread spans the acceptance curve from high (small Dt) to low (large Dt) so the
# 1/Dt fit brackets the ~50% target; 4.0 pushes acceptance too low, so 1/2/3.
DT_CALIBRATION_TRIALS = (1.0, 2.0, 3.0)


async def run_dt_calibration(workflow, _wd, hamiltonian_file, pilot_steps, pilot_queue_label, target_acceptance, gen_input):
    """Shared pilot_a Dt calibration -- always via MCMC sampling jobs.

    The production driver may be MCMC or VMC, but Dt is a pure-sampling concept,
    so calibration always runs plain MCMC jobs (driver asymmetry: for VMC,
    pilot_a is MCMC while pilot_b / production is VMC). Runs one short
    MCMC job per trial Dt, measures each acceptance ratio, and fits
    acceptance-vs-``1/Dt`` to the ``target_acceptance``.

    Args:
        workflow: the calling workflow (for ``_submit_and_wait`` / ``_new_run_id``
            / ``_input_filename`` / ``_output_filename``, inherited from Workflow).
        _wd: project working directory (holds the input h5).
        hamiltonian_file: basename of the input h5 (symlinked into each pilot dir).
        pilot_steps: measurement steps per calibration run.
        pilot_queue_label: queue for the (short) calibration jobs.
        target_acceptance: target acceptance ratio to solve for.
        gen_input: callable ``(sub_dir, input_filename, dt)`` that writes an
            **mcmc-type** TOML with the given trial Dt.

    Returns:
        float: the fitted Dt hitting ``target_acceptance``.
    """
    import asyncio

    from .output_parser import parse_mcmc_output
    from .state import get_job_by_step

    h5_src = os.path.join(_wd, hamiltonian_file)
    pilot_a_dir = os.path.join(_wd, "_pilot_a")
    os.makedirs(pilot_a_dir, exist_ok=True)

    logger.info("")
    logger.info("-- pilot_a: calibrate Dt (MCMC sampling) " + "-" * 10)
    logger.info(f"  trial Dt = {list(DT_CALIBRATION_TRIALS)}, target acceptance = {target_acceptance:.2%}")

    async def _run_calib(sub_dir, dt_v, idx):
        os.makedirs(sub_dir, exist_ok=True)
        h5_link = os.path.join(sub_dir, hamiltonian_file)
        if os.path.isfile(h5_src) and not os.path.exists(h5_link):
            os.symlink(h5_src, h5_link)
        rec = get_job_by_step(sub_dir, 0)
        if rec.get("status") in ("fetched", "submitted", "completed"):
            inp_f, out_f, run_id_c = rec["input_file"], rec["output_file"], rec.get("run_id", "")
        else:
            run_id_c = workflow._new_run_id()
            inp_f = workflow._input_filename(0, run_id_c)
            out_f = workflow._output_filename(0, run_id_c)
            gen_input(sub_dir, inp_f, dt_v)
        logger.info(f"  _pilot{idx}: Dt={dt_v}, {pilot_steps} steps")
        await workflow._submit_and_wait(inp_f, out_f, work_dir=sub_dir, queue_label=pilot_queue_label, step=0, run_id=run_id_c)
        acc = parse_mcmc_output(sub_dir).acceptance_ratio
        logger.info(
            f"  _pilot{idx} (Dt={dt_v}): acceptance = {acc:.4f}"
            if acc is not None
            else f"  _pilot{idx} (Dt={dt_v}): acceptance = N/A"
        )
        return dt_v, acc

    tasks = [
        asyncio.create_task(_run_calib(os.path.join(pilot_a_dir, f"_pilot{i}"), dt_v, i))
        for i, dt_v in enumerate(DT_CALIBRATION_TRIALS, start=1)
    ]
    results = await asyncio.gather(*tasks)

    dts = [dt_v for dt_v, acc in results if acc is not None]
    accs = [acc for dt_v, acc in results if acc is not None]
    if len(dts) < 2:
        raise RuntimeError(f"Only {len(dts)} Dt-calibration run(s) returned an acceptance ratio; need >= 2 to fit.")

    dt_opt = fit_Dt_for_acceptance(dts, accs, target=target_acceptance)
    logger.info("")
    logger.info("-- Dt-calibration summary " + "-" * 24)
    for dt_v, acc in zip(dts, accs):
        logger.info(f"  Dt={dt_v}: acceptance = {acc:.4f}")
    logger.info(f"  target acceptance = {target_acceptance:.4f}")
    logger.info(f"  optimal Dt        = {dt_opt:.4f}")
    logger.info("-" * 50)
    return dt_opt


def resolve_workflow_num_mcmc_per_measurement(value, cached, hamiltonian_file, input_file):
    """Resolve a workflow's ``num_mcmc_per_measurement``, expanding ``'auto'``.

    Shared by ``MCMC_Workflow`` and ``VMC_Workflow`` so the 'auto' contract
    (read once, cache, reuse) is identical. A concrete int / ``None`` passes
    through. ``'auto'`` is resolved once from the input h5 staged next to the
    generated TOML and cached, so every continuation writes the same concrete
    int -- matching the checkpoint's frozen value.

    Args:
        value: the workflow's ``num_mcmc_per_measurement`` (int, ``None``, or ``'auto'``).
        cached: the previously-resolved int, or ``None`` if not yet resolved.
        hamiltonian_file: basename of the input h5 (relative to the TOML dir).
        input_file: path of the TOML being generated (its dir holds the h5).

    Returns:
        ``(resolved_value, new_cache)`` -- the caller stores ``new_cache``.

    Raises:
        FileNotFoundError: ``'auto'`` requested but the input h5 is not staged.
    """
    if value != "auto":
        return value, cached
    if cached is not None:
        return cached, cached
    h5_path = os.path.join(os.path.dirname(os.path.abspath(input_file)), hamiltonian_file)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"num_mcmc_per_measurement='auto' needs the input Hamiltonian h5 "
            f"'{h5_path}' to be staged before input generation; not found."
        )
    resolved = resolve_num_mcmc_per_measurement(h5_path)
    logger.info(f"  num_mcmc_per_measurement = {resolved} (auto = 4 * n_electron)")
    return resolved, resolved
