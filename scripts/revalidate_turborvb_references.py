"""Revalidate historical reference literals with an external TurboRVB build.

This opt-in provenance tool never imports jQMC or rewrites tests, SSoT, or
tolerances. It runs one external forward evaluation at each fixed configuration,
not a Monte Carlo acceptance experiment. Output is a new, self-contained run
directory; this is not a GDD completion certificate.
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

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "16ad828cf3efa9abc946f51edb3f0ce45c7d691bc0ab20dd5a103af6c1382eed"
LIBRARIES = (
    "libcomplex-serial.a",
    "libadjoint-backward-serial.a",
    "libadjoint-forward-serial.a",
    "libqlapack-serial.a",
    "libcommon-serial.a",
    "libpfapack.a",
)
OVERRIDE = """
    open(unit=99, file='electron_positions.dat', status='old')
    do i=1,nel
        read(99,*) kel(1:3,i)
    end do
    close(99)
    kelind(:,1:nel) = kel(:,1:nel)
    open(unit=99, file='jastrow_mode.dat', status='old')
    read(99,*) j
    close(99)
    if (j.eq.0.or.j.eq.2) then
        jasmat=0.d0
        jasmatsz=0.d0
        jasmat_c=0.d0
        jasmatsz_c=0.d0
    endif
    if (j.eq.0) then
        iesdr=0
        n_body_on=0
    endif
"""
OUTPUT = """
    if (iflagerr.ne.0) stop 2
    write(6,'(A,3ES26.17E3)') 'GDD_RAW_RY_LOGPSI ', eloc(1), vpot(1), logpsi(1)
    call subener(indt, nelup, neldo, winvup, winvdo, tabpip, eloc)
    write(6,'(A,ES26.17E3)') 'GDD_KINETIC_RY ', eloc(1)
    write(6,'(A,*(ES26.17E3))') 'GDD_G_INVERSE ', ainv(1:nelup*nelup)
    stop
"""
NAMELIST = """&simulation
ngen=1, nw=1, nbra=1, iseedr=25113183,
yes_sparse=.false., yes_sparse_choose=.false.
/
&pseudo
nintpsa=6, npsamax=3, pseudorandom=.false.
/
&vmc
tstep=2.d0, epscut=0.d0
/
&readio
/
&parameters
ieskin=1
/
"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args, cwd, **kwargs):
    return subprocess.run(
        list(map(str, args)),
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        timeout=120,
        **kwargs,
    ).stdout


def pseudo_input(inputs, turbo, work):
    """Mirror TurboGenius serialization, then ask TurboRVB for its cutoff."""
    with h5py.File(inputs / "water_ccecp_ccpvqz.h5") as file:
        arrays = {key: value[()] for key, value in file["ecp"].items()}
    result = ["ECP\n"]
    for ion in range(3):
        rows, counts = [], []
        for ell in range(int(arrays["ecp_max_ang_mom_plus_1"][ion]) + 1):
            indices = [i for i, n in enumerate(arrays["ecp_nucleus_index"]) if n == ion and arrays["ecp_ang_mom"][i] == ell]
            counts.append(len(indices))
            for i in indices:
                rows.append(
                    "{:14f} {:14f} {:14f}\n".format(
                        arrays["ecp_coefficient"][i], arrays["ecp_power"][i] + 2, arrays["ecp_exponent"][i]
                    )
                )
        body = " ".join(map(str, counts)) + "\n" + "".join(rows)
        (work / "pseudo.dat").write_text("ECP\n1 0 2\n" + body)
        output = command([turbo / "bin/pseudo.x"], work, input="0.00001\n")
        (work / f"pseudo-cutoff-{ion}.out").write_text(output)
        cutoff = float(next(line for line in output.splitlines() if "Suggested cut-off pseudo" in line).split()[-1])
        result.append(f"{ion + 1} {cutoff:.6f} 2\n" + body)
    (work / "pseudo.dat").write_text("".join(result))


def source_inputs(fun):
    """Evaluate only fixed input/reference assignments, never test calls."""
    namespace = {"np": np}
    for statement in fun.body:
        if not isinstance(statement, ast.Assign):
            continue
        target = statement.targets[0]
        base = target.value if isinstance(target, ast.Subscript) else target
        if not isinstance(base, ast.Name):
            continue
        if base.id.startswith(("old_r_", "new_r_", "R_AS_turborvb_")) or base.id.endswith(("_turborvb", "_turbo")):
            exec(compile(ast.Module(body=[statement], type_ignores=[]), "<committed-reference-input>", "exec"), namespace)
    return namespace


def compare_references(ns, outputs, inputs):
    old, new = outputs["old"], outputs["new"]
    energy, _, logpsi = new["GDD_RAW_RY_LOGPSI"]
    kinetic = new["GDD_KINETIC_RY"][0] / 2
    ratio = math.exp(2 * (logpsi - old["GDD_RAW_RY_LOGPSI"][2]))
    actual = {"kinetic": kinetic, "potential": energy / 2 - kinetic}
    refs = {"kinetic": ns["kinc_ref_turborvb"], "potential": ns["vpot_ref_turborvb"] + ns.get("vpotoff_ref_turborvb", 0)}
    if "geminal_old_inv_turborvb" in ns:
        radii = {}
        for side in ("old", "new"):
            inverse = outputs[side]["GDD_G_INVERSE"].reshape((4, 4), order="F")
            geminal = np.linalg.inv(inverse)
            scale = np.min(np.sum(geminal**2, axis=0))
            frobenius = np.sum(inverse**2)
            radii[side] = (scale * frobenius) ** (-3 / 8)
            for name, value, ref in (
                ("geminal", geminal, ns[f"geminal_{side}_turborvb"]),
                ("inverse", inverse, ns[f"geminal_{side}_inv_turborvb"]),
                ("S", scale, ns[f"S_{side}_turborvb"]),
                ("F", frobenius, ns[f"F_{side}_turborvb"]),
                ("R_AS", radii[side], ns[f"R_AS_turborvb_{side}"]),
            ):
                actual[f"{side}_{name}"] = value
                refs[f"{side}_{name}"] = ref
        ratio *= ((max(radii["new"], 0.3) / radii["new"]) / (max(radii["old"], 0.3) / radii["old"])) ** 2
        actual["reweight"] = (radii["new"] / max(radii["new"], 0.3)) ** 2
        refs["reweight"] = ns["reweight_turborvb"]
        with h5py.File(inputs / "water_ccecp_ccpvqz.h5") as file:
            positions = file["nucleus/nucleus_coord"][()]
            # This committed TREXIO stores valence charges (O=6, H=1), not Z.
            charges = file["nucleus/nucleus_charge"][()]
        factors = {}
        for side in ("old", "new"):
            point = ns[f"{side}_r_up_carts"][2]
            distances = np.linalg.norm(positions - point, axis=1)
            index = np.argmin(distances)
            radius, charge = distances[index], charges[index]
            factors[side] = (1 + charge**2 * radius) / (charge**2 * (1 + radius))
        fa, fb = factors["old"], factors["new"]
        distance2 = np.sum((ns["new_r_up_carts"][2] - ns["old_r_up_carts"][2]) ** 2)
        transition = fa / fb * math.exp(-distance2 * (1 / fb**2 - 1 / fa**2) / 8)
        for name, value in (("fa", fa), ("fb", fb), ("T_ratio", transition), ("final_ratio", ratio * transition)):
            actual[name], refs[name] = value, ns[f"{name}_turbo"]
    actual["wavefunction_ratio"], refs["wavefunction_ratio"] = ratio, ns["WF_ratio_ref_turborvb"]
    return {
        name: {
            "actual": np.asarray(value).tolist(),
            "historical": np.asarray(refs[name]).tolist(),
            "max_absolute_difference": float(np.max(np.abs(np.asarray(value) - refs[name]))),
        }
        for name, value in actual.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turborvb-root", type=Path, required=True)
    parser.add_argument("--blas", type=Path, required=True)
    parser.add_argument("--compiler", default="gfortran")
    parser.add_argument("--output-parent", type=Path, default=Path(tempfile.gettempdir()))
    args = parser.parse_args()
    turbo = args.turborvb_root.resolve()
    source = turbo / "src/a_module_tests/testad.f90"
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Unsupported testad source: review the harness extraction before changing its pin")
    work = Path(tempfile.mkdtemp(prefix="gdd-turborvb-reference-", dir=args.output_parent))
    print(work, flush=True)
    original = source.read_text().splitlines(keepends=True)
    forward = "".join(original[:398])
    marker = "    write (6, *) ' Initial electron coordinates '"
    if forward.count(marker) != 1 or original[2136].strip() != "contains":
        raise ValueError("Unexpected external harness layout")
    (work / "probe.f90").write_text(forward.replace(marker, OVERRIDE + marker) + OUTPUT + "".join(original[2136:]))
    libraries = [turbo / "build" / name for name in LIBRARIES]
    flags = [
        "-fallow-argument-mismatch",
        "-D_SIMD",
        "-D__FFTW",
        "-D__USE_INTERNAL_FFTW",
        f"-I{turbo / 'build/serial_modules'}",
        "-w",
        "-g",
        "-cpp",
        "-ffree-line-length-none",
        "-O2",
    ]
    commands = [
        [args.compiler, *flags, "-c", "probe.f90", "-o", "probe.o"],
        [args.compiler, "-fallow-argument-mismatch", "probe.o", "-o", "probe.x", *libraries, args.blas, "-lm", "-ldl"],
    ]
    for invocation in commands:
        command(invocation, work)
    (work / "datasvmc.d").write_text(NAMELIST)
    inputs = ROOT / "tests/trexio_example_files"
    pseudo_input(inputs, turbo, work)

    def portable(text):
        """Paths in the report relative to this checkout or as ``<turborvb>/...`` / ``<work>/...``: no local directories."""
        return str(text).replace(f"{work}/", "<work>/").replace(f"{ROOT}/", "").replace(f"{turbo}/", "<turborvb>/")

    report = {
        "kind": "external-fixed-configuration-revalidation-not-ci-certificate",
        "build_commands": [[portable(arg) for arg in cmd] for cmd in commands],
        "external_source_checkout": command(["git", "rev-parse", "HEAD"], turbo).strip(),
        "external_tracked_diff": command(["git", "diff", "HEAD", "--stat"], turbo),
        "cases": {},
    }
    hashed = [
        source,
        *libraries,
        args.blas,
        turbo / "bin/pseudo.x",
        work / "probe.f90",
        work / "probe.x",
        work / "pseudo.dat",
        work / "datasvmc.d",
        Path(__file__).resolve(),
    ]
    for module in ("AE", "ECP"):
        test = ROOT / f"tests/test_jqmc_kernel/test_integration_turborvb_{module}.py"
        hashed.append(test)
        for fun in ast.walk(ast.parse(test.read_text())):
            if not isinstance(fun, ast.FunctionDef) or not fun.name.startswith("test_"):
                continue
            ns = source_inputs(fun)
            mode = 0 if "_wo_" in fun.name else 2 if "_w_2b_Jastrow" in fun.name else 1
            filename = (
                "turborvb_WF_w_1b_2b_1b3b_ae.txt"
                if module == "AE"
                else ("turborvb_WF_w_2b_3b_w_ecp.txt" if "_2b_3b_" in fun.name else "turborvb_WF_w_2b_1b3b_w_ecp.txt")
            )
            hashed.append(inputs / filename)
            shutil.copyfile(inputs / filename, work / "fort.10")
            (work / "jastrow_mode.dat").write_text(f"{mode}\n")
            outputs, coordinates = {}, {}
            for side in ("old", "new"):
                points = np.concatenate((ns[f"{side}_r_up_carts"], ns[f"{side}_r_dn_carts"]))
                coordinates[side] = points.tolist()
                np.savetxt(work / "electron_positions.dat", points, fmt="%.17g")
                with (work / "datasvmc.d").open() as inp:
                    output = command([work / "probe.x"], work, stdin=inp)
                log = work / f"{fun.name}-{side}.out"
                log.write_text(output)
                hashed.append(log)
                if "ERROR" in output or "GDD_RAW_RY_LOGPSI" not in output:
                    raise ValueError(f"External evaluation failed: {log}")
                outputs[side] = {
                    line.split()[0]: np.array([float(x) for x in line.split()[1:]])
                    for line in output.splitlines()
                    if line.startswith("GDD_")
                }
            report["cases"][fun.name] = {
                "input": filename,
                "jastrow_mode": mode,
                "coordinates": coordinates,
                "quantities": compare_references(ns, outputs, inputs),
            }
    hashed.extend(inputs / name for name in ("H2_ae_ccpvqz.h5", "water_ccecp_ccpvqz.h5"))
    report["sha256"] = {portable(path): digest(path) for path in hashed}
    (work / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for name, case in report["cases"].items():
        print(name, {key: value["max_absolute_difference"] for key, value in case["quantities"].items()}, flush=True)
    print("Report:", work / "report.json")


if __name__ == "__main__":
    main()
