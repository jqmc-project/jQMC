"""Binding energies E_dimer - E_monomer1 - E_monomer2 from the Final output blocks."""

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

import math
import re
from pathlib import Path

HA_TO_KCAL = 627.509
SYSTEMS = ("03_S22_water_dimer", "01_S22_water_monomer_1", "02_S22_water_monomer_2")
RUNS = (
    ("JSD", "VMC", "03mcmc_JSD/out_mcmc"),
    ("JSD", "LRDMC", "04lrdmc_JSD/out_lrdmc"),
    ("JAGP", "VMC", "06mcmc_JAGP/out_mcmc"),
    ("JAGP", "LRDMC", "07lrdmc_JAGP/out_lrdmc"),
)
ENERGY = re.compile(r"Total Energy: E = (\S+) \+- (\S+) Ha")


def final_energy(path: Path) -> tuple[float, float]:
    """Return the last 'Total Energy' line of a jqmc output."""
    mean, error = ENERGY.findall(path.read_text())[-1]
    return float(mean), float(error)


root = Path(__file__).resolve().parent
for ansatz, method, output in RUNS:
    (e_d, s_d), (e_1, s_1), (e_2, s_2) = (final_energy(root / system / output) for system in SYSTEMS)
    e_bind = e_d - e_1 - e_2
    s_bind = math.sqrt(s_d**2 + s_1**2 + s_2**2)
    print(
        f"{ansatz:4s} {method:5s}  E_bind = {e_bind * 1000:+.2f} +- {s_bind * 1000:.2f} mHa"
        f" = {e_bind * HA_TO_KCAL:+.2f} +- {s_bind * HA_TO_KCAL:.2f} kcal/mol"
    )
