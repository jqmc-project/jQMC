"""run_qe2jqmc keeps explicitly supplied DTN sizes instead of replacing them with defaults (issue #535).

``--dtn-C`` / ``dtn_c`` must be >= 3 (``Jastrow_DTN_data.sanity_check``; the
TREXIO route and ``dtn_c = 2`` on the QE route already reject smaller values),
and the polynomial order of an active DTN channel must be >= 1 because its cusp
constraint fixes the linear coefficient. Replacing an explicit 0 by the
default (``x or default``) would let the conversion return normally with a
different Jastrow than requested.
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

import re
import shutil
from pathlib import Path

import h5py
import pytest

from jqmc.jqmc_qe import run_qe2jqmc
from tests._test_data_cache import qe_example_files_root

pytestmark = [pytest.mark.with_chdir]

_LI_K025 = qe_example_files_root() / "Li_k025" / "out" / "li_k025.save"

_DTN_ATTRS = "wavefunction_data/jastrow_data/jastrow_dtn_data"

# Expected rejection of an explicit 0 per option (the builder's own contract).
_ZERO_REJECTION = {
    "dtn_c": "cutoff_smoothness = 0 must be >= 3",
    "dtn_n_u": "n_order_u = 0 must be >= 1",
    "dtn_n_chi": "n_order_chi = 0 must be >= 1",
    "dtn_n_eN": "n_order_eN = 0 must be >= 1",
    "dtn_n_ee": "n_order_ee = 0 must be >= 1",
}


def _convert(tmp_path, jastrow_params):
    save = tmp_path / "li_k025.save"
    if not save.exists():
        shutil.copytree(_LI_K025, save)
    out = tmp_path / "hamiltonian_data.h5"
    run_qe2jqmc(
        qe_params={"qe_save_dir": str(save), "kind": "pw"},
        jastrow_params=jastrow_params,
        hamiltonian_h5=str(out),
    )
    return out


@pytest.mark.light
@pytest.mark.parametrize(
    "preset,option",
    [
        ("u", "dtn_c"),
        ("u", "dtn_n_u"),
        ("u_chi", "dtn_n_chi"),
        ("u_f", "dtn_n_eN"),
        ("u_f", "dtn_n_ee"),
    ],
)
def test_explicit_zero_dtn_size_is_rejected_not_defaulted(tmp_path, preset, option):
    """An explicit 0 reaches the DTN builder's validation and raises; no Hamiltonian file is written."""
    with pytest.raises(ValueError, match=re.escape(_ZERO_REJECTION[option])):
        _convert(tmp_path, {"jastrow_dtn": preset, option: 0})
    assert not (tmp_path / "hamiltonian_data.h5").exists()


@pytest.mark.light
def test_dtn_sizes_keep_defaults_when_omitted_and_explicit_values_otherwise(tmp_path):
    """Omitted sizes take the documented defaults (C=4, N_u=N_chi=8, N_eN=N_ee=4); explicit ones are stored as given."""
    defaults = {"dtn_c": 4, "dtn_n_u": 8, "dtn_n_chi": 8, "dtn_n_eN": 4, "dtn_n_ee": 4}
    explicit = {"dtn_c": 3, "dtn_n_u": 5, "dtn_n_chi": 6, "dtn_n_eN": 2, "dtn_n_ee": 3}
    stored_name = {
        "dtn_c": "cutoff_smoothness",
        "dtn_n_u": "n_order_u",
        "dtn_n_chi": "n_order_chi",
        "dtn_n_eN": "n_order_eN",
        "dtn_n_ee": "n_order_ee",
    }
    for requested, expected in (({}, defaults), (explicit, explicit)):
        out = _convert(tmp_path, {"jastrow_dtn": "u_chi_f", **requested})
        with h5py.File(out, "r") as f:
            attrs = f[_DTN_ATTRS].attrs
            stored = {option: int(attrs[name]) for option, name in stored_name.items()}
        assert stored == expected
        out.unlink()
