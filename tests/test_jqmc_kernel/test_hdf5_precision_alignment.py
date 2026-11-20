"""Exact dtype and stored-value contracts for Hamiltonian checkpoint loading."""

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

import dataclasses
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from jqmc._precision import get_dtype_np
from jqmc._setting import get_tol_hdf5_roundtrip
from jqmc.hamiltonians import Hamiltonian_data
from jqmc.wavefunction import evaluate_ln_wavefunction
from tests.canonical_wavefunction_fixtures import CANONICAL_PATTERN_IDS, build_canonical_hamiltonian, canonical_signature


pytestmark = [pytest.mark.light, pytest.mark.without_chdir]


def _checkpoint_dtype_zone(path):
    """Map the declared checkpoint fields, independently of alignment helpers."""
    if path.endswith(("/radial/exponents", "/radial/coefficients")):
        # AO radial field family; its declared dtype is the fp64 model storage
        # in both modes (see _declared_component_dtype), not the ao_eval zone.
        return "ao_eval"
    if path == "wavefunction_data/geminal_data/lambda_matrix":
        return "det_eval"
    if path.startswith("wavefunction_data/jastrow_data/jastrow_dtn_data/") and path.rsplit("/", 1)[-1] in {
        "alpha_indep",
        "beta_indep",
        "gamma_indep",
    }:
        # DTN independent coefficients: Jastrow parameter storage, float64 in both
        # modes (issue #781), not the jastrow_eval zone.
        return "jastrow_parameter_storage"
    return None


def _declared_component_dtype(zone):
    """Component dtype a declared checkpoint field must have after load.

    AO radial exponents/coefficients are fp64 model storage in both precision
    modes (issue #124; IMPL-HAM-HDF5-BASIS-FP64-STORAGE-001). DTN independent coefficients are
    Jastrow parameter storage, float64 in both modes (issue #781). AGP lambda
    follows its evaluation zone.
    """
    if zone in ("ao_eval", "jastrow_parameter_storage"):
        return np.dtype(np.float64)
    return np.dtype(get_dtype_np(zone))


def _loaded_checkpoint_field(hamiltonian, path):
    value = hamiltonian
    for component in path.split("/"):
        value = value[int(component)] if component.isdigit() else getattr(value, component)
    return np.asarray(value)


@pytest.mark.parametrize("pattern_id", CANONICAL_PATTERN_IDS)
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Checkpoint field/dtype roundtrip; no electron-coordinate evaluation."
)
def test_hdf5_load_aligns_declared_fields_without_losing_values(tmp_path, pattern_id):
    """Checkpoint AO, AGP and DTN fields adopt their declared dtypes without losing complex components.

    The sixteen canonical synthetic Hamiltonians exercise real/complex GTO,
    pGTO, MO/AO representations, composites, J3 and DTN. A temporary checkpoint
    deliberately stores alignment fields with the opposite component width.
    Loaded arrays must have the declared dtype (fp64 model storage for AO
    radial arrays and DTN independent coefficients in both modes; det_eval for
    AGP lambda)
    and exactly the input values independently rounded to the checkpoint
    width and cast to the target.
    The input's expanded radial view accounts for pGTO template compression.
    MO, AGPn and spline arrays without an alignment rule
    retain their stored dtype and values. No TREXIO input, electron positions,
    numerical tolerance or statistical sampling is used.
    """
    hamiltonian = build_canonical_hamiltonian(pattern_id)
    checkpoint = Path(tmp_path) / "opposite-width.h5"
    hamiltonian.save_to_hdf5(str(checkpoint))
    expected = {}
    signature = canonical_signature(pattern_id)
    orbitals = list(signature["determinant"]["orbitals"].values()) + [signature["jastrow"]["j3_orbital_signature"]]
    required_zones = set()
    if any("gto" in orbital for orbital in orbitals):
        required_zones.add("ao_eval")
    if signature["determinant"]["kind"] == "AGP":
        required_zones.add("det_eval")
    if signature["jastrow"]["dtn_channel_mask"] != "off":
        required_zones.add("jastrow_parameter_storage")
    exercised_zones = set()
    with h5py.File(checkpoint, "r+") as handle:
        fields = []

        def collect(path, item):
            if isinstance(item, h5py.Dataset) and item.dtype.kind in {"f", "c"}:
                if _checkpoint_dtype_zone(path) or path.endswith(("/mo_coefficients", "/lambda_diag", "/spline_coefficients")):
                    fields.append(path)

        handle.visititems(collect)
        for path in fields:
            stored = np.asarray(handle[path])
            zone = _checkpoint_dtype_zone(path)
            if zone is None:
                expected[path] = stored
                continue
            exercised_zones.add(zone)
            component = _declared_component_dtype(zone)
            if np.iscomplexobj(stored):
                target = np.dtype(np.complex128 if component == np.dtype(np.float64) else np.complex64)
                source = np.complex64 if target == np.dtype(np.complex128) else np.complex128
            else:
                target = component
                source = np.float32 if target == np.dtype(np.float64) else np.float64
            stored = stored.astype(source)
            del handle[path]
            handle.create_dataset(path, data=stored)
            original = _loaded_checkpoint_field(hamiltonian, path)
            expected[path] = original.astype(source).astype(target)
            assert stored.dtype != target
    assert expected, "Every canonical fixture must exercise aligned or preserved array fields"
    assert exercised_zones == required_zones
    loaded = Hamiltonian_data.load_from_hdf5(str(checkpoint))
    for path, reference in expected.items():
        actual = _loaded_checkpoint_field(loaded, path)
        np.testing.assert_equal(actual.dtype.str, reference.dtype.str, err_msg=path)
        np.testing.assert_array_equal(actual, reference, err_msg=path)


def _assert_same_stored_values(loaded, original, path="hamiltonian"):
    """Compare every dataclass field (pytree and static) by stored value, recursively."""
    if dataclasses.is_dataclass(original) and not isinstance(original, type):
        assert type(loaded) is type(original), path
        for field in dataclasses.fields(original):
            _assert_same_stored_values(getattr(loaded, field.name), getattr(original, field.name), f"{path}.{field.name}")
    elif isinstance(original, (list, tuple)):
        assert len(loaded) == len(original), path
        for index, (item_loaded, item_original) in enumerate(zip(loaded, original, strict=True)):
            _assert_same_stored_values(item_loaded, item_original, f"{path}[{index}]")
    elif isinstance(original, dict):
        assert loaded.keys() == original.keys(), path
        for key in original:
            _assert_same_stored_values(loaded[key], original[key], f"{path}[{key!r}]")
    elif original is None or isinstance(original, str):
        assert loaded == original, path
    else:
        actual, reference = np.asarray(loaded), np.asarray(original)
        np.testing.assert_equal(actual.dtype.str, reference.dtype.str, err_msg=path)
        np.testing.assert_array_equal(actual, reference, err_msg=path)


@pytest.mark.parametrize("pattern_id", CANONICAL_PATTERN_IDS)
@pytest.mark.gdd_electron_configuration(
    mode="not-required", reason="Checkpoint field-value roundtrip; no electron-coordinate evaluation."
)
def test_hdf5_roundtrip_restores_every_stored_field_value(tmp_path, pattern_id):
    """Every field of a canonical Hamiltonian, static flags included, survives save -> load by value.

    The reference is the in-memory Hamiltonian that was written. Every
    dataclass field is compared recursively, including the non-pytree
    (static) fields such as the geminal ``time_reversal`` flag, the electron
    counts and the basis/orbital metadata, so a field that the writer omits
    and the loader refills with its dataclass default is detected whenever
    the fixture value differs from that default (for example the
    spin-unrestricted AGP/AGPn patterns with ``time_reversal=False``).
    Values and component dtypes must match exactly: the checkpoint stores
    the in-memory values without conversion in both precision modes.
    """
    hamiltonian = build_canonical_hamiltonian(pattern_id)
    checkpoint = Path(tmp_path) / "roundtrip.h5"
    hamiltonian.save_to_hdf5(str(checkpoint))
    loaded = Hamiltonian_data.load_from_hdf5(str(checkpoint))
    _assert_same_stored_values(loaded, hamiltonian)


_UNRESTRICTED_GEMINAL_PATTERN_IDS = ("WF-PATTERN-003", "WF-PATTERN-006", "WF-PATTERN-007", "WF-PATTERN-014")
_GEMINAL_ONLY_FLAGS = dict(
    opt_J1_param=False,
    opt_J2_param=False,
    opt_J3_param=False,
    opt_geminal=True,
    opt_JDTN_u_param=False,
    opt_JDTN_chi_param=False,
    opt_JDTN_f_param=False,
)


def _ln_psi(wavefunction, configurations):
    return np.asarray(
        [float(np.real(evaluate_ln_wavefunction(wavefunction, jnp.asarray(up), jnp.asarray(dn)))) for up, dn in configurations]
    )


@pytest.mark.parametrize("pattern_id", _UNRESTRICTED_GEMINAL_PATTERN_IDS)
@pytest.mark.gdd_electron_configuration(mode="required", position_source="seeded-random")
def test_hdf5_roundtrip_unrestricted_geminal_keeps_time_reversal_through_optimizer_step(tmp_path, pattern_id):
    """A spin-unrestricted AGP/AGPn reloads with ``time_reversal=False`` and takes the same optimizer step.

    The canonical spin-unrestricted AGP (WF-PATTERN-007) and AGPn patterns
    (003, 006, 014) are written with ``time_reversal=False``. After
    ``save_to_hdf5`` -> ``load_from_hdf5`` the flag must equal the stored
    one. The flag selects the variational manifold of the next optimizer
    step (Hermitian projection of the AGP lambda paired block, or
    ``C_dn = conj(C_up)`` for AGPn, when ``True``), so the same geminal step
    (blocks and seeded update vector of the in-memory wavefunction, applied
    through ``apply_block_updates``) is applied to the in-memory and to the
    reloaded wavefunction. The reference is the in-memory wavefunction:
    ln|Psi| at seeded electron configurations must agree before and after
    the step within the HDF5 round-trip tolerance family.
    """
    hamiltonian = build_canonical_hamiltonian(pattern_id)
    geminal = hamiltonian.wavefunction_data.geminal_data
    assert geminal.time_reversal is False
    checkpoint = Path(tmp_path) / "unrestricted.h5"
    hamiltonian.save_to_hdf5(str(checkpoint))
    loaded = Hamiltonian_data.load_from_hdf5(str(checkpoint))
    assert loaded.wavefunction_data.geminal_data.time_reversal == geminal.time_reversal

    rng = np.random.default_rng(592)
    n_up, n_dn = int(geminal.num_electron_up), int(geminal.num_electron_dn)
    configurations = [(1.2 * rng.normal(size=(n_up, 3)), 1.2 * rng.normal(size=(n_dn, 3))) for _ in range(3)]
    blocks = hamiltonian.wavefunction_data.get_variational_blocks(state_token=("hdf5-roundtrip", 0), **_GEMINAL_ONLY_FLAGS)
    thetas = 0.05 * rng.normal(size=sum(block.size for block in blocks))
    stepped_reference = hamiltonian.wavefunction_data.apply_block_updates(blocks, thetas, learning_rate=1.0)
    stepped_loaded = loaded.wavefunction_data.apply_block_updates(blocks, thetas, learning_rate=1.0)

    atol, rtol = get_tol_hdf5_roundtrip("strict")
    np.testing.assert_allclose(
        _ln_psi(loaded.wavefunction_data, configurations),
        _ln_psi(hamiltonian.wavefunction_data, configurations),
        atol=atol,
        rtol=rtol,
    )
    np.testing.assert_allclose(
        _ln_psi(stepped_loaded, configurations),
        _ln_psi(stepped_reference, configurations),
        atol=atol,
        rtol=rtol,
    )
