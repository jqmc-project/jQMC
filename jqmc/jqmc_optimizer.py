"""Shared wave-function optimization driver.

This module owns the optimization control flow independently of the MCMC
sampler.  MCMC.run_optimize is a facade that supplies the
sampling/estimator interface consumed here.
"""

# Copyright (C) 2024- Kosuke Nakano
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the copyright holder nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import hashlib
import numbers
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from dataclasses import replace as dataclass_replace
from logging import getLogger
from typing import Any, Callable, Literal, Protocol, Sequence, TypeAlias

import jax
import numpy as np
import numpy.typing as npt
from jax import numpy as jnp
from mpi4py import MPI

from . import jqmc_optimizer_kernel as _kernel
from ._driver_helpers import (
    _check_mpi4py_jax_distribution_consistency,
    _init_external_control_file,
    _log_memory_usage,
    _read_external_stop_flag,
    _remove_external_control_file,
)
from ._setting import (
    OPTIMIZER_DEFAULT_ASR_FALLBACK_GAMMA,
    OPTIMIZER_DEFAULT_ASR_GAMMA_MAX,
    OPTIMIZER_DEFAULT_CG_MAX_ITER,
    OPTIMIZER_DEFAULT_CG_TOL,
    OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF,
    OPTIMIZER_DEFAULT_LM_REFERENCE_WEIGHT_MIN,
    OPTIMIZER_DERIVATIVE_SAMPLE_BATCH,
    OPTIMIZER_DIAGNOSTICS_TOP_COORDINATES,
    get_optimizer_roundoff_tolerance,
    min_S_diag_abs,
)
from .hamiltonians import Hamiltonian_data
from .wavefunction import VariationalParameterBlock

logger = getLogger("jqmc").getChild(__name__)
num_sep_line = 66

# Limit host tie gathers to a short batch of rows regardless of sample count.
_HOST_TIE_SAMPLE_CHUNK = 8
_NUMPY_STRONG_SCALAR_PROMOTION = np.lib.NumpyVersion(np.__version__) >= "2.0.0"


def _host_tie_scale(coefficients: np.ndarray) -> np.ndarray | np.generic | None:
    """Preserve the scalar multiplication dtype when batching COO entries."""
    if _NUMPY_STRONG_SCALAR_PROMOTION:
        return np.conj(coefficients)
    # NumPy 1.x can select different operation dtypes for different scalar
    # values. Unit real coefficients can still share one scalar.
    if np.isrealobj(coefficients) and np.all(coefficients == 1):
        return coefficients[0]
    return None


DerivativeArray: TypeAlias = np.ndarray | jax.Array
DerivativeArrayConverter: TypeAlias = Callable[[npt.ArrayLike], DerivativeArray]
ArrayReduction: TypeAlias = Callable[[npt.ArrayLike], npt.ArrayLike]


@dataclass(frozen=True)
class ParameterSelection:
    """Wave-function parameter families included in an optimization."""

    opt_J1_param: bool = True
    opt_J2_param: bool = True
    opt_J3_param: bool = True
    opt_geminal: bool = False
    opt_J3_basis_exp: bool = False
    opt_J3_basis_coeff: bool = False
    opt_lambda_basis_exp: bool = False
    opt_lambda_basis_coeff: bool = False
    opt_JDTN_u_param: bool = True
    opt_JDTN_chi_param: bool = True
    opt_JDTN_f_param: bool = True

    def variational_block_kwargs(self) -> dict[str, bool]:
        """Return the names accepted by ``get_variational_blocks``."""
        return {name: value for name, value in vars(self).items()}

    def gradient_flag_kwargs(self) -> dict[str, bool]:
        """Translate public parameter names to derivative-storage flags."""
        return {
            "j1_param": self.opt_J1_param,
            "j2_param": self.opt_J2_param,
            "j3_matrix": self.opt_J3_param,
            "lambda_matrix": self.opt_geminal,
            "j3_basis_exp": self.opt_J3_basis_exp,
            "j3_basis_coeff": self.opt_J3_basis_coeff,
            "lambda_basis_exp": self.opt_lambda_basis_exp,
            "lambda_basis_coeff": self.opt_lambda_basis_coeff,
            "j3_poly_alpha": self.opt_JDTN_u_param,
            "j3_poly_beta": self.opt_JDTN_chi_param,
            "j3_poly_gamma": self.opt_JDTN_f_param,
        }


def _format_value_and_error(value: float, error: float, min_decimals: int = 3) -> str:
    """``"value +- error"`` in fixed point, with at least two significant digits of the error.

    A fixed ``%.3f`` printed every error below 5e-4 as ``0.000`` and gave one
    significant digit below 0.01. Errors of 0.01 and above keep the three decimals.
    """
    decimals = min_decimals
    if np.isfinite(error) and error > 0.0:
        decimals = max(min_decimals, 1 - int(np.floor(np.log10(error))))
    return f"{value:.{decimals}f} +- {error:.{decimals}f}"


def _validated_coordinate_blocks(
    blocks: Sequence[VariationalParameterBlock],
) -> tuple[tuple[VariationalParameterBlock, ...], tuple[str, int]]:
    """Return one structurally valid, same-state parameter publication."""
    from .wavefunction import VariationalParameterBlock

    published = tuple(blocks)
    if not published:
        raise ValueError("coordinate maps require at least one variational block.")
    if any(not isinstance(block, VariationalParameterBlock) for block in published):
        raise ValueError("coordinate maps require fully validated VariationalParameterBlock objects.")
    names = [block.name for block in published]
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    duplicates = sorted(duplicates)
    if duplicates:
        raise ValueError(f"duplicate variational block names: {duplicates}.")
    token = published[0].state_token
    for block in published:
        if block.state_token != token:
            raise ValueError("all variational blocks must carry the same state_token.")
        if block.tangent_spec.state_token != block.state_token:
            raise ValueError(f"block {block.name!r} has a tangent_spec state_token mismatch.")
        values = np.asarray(block.values)
        if tuple(values.shape) != tuple(block.shape) or int(values.size) != int(block.size):
            raise ValueError(f"block {block.name!r} has inconsistent values/shape/size metadata.")
    return published, token


def _is_frozen_dense_buffer(array: np.ndarray) -> bool:
    """Return whether ``array`` is a dense view of read-only NumPy-owned memory.

    No NumPy array sharing this memory can write it while its owner stays
    read-only, and a ``copy=True`` duplicate would have the same strides, so
    publishing the array as is keeps both immutability and the memory layout.
    Only the owner of the base buffer could make it writable again
    (``setflags(write=True)``); the host publication creates these bases
    privately (the MCMC post-warmup stack, an earlier ``build`` copy) and
    never does.
    """
    if not (array.flags.c_contiguous or array.flags.f_contiguous):
        return False
    current: object = array
    while isinstance(current, np.ndarray):
        if current.flags.writeable:
            return False
        if current.base is None:
            return True
        current = current.base
    return False


def _readonly_source_array(values: npt.ArrayLike, *, block_name: str, component: str) -> np.ndarray:
    """Return one immutable numeric derivative array without a lossy cast.

    Mutable, strided or foreign-owned (e.g. JAX) inputs are copied
    defensively; an already immutable dense NumPy publication is shared, so
    re-validating or re-building a published source copies nothing.
    """
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"derivative source {block_name!r}.{component} must be numeric.")
    if _is_frozen_dense_buffer(array):
        return array
    copied = np.array(array, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, eq=False)
class BlockDerivativeSource:
    """Exact complex derivative components for one variational block.

    The derivative policy has one owner, the matching
    :class:`VariationalParameterBlock`.  Consequently this payload stores no
    policy tag of its own: real blocks publish ``direct``; holomorphic complex
    blocks publish ``d_p``; and non-holomorphic/tied blocks publish both
    ``d_p`` and ``d_pbar``.  Arrays have one or more leading sample axes and
    the block's storage shape on their trailing axes.
    """

    name: str
    direct: np.ndarray | None
    d_p: np.ndarray | None
    d_pbar: np.ndarray | None

    @classmethod
    def build(
        cls,
        *,
        block: VariationalParameterBlock,
        direct: npt.ArrayLike | None = None,
        d_p: npt.ArrayLike | None = None,
        d_pbar: npt.ArrayLike | None = None,
    ) -> "BlockDerivativeSource":
        """Validate and freeze one exact source against its owning block."""
        from .wavefunction import VariationalParameterBlock

        if not isinstance(block, VariationalParameterBlock):
            raise ValueError("BlockDerivativeSource requires a validated VariationalParameterBlock.")
        components = {"direct": direct, "d_p": d_p, "d_pbar": d_pbar}
        present = {name for name, values in components.items() if values is not None}
        expected = {
            "direct_real": {"direct"},
            "holomorphic": {"d_p"},
            "wirtinger_pair": {"d_p", "d_pbar"},
        }[block.derivative_policy]
        if present != expected:
            raise ValueError(
                f"derivative_policy={block.derivative_policy!r} for block {block.name!r} "
                f"requires components {sorted(expected)}; got {sorted(present)}."
            )

        frozen: dict[str, np.ndarray | None] = {"direct": None, "d_p": None, "d_pbar": None}
        component_shapes: set[tuple[int, ...]] = set()
        block_shape = tuple(block.shape)
        for name in expected:
            array = _readonly_source_array(components[name], block_name=block.name, component=name)
            if array.ndim <= len(block_shape):
                raise ValueError(
                    f"derivative source {block.name!r}.{name} requires at least one leading sample axis "
                    f"before trailing shape {block_shape}; got {array.shape}."
                )
            trailing_shape = tuple(array.shape[-len(block_shape) :]) if block_shape else ()
            if trailing_shape != block_shape:
                raise ValueError(
                    f"derivative source {block.name!r}.{name} has trailing shape {trailing_shape}; expected {block_shape}."
                )
            component_shapes.add(tuple(array.shape))
            frozen[name] = array
        if len(component_shapes) != 1:
            raise ValueError(f"all derivative components for block {block.name!r} must have the same shape.")
        return cls(name=block.name, **frozen)


@dataclass(frozen=True, eq=False)
class DeviceBlockDerivativeSource:
    """Exact derivative components for the production optimizer path.

    This is the production counterpart of :class:`BlockDerivativeSource`.
    Log-derivative payloads stay on device; the much larger local-energy
    derivative payload may stay on host until reduced-LM contraction.
    """

    name: str
    direct: DerivativeArray | None
    d_p: DerivativeArray | None
    d_pbar: DerivativeArray | None

    @classmethod
    def _build(
        cls,
        *,
        block: VariationalParameterBlock,
        converter: DerivativeArrayConverter,
        location: str,
        direct: npt.ArrayLike | None = None,
        d_p: npt.ArrayLike | None = None,
        d_pbar: npt.ArrayLike | None = None,
    ) -> "DeviceBlockDerivativeSource":
        components = {"direct": direct, "d_p": d_p, "d_pbar": d_pbar}
        present = {name for name, values in components.items() if values is not None}
        expected = {
            "direct_real": {"direct"},
            "holomorphic": {"d_p"},
            "wirtinger_pair": {"d_p", "d_pbar"},
        }[block.derivative_policy]
        if present != expected:
            raise ValueError(
                f"derivative_policy={block.derivative_policy!r} for block {block.name!r} "
                f"requires components {sorted(expected)}; got {sorted(present)}."
            )
        arrays = {name: (None if values is None else converter(values)) for name, values in components.items()}
        shapes = {tuple(arrays[name].shape) for name in expected}
        if len(shapes) != 1:
            raise ValueError(f"all derivative components for block {block.name!r} must have the same shape.")
        shape = next(iter(shapes))
        block_shape = tuple(block.shape)
        if len(shape) <= len(block_shape) or (block_shape and shape[-len(block_shape) :] != block_shape):
            raise ValueError(
                f"{location} derivative source {block.name!r} has shape {shape}; "
                f"expected leading sample axes followed by {block_shape}."
            )
        return cls(name=block.name, **arrays)

    @classmethod
    def build(
        cls,
        *,
        block: VariationalParameterBlock,
        direct: npt.ArrayLike | None = None,
        d_p: npt.ArrayLike | None = None,
        d_pbar: npt.ArrayLike | None = None,
    ) -> "DeviceBlockDerivativeSource":
        return cls._build(
            block=block,
            converter=jnp.asarray,
            location="device",
            direct=direct,
            d_p=d_p,
            d_pbar=d_pbar,
        )

    @classmethod
    def build_host(
        cls,
        *,
        block: VariationalParameterBlock,
        direct: npt.ArrayLike | None = None,
        d_p: npt.ArrayLike | None = None,
        d_pbar: npt.ArrayLike | None = None,
    ) -> "DeviceBlockDerivativeSource":
        return cls._build(
            block=block,
            converter=np.asarray,
            location="host",
            direct=direct,
            d_p=d_p,
            d_pbar=d_pbar,
        )


def validate_device_derivative_sources(
    blocks: Sequence[VariationalParameterBlock],
    sources: Sequence[DeviceBlockDerivativeSource],
) -> tuple[DeviceBlockDerivativeSource, ...]:
    """Validate a device publication without materializing it on the host."""
    published, _token = _validated_coordinate_blocks(blocks)
    values = tuple(sources)
    if len(published) != len(values):
        raise ValueError("variational blocks and derivative sources must have the same length.")
    sample_shape = None
    for block, source in zip(published, values, strict=True):
        if not isinstance(source, DeviceBlockDerivativeSource) or source.name != block.name:
            raise ValueError(f"device derivative source order/name mismatch for block {block.name!r}.")
        rebuilt = DeviceBlockDerivativeSource.build(
            block=block,
            direct=source.direct,
            d_p=source.d_p,
            d_pbar=source.d_pbar,
        )
        component = rebuilt.direct if rebuilt.direct is not None else rebuilt.d_p
        assert component is not None
        leading = tuple(component.shape[: component.ndim - len(block.shape)]) if block.shape else tuple(component.shape)
        if sample_shape is None:
            sample_shape = leading
        elif leading != sample_shape:
            raise ValueError("all device derivative sources must have identical leading sample axes.")
    return values


def validate_derivative_sources(
    blocks: Sequence[VariationalParameterBlock],
    sources: Sequence[BlockDerivativeSource],
) -> tuple[BlockDerivativeSource, ...]:
    """Validate exact order, names, policies, storage shapes, and sample axes."""
    published_blocks, _token = _validated_coordinate_blocks(blocks)
    published_sources = tuple(sources)
    if len(published_blocks) != len(published_sources):
        raise ValueError("variational blocks and derivative sources must have the same length.")

    validated: list[BlockDerivativeSource] = []
    common_sample_shape: tuple[int, ...] | None = None
    for block, source in zip(published_blocks, published_sources, strict=True):
        if not isinstance(source, BlockDerivativeSource):
            raise ValueError("derivative publications require BlockDerivativeSource objects.")
        if source.name != block.name:
            raise ValueError(f"derivative source order/name mismatch: block {block.name!r} received source {source.name!r}.")
        for component_name in ("direct", "d_p", "d_pbar"):
            component_values = getattr(source, component_name)
            if isinstance(component_values, np.ndarray) and component_values.flags.writeable:
                raise ValueError(f"derivative source {source.name!r}.{component_name} must be an immutable published buffer.")
        rebuilt = BlockDerivativeSource.build(
            block=block,
            direct=source.direct,
            d_p=source.d_p,
            d_pbar=source.d_pbar,
        )
        component = rebuilt.direct if rebuilt.direct is not None else rebuilt.d_p
        assert component is not None
        sample_shape = tuple(component.shape[: component.ndim - len(block.shape)]) if block.shape else tuple(component.shape)
        if common_sample_shape is None:
            common_sample_shape = sample_shape
        elif sample_shape != common_sample_shape:
            raise ValueError(
                f"all derivative sources must have the same leading sample shape; "
                f"expected {common_sample_shape}, got {sample_shape} for {block.name!r}."
            )
        validated.append(source)
    return tuple(validated)


@dataclass(frozen=True)
class RealCoordinateMap:
    """Real scalar coordinates for real, complex, or mixed storage blocks.

    Storage is flattened block-by-block in C order.  Every real scalar owns
    one coordinate; every complex scalar owns adjacent ``(real, imag)``
    coordinates.  ``real_adjoint`` is the Euclidean adjoint under the real
    part of the complex inner product and is intentionally distinct from a
    plain transpose.
    """

    block_names: tuple[str, ...]
    block_fields: tuple[str, ...]
    block_shapes: tuple[tuple[int, ...], ...]
    block_storage_slices: tuple[slice, ...]
    block_coordinate_slices: tuple[slice, ...]
    storage_to_coordinate: tuple[tuple[int, ...], ...]
    coordinate_groups: tuple[tuple[int, ...], ...]
    storage_size: int
    coordinate_size: int
    storage_dtype: np.dtype

    @classmethod
    def from_blocks(cls, blocks: Sequence[VariationalParameterBlock]) -> "RealCoordinateMap":
        published, _token = _validated_coordinate_blocks(blocks)

        block_names: list[str] = []
        block_fields: list[str] = []
        block_shapes: list[tuple[int, ...]] = []
        block_storage_slices: list[slice] = []
        block_coordinate_slices: list[slice] = []
        storage_to_coordinate: list[tuple[int, ...]] = []
        storage_offset = 0
        coordinate_offset = 0
        storage_dtypes: list[np.dtype] = []

        for block in published:
            storage_start = storage_offset
            coordinate_start = coordinate_offset
            block_names.append(block.name)
            block_fields.append(block.field)
            block_shapes.append(tuple(block.shape))
            storage_dtypes.append(np.asarray(block.values).dtype)
            for _ in range(int(block.size)):
                if block.field == "real":
                    group = (coordinate_offset,)
                    coordinate_offset += 1
                elif block.field == "complex":
                    group = (coordinate_offset, coordinate_offset + 1)
                    coordinate_offset += 2
                else:  # VariationalParameterBlock validation makes this unreachable.
                    raise ValueError(f"Unknown parameter field: {block.field!r}.")
                storage_to_coordinate.append(group)
                storage_offset += 1
            block_storage_slices.append(slice(storage_start, storage_offset))
            block_coordinate_slices.append(slice(coordinate_start, coordinate_offset))

        groups = tuple(storage_to_coordinate)
        return cls(
            block_names=tuple(block_names),
            block_fields=tuple(block_fields),
            block_shapes=tuple(block_shapes),
            block_storage_slices=tuple(block_storage_slices),
            block_coordinate_slices=tuple(block_coordinate_slices),
            storage_to_coordinate=groups,
            coordinate_groups=groups,
            storage_size=storage_offset,
            coordinate_size=coordinate_offset,
            storage_dtype=np.dtype(np.result_type(*storage_dtypes)),
        )

    def encode(self, q: npt.ArrayLike) -> np.ndarray:
        """Expand real coordinates into flattened ambient storage."""
        coordinates = np.asarray(q)
        if coordinates.ndim < 1 or coordinates.shape[-1] != self.coordinate_size:
            raise ValueError(
                f"encode expected a final coordinate axis of size {self.coordinate_size}; got {coordinates.shape}."
            )
        if np.iscomplexobj(coordinates):
            raise ValueError("encode requires real-valued coordinates.")
        if not np.issubdtype(coordinates.dtype, np.number):
            raise ValueError("encode requires numeric real-valued coordinates.")

        result = np.empty(
            (*coordinates.shape[:-1], self.storage_size),
            dtype=np.result_type(coordinates.dtype, self.storage_dtype),
        )
        for storage_index, group in enumerate(self.storage_to_coordinate):
            if len(group) == 1:
                result[..., storage_index] = coordinates[..., group[0]]
            else:
                result[..., storage_index] = coordinates[..., group[0]] + 1.0j * coordinates[..., group[1]]
        return result

    def real_adjoint(self, storage_covectors: npt.ArrayLike) -> np.ndarray:
        r"""Pull storage covectors back under ``Re(<u, E q>_C)``."""
        covectors = np.asarray(storage_covectors)
        if covectors.ndim < 1 or covectors.shape[-1] != self.storage_size:
            raise ValueError(f"real_adjoint expected a final storage axis of size {self.storage_size}; got {covectors.shape}.")
        if not np.issubdtype(covectors.dtype, np.number):
            raise ValueError("real_adjoint requires numeric storage covectors.")

        result = np.empty(
            (*covectors.shape[:-1], self.coordinate_size),
            dtype=np.real(covectors).dtype,
        )
        for storage_index, group in enumerate(self.storage_to_coordinate):
            result[..., group[0]] = np.real(covectors[..., storage_index])
            if len(group) == 2:
                result[..., group[1]] = np.imag(covectors[..., storage_index])
        return result


@dataclass(frozen=True)
class OptimizationCoordinateMap:
    """Complete real-coordinate map for scalar, tied, and factor blocks."""

    blocks: tuple[VariationalParameterBlock, ...]
    state_token: tuple[str, int]
    field_map: RealCoordinateMap
    block_storage_slices: tuple[slice, ...]
    block_coordinate_slices: tuple[slice, ...]
    coordinate_groups: tuple[tuple[int, ...], ...]
    storage_size: int
    coordinate_size: int

    @classmethod
    def from_blocks(cls, blocks: Sequence[VariationalParameterBlock]) -> "OptimizationCoordinateMap":
        published, token = _validated_coordinate_blocks(blocks)
        field_map = RealCoordinateMap.from_blocks(published)
        block_coordinate_slices: list[slice] = []
        coordinate_groups: list[tuple[int, ...]] = []
        coordinate_offset = 0
        for block, field_coordinate_slice, storage_slice in zip(
            published,
            field_map.block_coordinate_slices,
            field_map.block_storage_slices,
            strict=True,
        ):
            spec = block.tangent_spec
            if spec.kind == "linear_tie":
                assert spec.linear_tie is not None
                block_coordinate_size = spec.linear_tie.coordinate_size
                coordinate_groups.extend((coordinate_offset + q,) for q in range(block_coordinate_size))
            else:
                block_coordinate_size = field_coordinate_slice.stop - field_coordinate_slice.start
                for storage_index in range(storage_slice.start, storage_slice.stop):
                    ambient_group = field_map.storage_to_coordinate[storage_index]
                    coordinate_groups.append(
                        tuple(
                            coordinate_offset + ambient_index - field_coordinate_slice.start for ambient_index in ambient_group
                        )
                    )
            block_coordinate_slices.append(slice(coordinate_offset, coordinate_offset + block_coordinate_size))
            coordinate_offset += block_coordinate_size

        return cls(
            blocks=published,
            state_token=token,
            field_map=field_map,
            block_storage_slices=field_map.block_storage_slices,
            block_coordinate_slices=tuple(block_coordinate_slices),
            coordinate_groups=tuple(coordinate_groups),
            storage_size=field_map.storage_size,
            coordinate_size=coordinate_offset,
        )

    @staticmethod
    def _field_encode(block: VariationalParameterBlock, coordinates: npt.ArrayLike) -> np.ndarray:
        q = np.asarray(coordinates)
        if block.field == "real":
            return q
        result = np.empty((*q.shape[:-1], block.size), dtype=np.result_type(q.dtype, np.asarray(block.values).dtype))
        result[...] = q[..., 0::2] + 1.0j * q[..., 1::2]
        return result

    @staticmethod
    def _field_real_adjoint(block: VariationalParameterBlock, covectors: npt.ArrayLike) -> np.ndarray:
        values = np.asarray(covectors)
        if block.field == "real":
            return np.real(values)
        result = np.empty((*values.shape[:-1], 2 * block.size), dtype=np.real(values).dtype)
        result[..., 0::2] = np.real(values)
        result[..., 1::2] = np.imag(values)
        return result

    @staticmethod
    def _project_whitened_stiefel_tangent(whitened: np.ndarray, ambient: np.ndarray) -> np.ndarray:
        """Orthogonally project onto ``dB B^H + B dB^H = 0``.

        The small row-Gram Lyapunov solve remains correct when the published
        coefficients are only numerically, rather than bitwise, orthonormal.
        """
        row_gram = whitened @ whitened.conj().T
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (row_gram + row_gram.conj().T))
        if np.any(eigenvalues <= 0.0):
            raise FloatingPointError("S-Stiefel row Gram must be positive definite.")
        rhs = ambient @ whitened.conj().T + whitened @ np.swapaxes(ambient.conj(), -1, -2)
        rhs = 0.5 * (rhs + np.swapaxes(rhs.conj(), -1, -2))
        transformed = np.einsum("ij,...jk,kl->...il", eigenvectors.conj().T, rhs, eigenvectors, optimize=True)
        denominator = eigenvalues[:, None] + eigenvalues[None, :]
        multiplier = np.einsum(
            "ij,...jk,kl->...il",
            eigenvectors,
            transformed / denominator,
            eigenvectors.conj().T,
            optimize=True,
        )
        multiplier = 0.5 * (multiplier + np.swapaxes(multiplier.conj(), -1, -2))
        return ambient - multiplier @ whitened

    @staticmethod
    def _stiefel_forward(block: VariationalParameterBlock, ambient: npt.ArrayLike) -> np.ndarray:
        spec = block.tangent_spec
        assert spec.metric_sqrt is not None
        assert spec.metric_inv_sqrt is not None
        sqrt = spec.metric_sqrt
        inv_sqrt = spec.metric_inv_sqrt
        coefficients = np.asarray(block.values)
        delta = np.asarray(ambient).reshape(*np.asarray(ambient).shape[:-1], *block.shape)
        whitened = coefficients @ sqrt.T
        z = delta @ sqrt.T
        tangent = OptimizationCoordinateMap._project_whitened_stiefel_tangent(whitened, z) @ inv_sqrt.T
        return tangent.reshape(*tangent.shape[: -len(block.shape)], block.size)

    @staticmethod
    def _stiefel_real_adjoint(block: VariationalParameterBlock, covectors: npt.ArrayLike) -> np.ndarray:
        spec = block.tangent_spec
        assert spec.metric_sqrt is not None
        assert spec.metric_inv_sqrt is not None
        sqrt = spec.metric_sqrt
        inv_sqrt = spec.metric_inv_sqrt
        coefficients = np.asarray(block.values)
        values = np.asarray(covectors).reshape(*np.asarray(covectors).shape[:-1], *block.shape)
        whitened = coefficients @ sqrt.T
        z = values @ inv_sqrt.T
        ambient = OptimizationCoordinateMap._project_whitened_stiefel_tangent(whitened, z) @ sqrt.T
        return ambient.reshape(*ambient.shape[: -len(block.shape)], block.size)

    @staticmethod
    def _linear_tie_forward(block: VariationalParameterBlock, coordinates: npt.ArrayLike) -> np.ndarray:
        tie = block.tangent_spec.linear_tie
        assert tie is not None
        q = np.asarray(coordinates)
        result = np.zeros(
            (*q.shape[:-1], tie.output_size),
            dtype=np.result_type(q.dtype, tie.coefficients.dtype, np.asarray(block.values).dtype),
        )
        for output_index, coordinate_index, coefficient in zip(
            tie.output_indices,
            tie.coordinate_indices,
            tie.coefficients,
            strict=True,
        ):
            result[..., output_index] += coefficient * q[..., coordinate_index]
        return result

    @staticmethod
    def _linear_tie_real_adjoint(block: VariationalParameterBlock, covectors: npt.ArrayLike) -> np.ndarray:
        tie = block.tangent_spec.linear_tie
        assert tie is not None
        values = np.asarray(covectors)
        result = np.zeros(
            (*values.shape[:-1], tie.coordinate_size),
            dtype=np.result_type(np.real(values).dtype, np.real(tie.coefficients).dtype),
        )
        if values.ndim == 2 and values.shape[0] <= _HOST_TIE_SAMPLE_CHUNK:
            scale = _host_tie_scale(tie.coefficients)
            # A row block bounds the COO gather. add.at applies repeated
            # coordinate contributions in published order, like the loop.
            if scale is not None:
                contributions = np.real(scale * values[:, tie.output_indices])
                np.add.at(result, (slice(None), tie.coordinate_indices), contributions)
                return result
        for output_index, coordinate_index, coefficient in zip(
            tie.output_indices,
            tie.coordinate_indices,
            tie.coefficients,
            strict=True,
        ):
            result[..., coordinate_index] += np.real(np.conj(coefficient) * values[..., output_index])
        return result

    def _encode_block(self, index: int, coordinates: npt.ArrayLike) -> np.ndarray:
        block = self.blocks[index]
        spec = block.tangent_spec
        if spec.kind == "linear_tie":
            return self._linear_tie_forward(block, coordinates)
        ambient = self._field_encode(block, coordinates)
        if spec.kind == "s_stiefel_rows":
            return self._stiefel_forward(block, ambient)
        return ambient

    def _real_adjoint_block(self, index: int, covectors: npt.ArrayLike) -> np.ndarray:
        block = self.blocks[index]
        spec = block.tangent_spec
        if spec.kind == "linear_tie":
            return self._linear_tie_real_adjoint(block, covectors)
        ambient = np.asarray(covectors)
        if spec.kind == "s_stiefel_rows":
            ambient = self._stiefel_real_adjoint(block, ambient)
        return self._field_real_adjoint(block, ambient)

    def encode(self, q: npt.ArrayLike) -> np.ndarray:
        """Return the allowed flattened storage tangent ``delta_p = E q``."""
        coordinates = np.asarray(q)
        if coordinates.ndim < 1 or coordinates.shape[-1] != self.coordinate_size:
            raise ValueError(
                f"encode expected a final coordinate axis of size {self.coordinate_size}; got {coordinates.shape}."
            )
        if np.iscomplexobj(coordinates):
            raise ValueError("encode requires real-valued coordinates.")
        if not np.issubdtype(coordinates.dtype, np.number):
            raise ValueError("encode requires numeric real-valued coordinates.")
        parts = [
            self._encode_block(index, coordinates[..., coordinate_slice])
            for index, coordinate_slice in enumerate(self.block_coordinate_slices)
        ]
        return np.concatenate(parts, axis=-1)

    def real_adjoint(self, storage_covectors: npt.ArrayLike) -> np.ndarray:
        r"""Pull storage covectors back under ``Re(<u, E q>_C)``."""
        covectors = np.asarray(storage_covectors)
        if covectors.ndim < 1 or covectors.shape[-1] != self.storage_size:
            raise ValueError(f"real_adjoint expected a final storage axis of size {self.storage_size}; got {covectors.shape}.")
        if not np.issubdtype(covectors.dtype, np.number):
            raise ValueError("real_adjoint requires numeric storage covectors.")
        parts = [
            self._real_adjoint_block(index, covectors[..., storage_slice])
            for index, storage_slice in enumerate(self.block_storage_slices)
        ]
        return np.concatenate(parts, axis=-1)

    def _transform_derivative_block(
        self, index: int, primary: np.ndarray, secondary: np.ndarray | None, *, real: bool
    ) -> np.ndarray:
        if real:
            # Real route: Adj(conj p) with conj p = p for a real source; the
            # imaginary covector Adj(i p) is never formed.
            columns = self._real_adjoint_block(index, primary)
            if np.iscomplexobj(columns):
                raise TypeError(f"real optimizer transform produced complex columns for block {self.blocks[index].name!r}.")
            # A real Euclidean adjoint is the source itself: return owned columns.
            return np.array(columns, copy=True) if np.may_share_memory(columns, primary) else columns
        if secondary is None:
            real_covector = np.conj(primary)
            imag_covector = 1.0j * real_covector
        else:
            real_covector = np.conj(primary) + secondary
            imag_covector = 1.0j * (np.conj(primary) - secondary)
        real_columns = self._real_adjoint_block(index, real_covector)
        imag_columns = self._real_adjoint_block(index, imag_covector)
        return real_columns + 1.0j * imag_columns

    def transform_derivatives(self, sources: Sequence[BlockDerivativeSource]) -> np.ndarray:
        """Return exact directional columns in canonical real-q order.

        The result dtype is :func:`_transform_derivatives_dtype`. A real wave
        function (every block real structure, every source real with no
        ``d_pbar``) takes the real route: the columns are real (float64 for
        float64 sources) and their imaginary half, identically zero, is never
        computed. Every other input keeps the complex columns
        ``Adj(conj p [+ s]) + i Adj(i (conj p [- s]))``.
        """
        validated = validate_derivative_sources(self.blocks, sources)
        dtype = _transform_derivatives_dtype(self.blocks, validated)
        real = not np.issubdtype(dtype, np.complexfloating)
        native = _kernel._host_rust_kernel("optimizer_transform_derivatives")
        if native is not None:
            specs, leading_shape = _rust_block_specs(self, validated)
            columns = native(specs, int(np.prod(leading_shape)), self.coordinate_size, real)
            if columns.dtype != (np.float64 if real else np.complex128):
                raise TypeError(
                    f"jqmc_rs optimizer transform returned {columns.dtype} on the {'real' if real else 'complex'} route."
                )
            return columns.reshape(*leading_shape, self.coordinate_size).astype(dtype, copy=False)
        transformed: list[np.ndarray] = []
        chunk = _HOST_TIE_SAMPLE_CHUNK
        for index, (block, source) in enumerate(zip(self.blocks, validated, strict=True)):
            primary = source.direct if source.direct is not None else source.d_p
            secondary = source.d_pbar
            assert primary is not None
            leading_shape = primary.shape if not block.shape else primary.shape[: -len(block.shape)]
            primary = primary.reshape(*leading_shape, block.size)
            if secondary is not None:
                secondary = secondary.reshape(*leading_shape, block.size)
            if block.tangent_spec.linear_tie is None or primary.ndim != 2 or primary.shape[0] <= chunk:
                transformed.append(self._transform_derivative_block(index, primary, secondary, real=real))
                continue
            # Typical J3 sources have many sample rows and a large tied
            # storage axis. Only the completed directional block spans N.
            # The adjoint chooses vectorized or ordered scalar COO per chunk.
            first = self._transform_derivative_block(
                index, primary[:chunk], None if secondary is None else secondary[:chunk], real=real
            )
            columns = np.empty((primary.shape[0], first.shape[1]), dtype=first.dtype)
            columns[:chunk] = first
            for start in range(chunk, primary.shape[0], chunk):
                rows = slice(start, start + chunk)
                columns[rows] = self._transform_derivative_block(
                    index, primary[rows], None if secondary is None else secondary[rows], real=real
                )
            transformed.append(columns)
        result = transformed[0] if len(transformed) == 1 else np.concatenate(transformed, axis=-1)
        if result.dtype != dtype:
            raise TypeError(f"optimizer transform produced {result.dtype}; the transform contract requires {dtype}.")
        return result

    @staticmethod
    def _project_whitened_stiefel_tangent_device(whitened: jax.Array, ambient: jax.Array) -> jax.Array:
        row_gram = whitened @ whitened.conj().T
        eigenvalues, eigenvectors = jnp.linalg.eigh(0.5 * (row_gram + row_gram.conj().T))
        rhs = ambient @ whitened.conj().T + whitened @ jnp.swapaxes(ambient.conj(), -1, -2)
        rhs = 0.5 * (rhs + jnp.swapaxes(rhs.conj(), -1, -2))
        transformed = jnp.einsum("ij,...jk,kl->...il", eigenvectors.conj().T, rhs, eigenvectors)
        multiplier = jnp.einsum(
            "ij,...jk,kl->...il",
            eigenvectors,
            transformed / (eigenvalues[:, None] + eigenvalues[None, :]),
            eigenvectors.conj().T,
        )
        multiplier = 0.5 * (multiplier + jnp.swapaxes(multiplier.conj(), -1, -2))
        return ambient - multiplier @ whitened

    def _real_adjoint_block_device(self, index: int, covectors: jax.Array) -> jax.Array:
        block = self.blocks[index]
        spec = block.tangent_spec
        values = jnp.asarray(covectors)
        if spec.kind == "linear_tie":
            tie = spec.linear_tie
            assert tie is not None
            result = jnp.zeros(
                (*values.shape[:-1], tie.coordinate_size),
                dtype=jnp.result_type(jnp.real(values).dtype, jnp.asarray(tie.coefficients).real.dtype),
            )
            # Repeated coordinates must accumulate all COO contributions.
            return result.at[..., jnp.asarray(tie.coordinate_indices)].add(
                jnp.real(jnp.conj(jnp.asarray(tie.coefficients)) * values[..., jnp.asarray(tie.output_indices)])
            )
        if spec.kind == "s_stiefel_rows":
            assert spec.metric_sqrt is not None
            assert spec.metric_inv_sqrt is not None
            sqrt = jnp.asarray(spec.metric_sqrt)
            inv_sqrt = jnp.asarray(spec.metric_inv_sqrt)
            coefficients = jnp.asarray(block.values)
            reshaped = values.reshape(*values.shape[:-1], *block.shape)
            whitened = coefficients @ sqrt.T
            z = reshaped @ inv_sqrt.T
            values = (self._project_whitened_stiefel_tangent_device(whitened, z) @ sqrt.T).reshape(
                *values.shape[:-1], block.size
            )
        if block.field == "real":
            return jnp.real(values)
        result = jnp.empty((*values.shape[:-1], 2 * block.size), dtype=jnp.real(values).dtype)
        result = result.at[..., 0::2].set(jnp.real(values))
        return result.at[..., 1::2].set(jnp.imag(values))

    def transform_derivatives_device(
        self,
        sources: Sequence[DeviceBlockDerivativeSource],
        *,
        right: jax.Array | None = None,
        sync_scope: str | None = None,
        all_real: bool | None = None,
    ) -> jax.Array:
        """Transform exact sources on device, optionally contracting a right basis.

        ``right`` is split blockwise so reduced LM never materializes the full
        sample-by-coordinate O/D matrices.  ``sync_scope`` restricts an
        uncontracted publication to the ordered world- or sampler-owned
        blocks, avoiding a full-coordinate temporary in twist-averaged optimization.

        ``all_real`` selects the route. ``True`` takes the real route, whose
        columns are real (float64 for float64 sources) and whose identically
        zero imaginary covector branch is never formed; ``False`` keeps the
        complex columns ``Adj(conj p [+ s]) + i Adj(i (conj p [- s]))``.
        Optimizer entries decide it once over their communicator
        (:func:`_collective_transform_route`) and pass it to every transform, so
        every rank of a collective takes the same route. ``None`` (a direct,
        rank-local call) takes this publication's own route,
        :func:`_real_transform_route`, the host transform's dtype-only decision.
        ``True`` for a publication that is not real is an error.
        """
        if sync_scope not in (None, "world", "sampler"):
            raise ValueError("device derivative sync_scope must be 'world', 'sampler', or None.")
        validated = validate_device_derivative_sources(self.blocks, sources)
        real = _real_transform_route(self.blocks, validated) if all_real is None else bool(all_real)
        if real and not _real_transform_route(self.blocks, validated):
            raise ValueError(
                "all_real=True requires a real publication: real-structure blocks and real sources without d_pbar."
            )
        contracted = None
        transformed = []
        basis_host = None if right is None else np.asarray(right)
        basis = None if basis_host is None else jnp.asarray(basis_host)
        if basis is not None and basis.shape[0] != self.coordinate_size:
            raise ValueError("device derivative contraction basis has the wrong coordinate axis.")
        for index, (block, source, coordinate_slice) in enumerate(
            zip(self.blocks, validated, self.block_coordinate_slices, strict=True)
        ):
            if sync_scope is not None and block.sync_scope != sync_scope:
                continue
            if basis_host is not None and not np.any(basis_host[coordinate_slice]):
                continue
            if source.direct is not None:
                primary = source.direct
                secondary = None
            else:
                assert source.d_p is not None
                primary = source.d_p
                secondary = source.d_pbar
            leading_shape = primary.shape if not block.shape else primary.shape[: -len(block.shape)]
            primary_flat = primary.reshape(*leading_shape, block.size)
            if real:
                # Real route: Adj(conj p) = Adj(p) for a real source; the
                # imaginary covector Adj(i p), identically zero, is never formed.
                columns = self._real_adjoint_block_device(index, primary_flat)
            elif secondary is None:
                real_covector = jnp.conj(primary_flat)
                imag_covector = 1.0j * jnp.conj(primary_flat)
            else:
                secondary_flat = secondary.reshape(*leading_shape, block.size)
                real_covector = jnp.conj(primary_flat) + secondary_flat
                imag_covector = 1.0j * (jnp.conj(primary_flat) - secondary_flat)
            if not real:
                columns = self._real_adjoint_block_device(index, real_covector) + 1.0j * self._real_adjoint_block_device(
                    index, imag_covector
                )
            if basis is None:
                transformed.append(columns)
            else:
                contribution = columns @ basis[coordinate_slice]
                contracted = contribution if contracted is None else contracted + contribution
        if basis is not None:
            if contracted is None:
                raise ValueError("device derivative contraction basis has no active block contribution.")
            return contracted
        if not transformed:
            raise ValueError(f"device derivative publication has no blocks for sync_scope={sync_scope!r}.")
        return jnp.concatenate(transformed, axis=-1)

    def transform_host_derivatives_reduced(
        self,
        sources: Sequence[DeviceBlockDerivativeSource],
        right: npt.ArrayLike,
        *,
        sample_batch_size: int = OPTIMIZER_DERIVATIVE_SAMPLE_BATCH,
        all_real: bool | None = None,
    ) -> np.ndarray:
        """Stream host-resident exact sources through device contraction.

        ``all_real`` is the route decision passed to every chunk's
        :meth:`transform_derivatives_device` call (``None``: the publication's
        own route, decided once here for all chunks).
        """
        values = tuple(sources)
        if len(values) != len(self.blocks):
            raise ValueError("variational blocks and host derivative sources must have the same length.")
        basis = np.asarray(right)
        if basis.ndim != 2 or basis.shape[0] != self.coordinate_size:
            raise ValueError("host derivative contraction basis has the wrong coordinate axis.")
        if sample_batch_size <= 0:
            raise ValueError("host derivative sample batch size must be positive.")

        host_sources = []
        sample_shape = None
        for block, source in zip(self.blocks, values, strict=True):
            if not isinstance(source, DeviceBlockDerivativeSource) or source.name != block.name:
                raise ValueError(f"host derivative source order/name mismatch for block {block.name!r}.")
            source = DeviceBlockDerivativeSource.build_host(
                block=block,
                direct=source.direct,
                d_p=source.d_p,
                d_pbar=source.d_pbar,
            )
            primary = source.direct if source.direct is not None else source.d_p
            assert primary is not None
            leading_shape = tuple(primary.shape[: primary.ndim - len(block.shape)]) if block.shape else tuple(primary.shape)
            if sample_shape is None:
                sample_shape = leading_shape
            elif leading_shape != sample_shape:
                raise ValueError("all host derivative sources must have identical leading sample axes.")
            host_sources.append(source)
        if sample_shape is None:
            raise ValueError("host derivative contraction requires at least one variational block.")

        sample_count = int(np.prod(sample_shape))
        if all_real is None:
            all_real = _real_transform_route(self.blocks, host_sources)
        reduced_chunks = []
        for start in range(0, sample_count, sample_batch_size):
            stop = min(start + sample_batch_size, sample_count)
            device_sources = []
            for block, source in zip(self.blocks, host_sources, strict=True):
                components = {}
                for name in ("direct", "d_p", "d_pbar"):
                    component = getattr(source, name)
                    if component is not None:
                        flattened = component.reshape((sample_count, *block.shape))
                        components[name] = flattened[start:stop]
                device_sources.append(DeviceBlockDerivativeSource.build(block=block, **components))
            reduced = self.transform_derivatives_device(device_sources, right=basis, all_real=all_real)
            reduced.block_until_ready()
            reduced_chunks.append(np.asarray(reduced))
        return np.concatenate(reduced_chunks, axis=0).reshape(*sample_shape, basis.shape[1])

    def layout_fingerprint(self) -> tuple[Any, ...]:
        """Return the hashable structural identity of this coordinate layout."""
        block_fingerprints: list[tuple[Any, ...]] = []
        for block in self.blocks:
            spec = block.tangent_spec
            tangent_payload: tuple[Any, ...]
            if spec.kind == "linear_tie":
                tie = spec.linear_tie
                assert tie is not None
                tangent_payload = (
                    tie.output_size,
                    tie.coordinate_size,
                    tuple(int(value) for value in tie.output_indices),
                    tuple(int(value) for value in tie.coordinate_indices),
                    tuple(complex(value) for value in tie.coefficients),
                )
            elif spec.kind == "s_stiefel_rows":
                assert spec.metric_sqrt is not None
                tangent_payload = (tuple(spec.metric_sqrt.shape), np.dtype(spec.metric_sqrt.dtype).str)
            else:
                tangent_payload = ()
            block_fingerprints.append(
                (
                    block.name,
                    tuple(block.shape),
                    np.dtype(np.asarray(block.values).dtype).str,
                    block.field,
                    block.derivative_policy,
                    block.sync_scope,
                    spec.kind,
                    tangent_payload,
                )
            )
        return (
            "real-coordinate-map-v1",
            tuple(block_fingerprints),
            self.coordinate_size,
            self.coordinate_groups,
        )


def _array_dtype(values: Any) -> np.dtype:
    """Dtype of a NumPy or JAX array without a device-to-host transfer."""
    dtype = getattr(values, "dtype", None)
    return np.dtype(dtype) if dtype is not None else np.asarray(values).dtype


def _transform_derivatives_dtype(
    blocks: Sequence[VariationalParameterBlock],
    sources: Sequence[BlockDerivativeSource | DeviceBlockDerivativeSource],
) -> np.dtype:
    """Result dtype of :meth:`OptimizationCoordinateMap.transform_derivatives`.

    The single route decision of the host transform (NumPy and ``jqmc_rs``
    alike) and of the multi-rank route decision (:func:`_collective_transform_batch`).
    The real route applies when every block is real structure (real
    field, real-typed tie coefficients and S-Stiefel coefficients/metric
    factors) and every source is real-typed with no ``d_pbar``; its dtype is
    real. Every other input takes the complex route. The precision follows the
    sources and the tangent data: ``result_type`` of the real (complex) route's
    seed ``float32`` (``complex64``) and every source, tie-coefficient and
    S-Stiefel dtype, with non-floating dtypes counted as float64. Only dtypes
    are read, so device sources are never synchronized.
    """
    dtypes: list[np.dtype] = []
    real = True
    for block, source in zip(blocks, sources, strict=True):
        primary = source.direct if source.direct is not None else source.d_p
        assert primary is not None
        real = real and block.field == "real" and source.d_pbar is None
        dtypes.extend(_array_dtype(component) for component in (primary, source.d_pbar) if component is not None)
        spec = block.tangent_spec
        if spec.kind == "linear_tie":
            assert spec.linear_tie is not None
            dtypes.append(_array_dtype(spec.linear_tie.coefficients))
        elif spec.kind == "s_stiefel_rows":
            assert spec.metric_sqrt is not None
            assert spec.metric_inv_sqrt is not None
            dtypes.extend(_array_dtype(matrix) for matrix in (block.values, spec.metric_sqrt, spec.metric_inv_sqrt))
    inexact = [value if np.issubdtype(value, np.inexact) else np.dtype(np.float64) for value in dtypes]
    real = real and not any(np.issubdtype(value, np.complexfloating) for value in inexact)
    return np.result_type(np.float32 if real else np.complex64, *inexact)


def _real_transform_route(
    blocks: Sequence[VariationalParameterBlock],
    sources: Sequence[BlockDerivativeSource | DeviceBlockDerivativeSource],
) -> bool:
    """Whether the optimizer transform of ``sources`` takes the real route."""
    return not np.issubdtype(_transform_derivatives_dtype(blocks, sources), np.complexfloating)


def _rust_block_specs(
    coordinate_map: OptimizationCoordinateMap,
    sources: Sequence[BlockDerivativeSource],
) -> tuple[list[tuple[Any, ...]], tuple[int, ...]]:
    """Describe validated host sources for the ``jqmc_rs`` optimizer transform.

    Each block becomes ``(kind, field_complex, coordinate_start, coordinate_size,
    size, primary, secondary, tie, stiefel)`` with the sources as zero-copy
    ``(samples, size)`` float64/complex128 views (other dtypes are widened). Also
    returns the leading sample shape.
    """
    specs: list[tuple[Any, ...]] = []
    leading_shape: tuple[int, ...] | None = None
    for block, source, coordinate_slice in zip(
        coordinate_map.blocks, sources, coordinate_map.block_coordinate_slices, strict=True
    ):
        primary = source.direct if source.direct is not None else source.d_p
        secondary = source.d_pbar
        assert primary is not None
        leading = tuple(primary.shape) if not block.shape else tuple(primary.shape[: -len(block.shape)])
        leading_shape = leading if leading_shape is None else leading_shape
        rows = int(np.prod(leading))
        field_complex = block.field == "complex"
        primary_complex = field_complex or np.iscomplexobj(primary)
        primary_rows = np.ascontiguousarray(
            primary.reshape(rows, block.size), dtype=np.complex128 if primary_complex else np.float64
        )
        secondary_rows = (
            None if secondary is None else np.ascontiguousarray(secondary.reshape(rows, block.size), dtype=np.complex128)
        )
        spec = block.tangent_spec
        tie = None
        stiefel = None
        if spec.kind == "linear_tie":
            linear_tie = spec.linear_tie
            assert linear_tie is not None
            tie = (
                np.ascontiguousarray(linear_tie.output_indices, dtype=np.int64),
                np.ascontiguousarray(linear_tie.coordinate_indices, dtype=np.int64),
                np.ascontiguousarray(linear_tie.coefficients, dtype=np.complex128),
            )
        elif spec.kind == "s_stiefel_rows":
            assert spec.metric_sqrt is not None
            assert spec.metric_inv_sqrt is not None
            matrices = (np.asarray(block.values), spec.metric_sqrt, spec.metric_inv_sqrt)
            real_structure = not field_complex and not any(np.iscomplexobj(matrix) for matrix in matrices)
            stiefel = tuple(
                np.ascontiguousarray(matrix, dtype=np.float64 if real_structure else np.complex128) for matrix in matrices
            )
        specs.append(
            (
                spec.kind,
                field_complex,
                coordinate_slice.start,
                coordinate_slice.stop - coordinate_slice.start,
                int(block.size),
                primary_rows,
                secondary_rows,
                tie,
                stiefel,
            )
        )
    assert leading_shape is not None
    return specs, leading_shape


def _variational_parameter_counts(coordinate_map: OptimizationCoordinateMap) -> dict[str, Any]:
    """Per-block and total counts of a coordinate map: stored values and real optimization coordinates."""
    blocks = [
        {
            "name": block.name,
            "shape": tuple(int(n) for n in block.shape),
            "values": int(block.size),
            "coordinates": int(coordinates.stop - coordinates.start),
        }
        for block, coordinates in zip(coordinate_map.blocks, coordinate_map.block_coordinate_slices, strict=True)
    ]
    return {
        "blocks": blocks,
        "values": sum(block["values"] for block in blocks),
        "coordinates": int(coordinate_map.coordinate_size),
    }


def variational_parameter_counts(
    wavefunction_data: Any,
    parameters: ParameterSelection | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Number of variational parameters a VMC optimization of ``wavefunction_data`` would optimize.

    No sampling is needed: the count depends only on the wavefunction and on the parameter selection. It is
    the optimizer's own count (the same blocks and coordinate map as ``run_optimize``), so the value logged as
    ``Number of variational parameters`` at every optimization step is ``values`` here.

    Args:
        wavefunction_data: a ``Wavefunction_data`` (e.g. ``Hamiltonian_data.wavefunction_data``).
        parameters: a ``ParameterSelection``, or a mapping such as a whole ``[vmc]`` input section; only the
            fields of ``ParameterSelection`` are read from it, and the missing ones take its defaults.

    Returns:
        ``{"blocks": [{"name", "shape", "values", "coordinates"}, ...], "values": int, "coordinates": int}``:
        ``values`` counts the stored parameter values; ``coordinates`` the real coordinates the optimizer
        solves for (the dimension of the SR/LM matrices: a complex value gives two, tied values give their
        independent coordinates).
    """
    if parameters is None:
        parameters = ParameterSelection()
    elif not isinstance(parameters, ParameterSelection):
        names = {f.name for f in dataclass_fields(ParameterSelection)}
        parameters = ParameterSelection(**{k: v for k, v in dict(parameters).items() if k in names})
    blocks = wavefunction_data.get_variational_blocks(
        state_token=("variational-parameter-count", 0),
        **parameters.variational_block_kwargs(),
    )
    return _variational_parameter_counts(OptimizationCoordinateMap.from_blocks(blocks))


@dataclass(frozen=True)
class OptimizationConfig:
    """Sampler-independent controls for one optimization invocation."""

    num_sampling_steps: int = 100
    num_opt_steps: int = 1
    wavefunction_dump_frequency: int = 10
    dump_wavefunction: bool = True
    max_time: int = 86400
    num_warmup_steps: int = 0
    num_bin_blocks: int = 100
    parameters: ParameterSelection = field(default_factory=ParameterSelection)
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    use_device_optimizer: bool = True
    fix_geminal_rank: int | None = None
    energy_aggregator: Callable[[Any], Any] | None = None
    sampler_weight: float | None = None
    # 0-based index of this rank's twist (the ``K`` of ``twist_K.h5``) in a
    # twist-decomposed run; it labels the twist-local step diagnostics.
    twist_index: int | None = None
    _manage_external_control: bool = field(default=True, kw_only=True)
    # The caller runs further optimization steps under the same budget after
    # this call (the twist-averaged chunked driver), so the last step of this
    # call is not the last step of the run for the max_time prediction.
    _more_steps_follow: bool = field(default=False, kw_only=True)


OptimizationStopReason: TypeAlias = Literal["max_time", "stop_flag"]
"""Why :func:`run_optimize` left its loop before ``num_opt_steps`` steps."""


def _optimization_budget_exhausted(elapsed: float, steps_done: int, max_time: float, *, next_step_follows: bool) -> bool:
    """Decide the ``max_time`` stop after an optimization step.

    The run stops when ``elapsed`` already exceeds ``max_time``, or when
    another step follows and would end after ``max_time``, its duration being
    predicted by the mean duration of the ``steps_done`` steps so far. A step
    is the unit of the optimization (its sampling is never cut short), so a
    step that cannot finish within the budget is not started: the run instead
    stops now and writes its checkpoint, rather than running past the budget
    into the batch limit.
    """
    if max_time < elapsed:
        return True
    if not next_step_follows or steps_done <= 0:
        return False
    return bool(max_time < elapsed + elapsed / steps_done)


_OPTIMIZER_CONTROL_FILE = "external_control_opt.toml"


@contextmanager
def _optimization_control_file(world_comm: MPI.Comm) -> Iterator[None]:
    """Keep one user control file alive across a chunked optimization."""
    _init_external_control_file(_OPTIMIZER_CONTROL_FILE, world_comm)
    try:
        yield
    finally:
        _remove_external_control_file(_OPTIMIZER_CONTROL_FILE, world_comm)


@dataclass(frozen=True)
class EnergyEstimate:
    """Real energy and variance estimates with real standard errors."""

    mean: float
    standard_error: float
    variance: float
    variance_standard_error: float

    def __post_init__(self) -> None:
        values = (self.mean, self.standard_error, self.variance, self.variance_standard_error)
        if any(np.iscomplexobj(value) for value in values):
            raise ValueError("energy estimates must be real scalars.")
        object.__setattr__(self, "mean", float(self.mean))
        object.__setattr__(self, "standard_error", float(self.standard_error))
        object.__setattr__(self, "variance", float(self.variance))
        object.__setattr__(self, "variance_standard_error", float(self.variance_standard_error))


def _source_leading_shape(block: VariationalParameterBlock, source: BlockDerivativeSource) -> tuple[int, ...]:
    component = source.direct if source.direct is not None else source.d_p
    assert component is not None
    return tuple(component.shape) if not block.shape else tuple(component.shape[: -len(block.shape)])


def _rebuild_source(block: VariationalParameterBlock, source: BlockDerivativeSource) -> BlockDerivativeSource:
    if not isinstance(source, BlockDerivativeSource):
        raise ValueError("derivative publications require BlockDerivativeSource objects.")
    return BlockDerivativeSource.build(
        block=block,
        direct=source.direct,
        d_p=source.d_p,
        d_pbar=source.d_pbar,
    )


@dataclass(frozen=True)
class JackknifeBlockMoments:
    """Complex sufficient statistics for one derivative-source block."""

    name: str
    source_sum: BlockDerivativeSource
    conjugate_source_energy_sum: BlockDerivativeSource

    @classmethod
    def build(
        cls,
        *,
        block: VariationalParameterBlock,
        source_sum: BlockDerivativeSource,
        conjugate_source_energy_sum: BlockDerivativeSource,
    ) -> "JackknifeBlockMoments":
        sources = tuple(_rebuild_source(block, source) for source in (source_sum, conjugate_source_energy_sum))
        if any(source.name != block.name for source in sources):
            raise ValueError(f"jackknife moment order/name mismatch for block {block.name!r}.")
        shapes = {_source_leading_shape(block, source) for source in sources}
        if len(shapes) != 1:
            raise ValueError(f"jackknife moments for block {block.name!r} must have identical leading bin shapes.")
        return cls(
            name=block.name,
            source_sum=sources[0],
            conjugate_source_energy_sum=sources[1],
        )


@dataclass(frozen=True)
class JackknifeBinMetadata:
    """Per-bin scalar and exact-source sufficient statistics."""

    batch_token: str
    bin_sample_counts: np.ndarray
    bin_weight_sums: np.ndarray
    bin_weighted_energy_sums: np.ndarray
    log_source_moments: tuple[JackknifeBlockMoments, ...]

    @classmethod
    def build(
        cls,
        *,
        blocks: Sequence[VariationalParameterBlock],
        batch_token: str,
        bin_sample_counts: npt.ArrayLike,
        bin_weight_sums: npt.ArrayLike,
        bin_weighted_energy_sums: npt.ArrayLike,
        log_source_moments: Sequence[JackknifeBlockMoments],
    ) -> "JackknifeBinMetadata":
        published, _token = _validated_coordinate_blocks(blocks)
        if not isinstance(batch_token, str) or not batch_token:
            raise ValueError("batch_token must be a non-empty string.")
        counts = np.asarray(bin_sample_counts)
        weights = np.asarray(bin_weight_sums)
        energies = np.asarray(bin_weighted_energy_sums)
        if counts.ndim != 1 or counts.size < 1:
            raise ValueError("rank-local jackknife metadata requires at least one nonempty bin.")
        if not np.issubdtype(counts.dtype, np.integer) or np.any(counts <= 0):
            raise ValueError("jackknife bin sample counts must be positive integers.")
        if weights.shape != counts.shape or energies.shape != counts.shape:
            raise ValueError("jackknife scalar moment arrays must match the bin-count shape.")
        if np.iscomplexobj(weights) or not np.issubdtype(weights.dtype, np.number) or not np.all(np.isfinite(weights)):
            raise ValueError("jackknife bin weights must be finite real scalars.")
        if not np.issubdtype(energies.dtype, np.number) or not np.all(np.isfinite(energies)):
            raise ValueError("jackknife weighted-energy sums must be finite numeric scalars.")
        moments = tuple(log_source_moments)
        if len(moments) != len(published):
            raise ValueError("jackknife moments and variational blocks must have the same length.")
        for block, moment in zip(published, moments, strict=True):
            if not isinstance(moment, JackknifeBlockMoments) or moment.name != block.name:
                received = getattr(moment, "name", None)
                raise ValueError(f"jackknife moment order/name mismatch: block {block.name!r} received {received!r}.")
            for source in (moment.source_sum, moment.conjugate_source_energy_sum):
                rebuilt = _rebuild_source(block, source)
                if _source_leading_shape(block, rebuilt) != (int(counts.size),):
                    raise ValueError(
                        f"jackknife moments for block {block.name!r} must have one leading bin axis of size {counts.size}."
                    )
        return cls(
            batch_token=batch_token,
            bin_sample_counts=_readonly_source_array(counts, block_name="jackknife", component="bin_sample_counts"),
            bin_weight_sums=_readonly_source_array(weights, block_name="jackknife", component="bin_weight_sums"),
            bin_weighted_energy_sums=_readonly_source_array(
                energies,
                block_name="jackknife",
                component="bin_weighted_energy_sums",
            ),
            log_source_moments=moments,
        )


@dataclass(frozen=True)
class OptimizationBatch:
    """Immutable exact-source sample batch consumed by the optimizer."""

    batch_token: str
    state_token: tuple[str, int]
    weights: np.ndarray
    local_energies: np.ndarray
    log_derivative_sources: tuple[BlockDerivativeSource, ...]
    local_energy_derivative_sources: tuple[BlockDerivativeSource, ...] | None
    jackknife_bin_metadata: JackknifeBinMetadata

    @classmethod
    def build(
        cls,
        *,
        blocks: Sequence[VariationalParameterBlock],
        batch_token: str,
        state_token: tuple[str, int],
        weights: npt.ArrayLike,
        local_energies: npt.ArrayLike,
        log_derivative_sources: Sequence[BlockDerivativeSource],
        local_energy_derivative_sources: Sequence[BlockDerivativeSource] | None,
        jackknife_bin_metadata: JackknifeBinMetadata,
    ) -> "OptimizationBatch":
        published, published_token = _validated_coordinate_blocks(blocks)
        if state_token != published_token:
            raise ValueError("OptimizationBatch state_token does not match its variational blocks.")
        if not isinstance(batch_token, str) or not batch_token:
            raise ValueError("batch_token must be a non-empty string.")
        if not isinstance(jackknife_bin_metadata, JackknifeBinMetadata):
            raise ValueError("OptimizationBatch requires JackknifeBinMetadata.")
        if jackknife_bin_metadata.batch_token != batch_token:
            raise ValueError("OptimizationBatch and jackknife metadata batch_token values must match.")

        weight_array = np.asarray(weights)
        energy_array = np.asarray(local_energies)
        if weight_array.ndim != 1 or weight_array.size < 1:
            raise ValueError("OptimizationBatch requires at least one flattened local sample.")
        if (
            np.iscomplexobj(weight_array)
            or not np.issubdtype(weight_array.dtype, np.number)
            or not np.all(np.isfinite(weight_array))
        ):
            raise ValueError("OptimizationBatch weights must be finite real scalars.")
        if np.any(weight_array < 0) or not np.any(weight_array > 0):
            raise ValueError("OptimizationBatch weights must be non-negative with a positive total.")
        if energy_array.shape != weight_array.shape:
            raise ValueError("OptimizationBatch local energies must match the flattened weight shape.")
        if not np.issubdtype(energy_array.dtype, np.number) or not np.all(np.isfinite(energy_array)):
            raise ValueError("OptimizationBatch local energies must be finite numeric scalars.")

        log_sources = validate_derivative_sources(published, log_derivative_sources)
        for block, source in zip(published, log_sources, strict=True):
            if _source_leading_shape(block, source) != (int(weight_array.size),):
                raise ValueError(
                    f"log derivative source {block.name!r} must have one flattened sample axis of size {weight_array.size}."
                )
        if local_energy_derivative_sources is None:
            energy_sources = None
        else:
            energy_sources = validate_derivative_sources(published, local_energy_derivative_sources)
            for block, source in zip(published, energy_sources, strict=True):
                if _source_leading_shape(block, source) != (int(weight_array.size),):
                    raise ValueError(
                        f"local-energy derivative source {block.name!r} must have one flattened sample axis "
                        f"of size {weight_array.size}."
                    )
        if int(np.sum(jackknife_bin_metadata.bin_sample_counts)) != int(weight_array.size):
            raise ValueError("jackknife bin sample counts must sum to the OptimizationBatch sample count.")

        return cls(
            batch_token=batch_token,
            state_token=state_token,
            weights=_readonly_source_array(weight_array, block_name="batch", component="weights"),
            local_energies=_readonly_source_array(energy_array, block_name="batch", component="local_energies"),
            log_derivative_sources=tuple(_rebuild_source(block, source) for block, source in zip(published, log_sources)),
            local_energy_derivative_sources=(
                None
                if energy_sources is None
                else tuple(_rebuild_source(block, source) for block, source in zip(published, energy_sources))
            ),
            jackknife_bin_metadata=jackknife_bin_metadata,
        )


@dataclass(frozen=True)
class DeviceJackknifeBlockMoments:
    """Device-resident per-bin source statistics for one block."""

    name: str
    source_sum: DeviceBlockDerivativeSource
    conjugate_source_energy_sum: DeviceBlockDerivativeSource


@dataclass(frozen=True)
class DeviceJackknifeBinMetadata:
    """Production jackknife payload; only reduced coordinate vectors leave devices."""

    batch_token: str
    bin_sample_counts: Any
    bin_weight_sums: Any
    bin_weighted_energy_sums: Any
    log_source_moments: tuple[DeviceJackknifeBlockMoments, ...]


@dataclass(frozen=True)
class DeviceOptimizationBatch:
    """Production batch with host-staged O/D and device jackknife moments."""

    batch_token: str
    state_token: tuple[str, int]
    weights: Any
    local_energies: Any
    log_derivative_sources: tuple[DeviceBlockDerivativeSource, ...]
    local_energy_derivative_sources: tuple[DeviceBlockDerivativeSource, ...] | None
    jackknife_bin_metadata: DeviceJackknifeBinMetadata


_JACKKNIFE_SOURCE_FIELDS = ("source_sum", "conjugate_source_energy_sum")


def _complex_typed(values: Any) -> Any:
    """Return ``values`` cast to its complex dtype, on the same host/device location."""
    promoted = values.astype(np.result_type(_array_dtype(values), np.complex64))
    if isinstance(promoted, np.ndarray):
        promoted.setflags(write=False)
    return promoted


def _complex_typed_sources(sources: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(
        dataclass_replace(
            source,
            **{name: _complex_typed(getattr(source, name)) for name in ("direct", "d_p") if getattr(source, name) is not None},
        )
        for source in sources
    )


def collective_real_route(local_all_real: bool, comm: MPI.Comm | None) -> bool:
    """Return the one real/complex optimizer route decision shared by every rank of ``comm``.

    This is the single source of truth for every real/complex route decision
    of a multi-rank optimizer route, host and device alike: the route
    is real only if it is real on every rank, ``MIN-Allreduce(local_all_real)``,
    and otherwise every array on every rank takes the complex route. A rank's
    ``local_all_real`` must hold only when every array it will transform or
    reduce on the route is real under the shared transform predicate
    (:func:`_transform_derivatives_dtype`) and its local energies are real.
    Collective over ``comm`` (every rank must call it once per decision); with
    ``comm=None`` or a one-rank ``comm`` it returns ``local_all_real``.
    Decide once at an entry and pass the result down; never recompute it at a
    collective site.
    """
    local = bool(local_all_real)
    if comm is None or _comm_size(comm) == 1:
        return local
    return bool(comm.allreduce(int(local), op=MPI.MIN))


def _transform_route_groups(batch: OptimizationBatch | DeviceOptimizationBatch) -> tuple[Any, ...]:
    """The batch's derivative-source groups, each transformed on one route."""
    moments = batch.jackknife_bin_metadata.log_source_moments
    return (
        batch.log_derivative_sources,
        batch.local_energy_derivative_sources,
        *(tuple(getattr(moment, name) for moment in moments) for name in _JACKKNIFE_SOURCE_FIELDS),
    )


def _local_real_route_flags(
    batch: OptimizationBatch | DeviceOptimizationBatch,
    coordinate_map: "OptimizationCoordinateMap",
) -> np.ndarray:
    """Rank-local realness (1) of each source group of the batch, then of its local energies."""
    return np.asarray(
        [sources is None or _real_transform_route(coordinate_map.blocks, sources) for sources in _transform_route_groups(batch)]
        + [not np.issubdtype(_array_dtype(batch.local_energies), np.complexfloating)],
        dtype=np.int32,
    )


def _complex_route_batch(
    batch: OptimizationBatch | DeviceOptimizationBatch,
    coordinate_map: "OptimizationCoordinateMap",
) -> OptimizationBatch | DeviceOptimizationBatch:
    """Cast every real-route source group and real local energies to the complex dtype."""

    def complex_route(sources: tuple[Any, ...] | None) -> tuple[Any, ...] | None:
        if sources is None or not _real_transform_route(coordinate_map.blocks, sources):
            return sources
        return _complex_typed_sources(sources)

    moments = batch.jackknife_bin_metadata.log_source_moments
    moment_groups = {
        name: complex_route(tuple(getattr(moment, name) for moment in moments)) for name in _JACKKNIFE_SOURCE_FIELDS
    }
    energies = batch.local_energies
    return dataclass_replace(
        batch,
        local_energies=energies if np.issubdtype(_array_dtype(energies), np.complexfloating) else _complex_typed(energies),
        log_derivative_sources=complex_route(batch.log_derivative_sources),
        local_energy_derivative_sources=complex_route(batch.local_energy_derivative_sources),
        jackknife_bin_metadata=dataclass_replace(
            batch.jackknife_bin_metadata,
            log_source_moments=tuple(
                dataclass_replace(moment, **{name: moment_groups[name][index] for name in _JACKKNIFE_SOURCE_FIELDS})
                for index, moment in enumerate(moments)
            ),
        ),
    )


def _collective_transform_batch(
    batch: OptimizationBatch | DeviceOptimizationBatch,
    coordinate_map: "OptimizationCoordinateMap",
    comm: MPI.Comm | None,
) -> OptimizationBatch | DeviceOptimizationBatch:
    """Decide the real/complex route of a multi-rank optimizer entry once, over ``comm``.

    The transform route (:func:`_real_transform_route`) and the local-energy
    dtype are rank-local properties, but their arrays meet in MPI and JAX
    collectives (buffer Allreduce/Allgatherv counts, the pickled LM pencil
    reductions, sharded device kernels). With Gamma and non-Gamma twists on
    different ranks a rank-local decision gives rank-dependent dtypes and
    mismatched collective buffers. One decision is made instead,
    ``real = collective_real_route(local_all_real, comm)``, where ``local_all_real`` holds
    only when every source group (log derivatives, local-energy derivatives,
    the jackknife source moments) and the local energies of this rank are
    real. When ``real`` is false every rank solves everything as complex: a
    real-route group and real local energies are cast to the complex dtype
    (the complex route of real sources has a zero imaginary part), so O, D,
    e, the pencils, the jackknife and the tall layout are complex on every
    rank. The returned batch carries the decision down; a one-rank call, and
    a collectively real batch, is returned unchanged.

    Call it once at a multi-rank entry and pass the returned batch to the
    implementations below it (#889). It handles host and device batches
    alike; an entry whose device transform can take a real route must call it
    before any device collective.
    """
    return _collective_transform_route(batch, coordinate_map, comm)[0]


def _collective_transform_route(
    batch: OptimizationBatch | DeviceOptimizationBatch,
    coordinate_map: "OptimizationCoordinateMap",
    comm: MPI.Comm | None,
) -> tuple[OptimizationBatch | DeviceOptimizationBatch, bool]:
    """:func:`_collective_transform_batch` together with its route decision ``all_real``.

    ``all_real = collective_real_route(local_all_real, comm)`` is the one
    decision of the entry; the returned batch is the input batch when it is
    true or ``comm`` has one rank, and its complex-route cast otherwise. Device
    entries pass ``all_real`` down to every device transform
    (:meth:`OptimizationCoordinateMap.transform_derivatives_device`), so the
    device real route is taken on every rank or on none.
    """
    all_real = collective_real_route(bool(np.all(_local_real_route_flags(batch, coordinate_map))), comm)
    if all_real or comm is None or _comm_size(comm) == 1:
        return batch, all_real
    return _complex_route_batch(batch, coordinate_map), False


@dataclass(frozen=True)
class SRCoordinateSamples:
    """Centered complex sample arrays in canonical real-coordinate order."""

    X_q_local: np.ndarray
    F_local: np.ndarray


def _describe_nonfinite_sr_inputs(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    O_q: np.ndarray,
    weighted_O: np.ndarray,
    weighted_energy: npt.ArrayLike,
) -> str:
    """Name the parameter blocks whose SR centering statistics are non-finite.

    Called only on the failure path, so its cost never reaches a healthy step.
    For each block it counts the non-finite entries of the raw derivative
    source (the sampler output), of the transformed real-coordinate columns,
    and of the (possibly MPI-reduced) weighted sums, and names the first
    offending sample. Non-finite values are reported, never repaired.
    """
    parts: list[str] = []
    for block, source, coordinate_slice in zip(
        coordinate_map.blocks,
        batch.log_derivative_sources,
        coordinate_map.block_coordinate_slices,
        strict=True,
    ):
        raw_bad = 0
        raw_total = 0
        for component in (source.direct, source.d_p, source.d_pbar):
            if component is not None:
                raw_bad += int(np.count_nonzero(~np.isfinite(component)))
                raw_total += int(np.size(component))
        columns = O_q[:, coordinate_slice]
        bad_mask = ~np.isfinite(columns)
        column_bad = int(np.count_nonzero(bad_mask))
        sum_bad = int(np.count_nonzero(~np.isfinite(weighted_O[coordinate_slice])))
        if raw_bad == 0 and column_bad == 0 and sum_bad == 0:
            continue
        bad_samples = np.flatnonzero(np.any(bad_mask, axis=1))
        first = f", first non-finite sample={int(bad_samples[0])}" if bad_samples.size else ""
        parts.append(
            f"block {block.name!r}: non-finite raw source entries={raw_bad}/{raw_total}, "
            f"transformed entries={column_bad}/{columns.size} in {bad_samples.size} sample(s){first}, "
            f"non-finite weighted sums={sum_bad}/{columns.shape[1]}"
        )
    if not np.isfinite(weighted_energy):
        parts.append(
            "weighted local-energy sum is non-finite "
            f"(local non-finite e_L={int(np.count_nonzero(~np.isfinite(batch.local_energies)))})"
        )
    return "Offending inputs: " + "; ".join(parts) + "."


def _owned_array_op(ufunc: np.ufunc, array: np.ndarray, operand: npt.ArrayLike) -> np.ndarray:
    """Reuse private workspace when the original operation keeps its dtype."""
    if np.result_type(array, operand) == array.dtype:
        return ufunc(array, operand, out=array)
    return ufunc(array, operand)


def build_sr_coordinate_samples(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    total_weight: npt.ArrayLike | None = None,
    total_weighted_log_derivatives: npt.ArrayLike | None = None,
    total_weighted_energy: npt.ArrayLike | None = None,
) -> SRCoordinateSamples:
    """Transform exact sources, then center and weight them for SR.

    Distributed callers provide the three already reduced global sufficient
    statistics.  Omitting all three selects the single-rank/local identity
    reduction.  Partial global statistics are rejected.
    """
    if not isinstance(batch, OptimizationBatch) or not isinstance(coordinate_map, OptimizationCoordinateMap):
        raise ValueError("SR coordinate samples require an OptimizationBatch and OptimizationCoordinateMap.")
    if batch.state_token != coordinate_map.state_token:
        raise ValueError("OptimizationBatch state_token does not match the current coordinate map.")
    provided = (
        total_weight is not None,
        total_weighted_log_derivatives is not None,
        total_weighted_energy is not None,
    )
    if any(provided) and not all(provided):
        raise ValueError("distributed SR coordinate samples require all global centering statistics.")

    weights = np.asarray(batch.weights)
    energies = np.asarray(batch.local_energies)
    O_q = coordinate_map.transform_derivatives(batch.log_derivative_sources)
    if O_q.shape != (weights.size, coordinate_map.coordinate_size):
        raise ValueError("transformed log derivatives do not match the batch sample and coordinate axes.")
    if all(provided):
        weight_sum = np.asarray(total_weight)
        weighted_O = np.asarray(total_weighted_log_derivatives)
        weighted_energy = np.asarray(total_weighted_energy)
    else:
        weight_sum = np.asarray(np.sum(weights))
        weighted_O = np.asarray(np.einsum("n,nq->q", weights, O_q))
        weighted_energy = np.asarray(np.dot(weights, energies))
    if weight_sum.ndim != 0 or np.iscomplexobj(weight_sum) or not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("SR total weight must be a positive finite real scalar.")
    if weighted_O.shape != (coordinate_map.coordinate_size,) or weighted_energy.ndim != 0:
        raise ValueError("SR global centering statistics have inconsistent shapes.")
    if not np.all(np.isfinite(weighted_O)) or not np.isfinite(weighted_energy):
        raise ValueError(
            "SR global centering statistics must be finite. "
            + _describe_nonfinite_sr_inputs(batch, coordinate_map, O_q, weighted_O, weighted_energy)
        )

    O_bar = weighted_O / weight_sum
    energy_bar = np.real(weighted_energy / weight_sum)
    sqrt_weight = np.sqrt(weights)
    # The transformed columns are this call's private array unless the
    # transform returned a (read-only) view of a published source; center
    # them in place then, with the same operands and bits as ``O_q - O_bar``.
    if O_q.flags.writeable and np.result_type(O_q, O_bar) == O_q.dtype:
        X = np.subtract(O_q, O_bar, out=O_q)
    else:
        X = np.subtract(O_q, O_bar)
    del O_q
    if np.iscomplexobj(X):
        np.conjugate(X, out=X)
    X = _owned_array_op(np.multiply, X, sqrt_weight[:, None])
    X = _owned_array_op(np.divide, X, np.sqrt(weight_sum))
    X.setflags(write=False)
    X_q = X.T
    F = -2.0 * sqrt_weight * (energies - energy_bar) / np.sqrt(weight_sum)
    F.setflags(write=False)
    return SRCoordinateSamples(X_q_local=X_q, F_local=F)


@dataclass(frozen=True)
class WideSRRealSystem:
    """Wide-SR preparation in surviving real coordinates."""

    X_preconditioned_local: np.ndarray
    b_preconditioned: np.ndarray
    metric_diagonal: np.ndarray
    inverse_sqrt_diagonal: np.ndarray
    active_indices: np.ndarray
    frozen_coordinate_groups: tuple[tuple[int, ...], ...]


def select_sr_real_regime(
    coordinate_size: int,
    sample_counts: Sequence[int],
    *,
    complex_samples: bool,
) -> Literal["wide", "tall"]:
    """Choose the wide or tall SR solve: the one predicate of every SR route.

    The wide solve has one unknown per surviving real coordinate,
    ``coordinate_size``. The tall solve has one unknown per real dual
    coordinate: one per sample for real samples (a real wave function; the
    identically zero imaginary block is omitted) and two per sample, ``[Re;
    Im]``, for complex samples. The smaller system is solved; equality is
    assigned to the tall route.

    ``sample_counts`` are the per-rank sample counts of every rank that the
    solve reduces over (the global count is their sum), and
    ``complex_samples`` is the dtype of the centered samples ``X``/``F``,
    which the entry's collective real/complex route decision
    (:func:`collective_real_route`) makes the same on every rank. The
    result is therefore the same on every rank.
    """
    if not isinstance(coordinate_size, (int, np.integer)) or int(coordinate_size) < 0:
        raise ValueError("SR coordinate_size must be a non-negative integer.")
    counts = tuple(int(count) for count in sample_counts)
    if not counts or any(count < 0 for count in counts) or sum(counts) <= 0:
        raise ValueError("SR sample_counts must be non-negative with a positive total.")
    dual_size = (2 if complex_samples else 1) * sum(counts)
    return "wide" if int(coordinate_size) < dual_size else "tall"


def _complex_sr_samples(X: Any, F: Any) -> bool:
    """Whether centered SR samples are complex (a NumPy or JAX ``X`` and ``F``)."""
    return any(np.issubdtype(_array_dtype(values), np.complexfloating) for values in (X, F))


@dataclass(frozen=True)
class TallSRRealLayout:
    """Tall-route rank-major real dual layout."""

    Z_valid_local: np.ndarray
    h_valid_local: np.ndarray
    sample_counts: tuple[int, ...]
    valid_counts: np.ndarray
    valid_displacements: np.ndarray


def build_tall_sr_real_layout(
    X_preconditioned_local: npt.ArrayLike,
    F_local: npt.ArrayLike,
    *,
    sample_counts: Sequence[int],
    rank: int,
) -> TallSRRealLayout:
    """Realify one rank as ``(rank, real/imag, valid sample)``.

    Complex samples contribute a real and an imaginary dual block. Real
    samples (a real wave function: real ``X`` and ``F``) have an identically
    zero imaginary block, which carries an exactly zero dual and leaves the
    solution unchanged, so it is omitted: ``Z_valid_local`` is a read-only
    view of ``X`` itself and the dual has one entry per sample.

    The host tall solve gathers ``Z_valid_local``/``h_valid_local`` of every
    rank with the rank-major ``valid_counts``/``valid_displacements``.
    """
    X = np.asarray(X_preconditioned_local)
    F = np.asarray(F_local)
    if X.ndim != 2 or F.ndim != 1 or X.shape[1] != F.size:
        raise ValueError("tall real SR requires X=(coordinate, sample) and matching F.")
    # X is the preconditioned matrix of prepare_wide_sr_real_system, whose
    # globally reduced metric diagonal already proved every row finite.
    if not np.all(np.isfinite(F)):
        raise ValueError("tall real SR samples must be finite.")
    counts = tuple(int(count) for count in sample_counts)
    if not counts or any(count < 0 for count in counts) or sum(counts) <= 0:
        raise ValueError("tall real SR sample_counts must have a positive total.")
    if not isinstance(rank, (int, np.integer)) or not 0 <= int(rank) < len(counts):
        raise ValueError("tall real SR rank is outside sample_counts.")
    if counts[int(rank)] != F.size:
        raise ValueError("tall real SR local sample count does not match sample_counts[rank].")

    real_dtype = np.result_type(X.real.dtype, F.real.dtype)
    complex_samples = np.iscomplexobj(X) or np.iscomplexobj(F)
    components = 2 if complex_samples else 1
    if complex_samples:
        Z_valid = np.concatenate(
            (np.asarray(X.real, dtype=real_dtype), np.asarray(X.imag, dtype=real_dtype)),
            axis=1,
        )
        h_valid = np.concatenate((np.asarray(F.real, dtype=real_dtype), np.asarray(-F.imag, dtype=real_dtype)))
    else:
        Z_valid = np.asarray(X, dtype=real_dtype).view()
        h_valid = np.asarray(F, dtype=real_dtype).view()
    valid_counts = components * np.asarray(counts, dtype=np.int64)
    valid_displacements = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        valid_displacements[1:] = np.cumsum(valid_counts[:-1])

    for array in (Z_valid, h_valid, valid_counts, valid_displacements):
        array.setflags(write=False)

    return TallSRRealLayout(
        Z_valid_local=Z_valid,
        h_valid_local=h_valid,
        sample_counts=counts,
        valid_counts=valid_counts,
        valid_displacements=valid_displacements,
    )


@dataclass(frozen=True)
class SolverWarmStart:
    """Persistent real-primal CG state guarded by a structural fingerprint."""

    representation: str
    values: np.ndarray
    layout_fingerprint: str

    @classmethod
    def build(cls, *, values: npt.ArrayLike, layout_fingerprint: str) -> "SolverWarmStart":
        array = np.asarray(values)
        if np.iscomplexobj(array) or array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError("wide SR warm-start values must be a finite real vector.")
        if not isinstance(layout_fingerprint, str) or not layout_fingerprint:
            raise ValueError("wide SR warm start requires a nonempty layout fingerprint.")
        frozen = np.array(array, copy=True)
        frozen.setflags(write=False)
        return cls(
            representation="wide_real_primal",
            values=frozen,
            layout_fingerprint=layout_fingerprint,
        )


@dataclass(frozen=True)
class WideSRTargetResult:
    """Wide real-coordinate SR result used by the production path.

    ``residual_norm`` is the absolute CG residual ``||r||``; ``rhs_norm`` is
    the global ``||b||`` of the solved system, so the relative CG rule
    is ``residual_norm <= cg_tol * rhs_norm``.
    """

    q: np.ndarray
    prepared: WideSRRealSystem
    warm_start_out: SolverWarmStart | None
    residual_norm: float
    num_iterations: int
    rhs_norm: float
    # True when the CG stopped on a breakdown (not on cg_max_iter or the tolerance).
    breakdown: bool


@dataclass(frozen=True)
class TallSRTargetResult:
    """Tall production result; a sample-indexed dual is never persisted."""

    q: np.ndarray
    prepared: WideSRRealSystem
    dual_layout: TallSRRealLayout
    warm_start_out: None
    residual_norm: float
    num_iterations: int
    rhs_norm: float
    breakdown: bool


@dataclass(frozen=True)
class LMRealPencil:
    """Centered LM pencil in real optimization coordinates."""

    H: np.ndarray
    S: np.ndarray
    H_0: float
    conditioning_metric: np.ndarray | None = None


@dataclass(frozen=True)
class LMRealResult:
    """Decoded result of one stabilized real generalized eigensolve."""

    q: np.ndarray
    predicted_energy: float
    reference_weight: float
    active_indices: np.ndarray


@dataclass(frozen=True)
class ASRBoundedResult:
    """Minimum of the physical aSR rational model on a closed interval."""

    gamma: float
    predicted_energy: float
    used_fallback: bool


@dataclass(frozen=True)
class RealCoordinateOptimizationTarget:
    """Final coordinate update and diagnostics before learning-rate scaling."""

    q: np.ndarray
    sr_q: np.ndarray
    coordinate_force: np.ndarray
    coordinate_force_std: np.ndarray
    signal_to_noise: np.ndarray
    sr_regime: str
    accepted_scale: float
    predicted_energy: float | None
    lm_reference_weight: float | None
    warm_start_out: SolverWarmStart | None
    converged: bool
    num_iterations: int
    residual_norm: float
    # Absolute CG stopping threshold ``cg_tol * ||b||`` of the reported
    # solve (issue #859); ``converged`` is ``residual_norm <= residual_tolerance``.
    residual_tolerance: float
    metric_diagonal: np.ndarray
    frozen_coordinate_groups: tuple[tuple[int, ...], ...]
    selected_coordinate_groups: tuple[tuple[int, ...], ...]
    rejection_reason: str | None = None
    linearized_wavefunction_norm: float | None = None
    # True when a reported CG stopped on a breakdown rather than on
    # ``cg_max_iter``; ``converged`` is False in both cases.
    cg_breakdown: bool = False


@dataclass(frozen=True)
class OptimizationDiagnostics:
    """Immutable diagnostics attached to one accepted coordinate result."""

    solver_route: str
    converged: bool
    num_iterations: int
    residual_norm: float
    metric_diagonal: np.ndarray
    frozen_coordinate_groups: tuple[tuple[int, ...], ...]
    selected_coordinate_groups: tuple[tuple[int, ...], ...]
    coordinate_force: np.ndarray
    coordinate_force_standard_error: np.ndarray
    energy_standard_error: float
    predicted_energy: float | None
    lm_reference_weight: float | None
    accepted_scale: float
    rejection_reason: str | None
    linearized_wavefunction_norm: float | None = None


@dataclass(frozen=True)
class BlockStorageUpdate:
    """One named, already-scaled storage update for an immutable state."""

    name: str
    state_token: tuple[str, int]
    delta: np.ndarray


@dataclass(frozen=True)
class OptimizationStepResult:
    """Final real displacement plus ordered typed storage updates."""

    q: np.ndarray
    updates: tuple[BlockStorageUpdate, ...]
    diagnostics: OptimizationDiagnostics
    warm_start_out: SolverWarmStart | None


def finalize_optimization_step(
    target: RealCoordinateOptimizationTarget,
    coordinate_map: OptimizationCoordinateMap,
    *,
    learning_rate: float,
    energy_standard_error: float = 0.0,
) -> OptimizationStepResult:
    """Scale once, decode once, and retain every block's storage dtype."""
    if not isinstance(target, RealCoordinateOptimizationTarget):
        raise TypeError("target must be a RealCoordinateOptimizationTarget.")
    if not isinstance(coordinate_map, OptimizationCoordinateMap):
        raise TypeError("coordinate_map must be an OptimizationCoordinateMap.")
    rate = float(learning_rate)
    if not np.isfinite(rate):
        raise ValueError("learning_rate must be finite.")
    q = np.asarray(target.q) * rate
    if np.iscomplexobj(q) or q.shape != (coordinate_map.coordinate_size,) or not np.all(np.isfinite(q)):
        raise FloatingPointError("final optimizer displacement must be a finite real coordinate vector.")

    updates: list[BlockStorageUpdate] = []
    for index, (block, coordinate_slice) in enumerate(
        zip(coordinate_map.blocks, coordinate_map.block_coordinate_slices, strict=True)
    ):
        decoded = coordinate_map._encode_block(index, q[coordinate_slice]).reshape(block.shape)
        block_dtype = np.asarray(block.values).dtype
        if block.field == "real" and np.iscomplexobj(decoded):
            if np.any(np.imag(decoded) != 0):
                raise ValueError(f"real block {block.name!r} decoded a complex update.")
            decoded = np.real(decoded)
        typed = np.asarray(decoded, dtype=block_dtype)
        if not np.all(np.isfinite(typed)):
            raise FloatingPointError(f"block {block.name!r} decoded a non-finite update.")
        updates.append(
            BlockStorageUpdate(
                name=block.name,
                state_token=coordinate_map.state_token,
                delta=_freeze_array(typed),
            )
        )

    diagnostics = OptimizationDiagnostics(
        solver_route=target.sr_regime,
        converged=target.converged,
        num_iterations=target.num_iterations,
        residual_norm=target.residual_norm,
        metric_diagonal=_freeze_array(target.metric_diagonal),
        frozen_coordinate_groups=target.frozen_coordinate_groups,
        selected_coordinate_groups=target.selected_coordinate_groups,
        coordinate_force=_freeze_array(target.coordinate_force),
        coordinate_force_standard_error=_freeze_array(target.coordinate_force_std),
        energy_standard_error=float(energy_standard_error),
        predicted_energy=target.predicted_energy,
        lm_reference_weight=target.lm_reference_weight,
        accepted_scale=target.accepted_scale,
        rejection_reason=target.rejection_reason,
        linearized_wavefunction_norm=(
            None if target.linearized_wavefunction_norm is None else abs(rate) * target.linearized_wavefunction_norm
        ),
    )
    return OptimizationStepResult(
        q=_freeze_array(q),
        updates=tuple(updates),
        diagnostics=diagnostics,
        warm_start_out=target.warm_start_out,
    )


def _log_optimization_batch_layout(
    batch: OptimizationBatch | DeviceOptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    sampler_size: int,
) -> None:
    """Emit batch-layout diagnostics without leaking source references.

    This helper is called unconditionally so logger configuration cannot
    change the executed control flow.  Keeping the source traversal in its
    own function also guarantees that the final (potentially very large)
    device derivative source goes out of scope on return.
    """
    local_sample_count = int(np.prod(batch.weights.shape))
    logger.debug(
        "Optimization batch: route=%s, local samples=%d, sampler ranks=%d, real coordinates=%d, storage parameters=%d.",
        "device" if isinstance(batch, DeviceOptimizationBatch) else "host",
        local_sample_count,
        sampler_size,
        coordinate_map.coordinate_size,
        coordinate_map.storage_size,
    )
    energy_sources = (
        batch.local_energy_derivative_sources
        if batch.local_energy_derivative_sources is not None
        else (None,) * len(coordinate_map.blocks)
    )
    for block, log_source, energy_source in zip(
        coordinate_map.blocks,
        batch.log_derivative_sources,
        energy_sources,
        strict=True,
    ):
        log_component = log_source.direct if log_source.direct is not None else log_source.d_p
        energy_component = (
            None if energy_source is None else energy_source.direct if energy_source.direct is not None else energy_source.d_p
        )
        logger.debug(
            "  Derivative source %s: O shape=%s, D shape=%s, policy=%s, tangent=%s.",
            block.name,
            tuple(log_component.shape),
            None if energy_component is None else tuple(energy_component.shape),
            block.derivative_policy,
            block.tangent_spec.kind,
        )


def _diagnostics_top_count(size: int) -> int:
    """Number of rows of the ``Top N coordinates by generalized-force signal-to-noise`` block."""
    return min(OPTIMIZER_DIAGNOSTICS_TOP_COORDINATES, size)


@dataclass(frozen=True)
class _TopCoordinateRow:
    """One row of the top-coordinates diagnostics block."""

    label: str
    force: float
    force_std: float
    signal_to_noise: float
    metric_diag: float


def _coordinate_label(
    coordinate_map: OptimizationCoordinateMap,
    coordinate_index: int,
    twist_index: int | None = None,
) -> str:
    """``block[qJ]`` for coordinate ``J`` of a block; ``block[tK:qJ]`` for a twist-local one of twist ``K``.

    The label has no spaces, so the log parser reads it as one token.
    """
    twist_prefix = "" if twist_index is None else f"t{int(twist_index)}:"
    for block, coordinate_slice in zip(
        coordinate_map.blocks,
        coordinate_map.block_coordinate_slices,
        strict=True,
    ):
        if coordinate_slice.start <= coordinate_index < coordinate_slice.stop:
            return f"{block.name}[{twist_prefix}q{coordinate_index - coordinate_slice.start}]"
    return f"q[{twist_prefix}{coordinate_index}]"


def _top_coordinate_rows(
    target: RealCoordinateOptimizationTarget,
    coordinate_map: OptimizationCoordinateMap,
    indices: np.ndarray,
    twist_index: int | None = None,
) -> tuple[_TopCoordinateRow, ...]:
    """The coordinates of ``indices`` with the largest generalized-force signal-to-noise, largest first."""
    indices = np.asarray(indices, dtype=np.int64)
    signal_to_noise = np.asarray(target.signal_to_noise)[indices]
    top_count = _diagnostics_top_count(indices.size)
    order = np.argsort(signal_to_noise, kind="stable")[::-1][:top_count]
    force = np.asarray(target.coordinate_force)
    force_std = np.asarray(target.coordinate_force_std)
    metric_diagonal = np.asarray(target.metric_diagonal)
    rows = []
    for position in order:
        coordinate_index = int(indices[position])
        rows.append(
            _TopCoordinateRow(
                label=_coordinate_label(coordinate_map, coordinate_index, twist_index),
                force=float(force[coordinate_index]),
                force_std=float(force_std[coordinate_index]),
                signal_to_noise=float(signal_to_noise[position]),
                metric_diag=float(metric_diagonal[coordinate_index]),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class _BlockDiagnosticRow:
    """One ``[optimizer block]`` entry with its ``update`` line (and its S-Stiefel tangent line, if any)."""

    label: str
    force_norm: float
    max_abs_force: float
    max_signal_to_noise: float
    metric_diag_min: float
    metric_diag_max: float
    delta_q_norm: float
    max_abs_delta_q: float
    delta_p_norm: float
    max_abs_delta_p: float
    stiefel_tangent: tuple[float, float] | None


def _block_diagnostic_rows(
    target: RealCoordinateOptimizationTarget,
    step_result: OptimizationStepResult,
    coordinate_map: OptimizationCoordinateMap,
    *,
    scope: str | None = None,
    twist_index: int | None = None,
) -> tuple[_BlockDiagnosticRow, ...]:
    """Per-block force, metric and update diagnostics, in coordinate-map order.

    ``scope`` keeps only the blocks of that ``sync_scope``; ``twist_index``
    labels them ``block[tK]``.
    """
    metric_diagonal = np.asarray(target.metric_diagonal)
    rows = []
    for block, coordinate_slice, update in zip(
        coordinate_map.blocks,
        coordinate_map.block_coordinate_slices,
        step_result.updates,
        strict=True,
    ):
        if scope is not None and block.sync_scope != scope:
            continue
        block_q = np.asarray(step_result.q[coordinate_slice])
        storage_delta = np.asarray(update.delta)
        block_force = np.asarray(target.coordinate_force[coordinate_slice])
        block_signal_to_noise = np.asarray(target.signal_to_noise[coordinate_slice])
        block_metric_diagonal = metric_diagonal[coordinate_slice]
        stiefel_tangent = None
        if block.tangent_spec.kind == "s_stiefel_rows":
            assert block.tangent_spec.metric_sqrt is not None
            whitened_tangent = storage_delta.reshape(block.shape) @ block.tangent_spec.metric_sqrt.T
            stiefel_tangent = (float(np.linalg.norm(whitened_tangent)), float(np.linalg.norm(whitened_tangent, ord=2)))
        rows.append(
            _BlockDiagnosticRow(
                label=block.name if twist_index is None else f"{block.name}[t{int(twist_index)}]",
                force_norm=float(np.linalg.norm(block_force)),
                max_abs_force=float(np.max(np.abs(block_force), initial=0.0)),
                max_signal_to_noise=float(np.max(block_signal_to_noise, initial=0.0)),
                metric_diag_min=float(np.min(block_metric_diagonal, initial=np.inf)),
                metric_diag_max=float(np.max(block_metric_diagonal, initial=-np.inf)),
                delta_q_norm=float(np.linalg.norm(block_q)),
                max_abs_delta_q=float(np.max(np.abs(block_q), initial=0.0)),
                delta_p_norm=float(np.linalg.norm(storage_delta)),
                max_abs_delta_p=float(np.max(np.abs(storage_delta), initial=0.0)),
                stiefel_tangent=stiefel_tangent,
            )
        )
    return tuple(rows)


def _scope_coordinate_counts(
    target: RealCoordinateOptimizationTarget,
    coordinate_map: OptimizationCoordinateMap,
    scope: str,
) -> tuple[int, int, int, int]:
    """``(real, storage, frozen coordinates, frozen groups)`` of one ``sync_scope``, from the target's frozen sets."""
    real = storage = 0
    in_scope = np.zeros(coordinate_map.coordinate_size, dtype=bool)
    for block, coordinate_slice, storage_slice in zip(
        coordinate_map.blocks,
        coordinate_map.block_coordinate_slices,
        coordinate_map.block_storage_slices,
        strict=True,
    ):
        if block.sync_scope == scope:
            real += int(coordinate_slice.stop - coordinate_slice.start)
            storage += int(storage_slice.stop - storage_slice.start)
            in_scope[coordinate_slice] = True
    # A twist-decomposed target freezes groups per scope, so each group lies in one scope.
    groups = [group for group in target.frozen_coordinate_groups if in_scope[int(group[0])]]
    return real, storage, sum(len(group) for group in groups), len(groups)


@dataclass(frozen=True)
class _TwistStepDiagnostics:
    """Step diagnostics of a twist-decomposed optimization over every optimized coordinate.

    The shared (``world``-scope) coordinates are the same on every rank; each
    twist adds its own local (``sampler``-scope) coordinates. Maxima are taken
    over all of them; twists are never averaged.
    """

    max_force: float
    max_force_std: float
    max_signal_to_noise: float
    twist_local_max_signal_to_noise: tuple[tuple[int, float], ...]
    top_rows: tuple[_TopCoordinateRow, ...]
    # Shared blocks once, then every twist-local block once per twist (``block[tK]``).
    block_rows: tuple[_BlockDiagnosticRow, ...]
    # ``(real, storage, active, frozen, frozen groups)``: shared + sum over twists of local.
    coordinates_all_twists: tuple[int, int, int, int, int]


def _gather_twist_step_diagnostics(
    target: RealCoordinateOptimizationTarget,
    step_result: OptimizationStepResult | None,
    coordinate_map: OptimizationCoordinateMap,
    *,
    sampler_comm: MPI.Comm,
    world_comm: MPI.Comm,
    twist_index: int,
) -> _TwistStepDiagnostics | None:
    """Gather each twist's local diagnostics to the world root (logging only).

    Collective over ``world_comm``: every rank must call it, once per step,
    also when its ``finalize_optimization_step`` failed (``step_result`` is
    then ``None`` and the rank sends nothing). Each twist's sampler leader
    sends a few scalars, at most N top rows, one row per local block and its
    local coordinate counts (the twist-local values agree over
    ``sampler_comm``); the other ranks send ``None``. Returns the merged
    diagnostics on world rank 0 and ``None`` elsewhere. The optimizer arrays
    are only read.
    """
    local = _coordinate_scope_indices(coordinate_map, "sampler")
    force = np.asarray(target.coordinate_force)
    force_std = np.asarray(target.coordinate_force_std)
    signal_to_noise = np.asarray(target.signal_to_noise)
    payload = None
    if sampler_comm.Get_rank() == 0 and step_result is not None:
        maxima = None
        if local.size:
            local_argmax = int(local[np.argmax(np.abs(force[local]))])
            maxima = (float(force[local_argmax]), float(force_std[local_argmax]), float(np.max(signal_to_noise[local])))
        payload = {
            "twist": int(twist_index),
            "maxima": maxima,
            "top_rows": _top_coordinate_rows(target, coordinate_map, local, twist_index),
            "block_rows": _block_diagnostic_rows(target, step_result, coordinate_map, scope="sampler", twist_index=twist_index),
            "counts": _scope_coordinate_counts(target, coordinate_map, "sampler"),
        }
    gathered = world_comm.gather(payload, root=0)
    if world_comm.Get_rank() != 0 or step_result is None:
        return None
    twists = sorted((entry for entry in gathered if entry is not None), key=lambda entry: entry["twist"])
    shared = _coordinate_scope_indices(coordinate_map, "world")
    forces, force_errors, maxima, rows = [], [], [], []
    if shared.size:
        shared_argmax = int(shared[np.argmax(np.abs(force[shared]))])
        forces.append(float(force[shared_argmax]))
        force_errors.append(float(force_std[shared_argmax]))
        maxima.append(float(np.max(signal_to_noise[shared])))
        rows.extend(_top_coordinate_rows(target, coordinate_map, shared))
    for entry in twists:
        if entry["maxima"] is not None:
            twist_force, twist_force_std, twist_max_snr = entry["maxima"]
            forces.append(twist_force)
            force_errors.append(twist_force_std)
            maxima.append(twist_max_snr)
        rows.extend(entry["top_rows"])
    max_force_position = int(np.argmax(np.abs(np.asarray(forces))))
    row_order = np.argsort(np.asarray([row.signal_to_noise for row in rows]), kind="stable")[::-1]

    # Block rows in this rank's coordinate-map order: a shared block once, a
    # twist-local block once per twist; local blocks that only other twists
    # hold follow, twist by twist.
    shared_rows = {row.label: row for row in _block_diagnostic_rows(target, step_result, coordinate_map, scope="world")}
    block_rows: list[_BlockDiagnosticRow] = []
    listed: set[str] = set()
    for block in coordinate_map.blocks:
        if block.sync_scope != "sampler":
            block_rows.append(shared_rows[block.name])
            continue
        for entry in twists:
            for row in entry["block_rows"]:
                if row.label == f"{block.name}[t{entry['twist']}]":
                    block_rows.append(row)
                    listed.add(row.label)
    for entry in twists:
        block_rows.extend(row for row in entry["block_rows"] if row.label not in listed)

    real, storage, frozen, frozen_groups = _scope_coordinate_counts(target, coordinate_map, "world")
    for entry in twists:
        twist_real, twist_storage, twist_frozen, twist_frozen_groups = entry["counts"]
        real += twist_real
        storage += twist_storage
        frozen += twist_frozen
        frozen_groups += twist_frozen_groups
    return _TwistStepDiagnostics(
        max_force=forces[max_force_position],
        max_force_std=force_errors[max_force_position],
        max_signal_to_noise=float(np.max(np.asarray(maxima))),
        twist_local_max_signal_to_noise=tuple(
            (entry["twist"], entry["maxima"][2]) for entry in twists if entry["maxima"] is not None
        ),
        top_rows=tuple(rows[int(position)] for position in row_order[: _diagnostics_top_count(len(rows))]),
        block_rows=tuple(block_rows),
        coordinates_all_twists=(real, storage, real - frozen, frozen, frozen_groups),
    )


def _gather_all_twist_counts(
    coordinate_map: OptimizationCoordinateMap,
    *,
    sampler_comm: MPI.Comm,
    world_comm: MPI.Comm,
) -> tuple[int, int] | None:
    """``(values, real coordinates)`` optimized over all twists: shared + sum over twists of local.

    Collective over ``world_comm``. Returns the totals on world rank 0 and
    ``None`` elsewhere.
    """
    shared_values = local_values = shared_coordinates = local_coordinates = 0
    for block, coordinate_slice in zip(coordinate_map.blocks, coordinate_map.block_coordinate_slices, strict=True):
        coordinates = int(coordinate_slice.stop - coordinate_slice.start)
        if block.sync_scope == "sampler":
            local_values += int(block.size)
            local_coordinates += coordinates
        else:
            shared_values += int(block.size)
            shared_coordinates += coordinates
    payload = (local_values, local_coordinates) if sampler_comm.Get_rank() == 0 else None
    gathered = world_comm.gather(payload, root=0)
    if world_comm.Get_rank() != 0:
        return None
    twists = [entry for entry in gathered if entry is not None]
    return (
        shared_values + sum(entry[0] for entry in twists),
        shared_coordinates + sum(entry[1] for entry in twists),
    )


def _log_optimization_diagnostics(
    target: RealCoordinateOptimizationTarget,
    step_result: OptimizationStepResult,
    coordinate_map: OptimizationCoordinateMap,
    *,
    use_lm: bool,
    lm_subspace_dim: int,
    cg: bool,
    twist_diagnostics: _TwistStepDiagnostics | None = None,
) -> None:
    """Emit the detailed optimizer diagnostics selected by ``verbosity=high``.

    The linear-method predicted energy is intentionally not logged.  It remains
    an internal acceptance input, while the user-visible diagnostics report the
    resulting gate decision and its reason.

    ``twist_diagnostics`` (world root of a twist-decomposed step only) adds
    the all-twists coordinate counts and replaces this rank's top coordinates
    and block rows by the ones merged over every twist.
    """
    metric_diagonal = np.asarray(target.metric_diagonal)
    frozen_coordinate_count = sum(len(group) for group in target.frozen_coordinate_groups)
    active_coordinate_count = coordinate_map.coordinate_size - frozen_coordinate_count
    active_mask = np.ones(coordinate_map.coordinate_size, dtype=bool)
    for group in target.frozen_coordinate_groups:
        active_mask[np.asarray(group, dtype=np.int64)] = False
    active_diagonal = metric_diagonal[active_mask]
    logger.debug("Optimizer diagnostics:")
    logger.debug(
        "  Coordinates: real=%d, storage=%d, active=%d, frozen=%d in %d atomic group(s).",
        coordinate_map.coordinate_size,
        coordinate_map.storage_size,
        active_coordinate_count,
        frozen_coordinate_count,
        len(target.frozen_coordinate_groups),
    )
    if twist_diagnostics is not None:
        logger.debug(
            "  Coordinates (all twists): real=%d, storage=%d, active=%d, frozen=%d in %d atomic group(s).",
            *twist_diagnostics.coordinates_all_twists,
        )
    logger.debug(
        "  SR metric diagonal: all[min=%.6e, max=%.6e], active[min=%.6e, max=%.6e].",
        float(np.min(metric_diagonal, initial=np.inf)),
        float(np.max(metric_diagonal, initial=-np.inf)),
        float(np.min(active_diagonal, initial=np.inf)),
        float(np.max(active_diagonal, initial=-np.inf)),
    )
    logger.debug(
        "  SR solve: regime=%s, method=%s, iterations=%d, residual=%.6e, tolerance=%.6e, converged=%s.",
        target.sr_regime,
        "CG" if cg else "direct",
        target.num_iterations,
        target.residual_norm,
        # Absolute threshold cg_tol * ||b|| that the residual is compared
        # with (cg_tol is relative since issue #859).
        target.residual_tolerance,
        target.converged,
    )
    logger.debug(
        "  SR direction: ||q_SR||=%.6e, max|q_SR|=%.6e.",
        float(np.linalg.norm(target.sr_q)),
        float(np.max(np.abs(target.sr_q), initial=0.0)),
    )

    # A twist-decomposed step prints, on the world root, the rows merged over
    # the shared coordinates and every twist's local ones.
    top_rows = (
        twist_diagnostics.top_rows
        if twist_diagnostics is not None
        else _top_coordinate_rows(target, coordinate_map, np.arange(target.signal_to_noise.size))
    )
    logger.debug("  Top %d coordinates by generalized-force signal-to-noise:", len(top_rows))
    for row in top_rows:
        logger.debug(
            "    %s: force=%+.6e +- %.6e, signal-to-noise=%.6e, metric_diag=%.6e.",
            row.label,
            row.force,
            row.force_std,
            row.signal_to_noise,
            row.metric_diag,
        )

    if use_lm and lm_subspace_dim == 0:
        logger.debug(
            "  aSR decision: scale=%.6e, model_fallback=%s.",
            target.accepted_scale,
            target.rejection_reason is not None,
        )
    elif use_lm:
        selected_coordinate_count = sum(len(group) for group in target.selected_coordinate_groups)
        logger.debug(
            "  LM subspace: SR collective + %d coordinate(s) in %d atomic group(s).",
            selected_coordinate_count,
            len(target.selected_coordinate_groups),
        )
        if target.rejection_reason is None:
            logger.debug(
                "  LM decision: accepted (reference weight=%.8f).",
                float(target.lm_reference_weight) if target.lm_reference_weight is not None else float("nan"),
            )
        else:
            logger.debug(
                "  LM decision: rejected; fallback scale=%.6e (%s).",
                target.accepted_scale,
                target.rejection_reason,
            )
            if target.lm_reference_weight is not None:
                logger.debug("  Rejected LM candidate reference weight=%.8f.", target.lm_reference_weight)

    if step_result.diagnostics.linearized_wavefunction_norm is not None:
        logger.debug(
            "  Change in linearized ln|Psi| norm = %.6e.",
            step_result.diagnostics.linearized_wavefunction_norm,
        )
    logger.debug(
        "  Final real-coordinate update: ||delta_q||=%.6e, max|delta_q|=%.6e.",
        float(np.linalg.norm(step_result.q)),
        float(np.max(np.abs(step_result.q), initial=0.0)),
    )

    block_rows = (
        twist_diagnostics.block_rows
        if twist_diagnostics is not None
        else _block_diagnostic_rows(target, step_result, coordinate_map)
    )
    for row in block_rows:
        logger.debug(
            "  [optimizer block] %s: ||force||=%.6e, max|force|=%.6e, max(S/N)=%.6e, metric_diag[min=%.6e, max=%.6e].",
            row.label,
            row.force_norm,
            row.max_abs_force,
            row.max_signal_to_noise,
            row.metric_diag_min,
            row.metric_diag_max,
        )
        logger.debug(
            "    update: ||delta_q||=%.6e, max|delta_q|=%.6e, ||delta_p||=%.6e, max|delta_p|=%.6e.",
            row.delta_q_norm,
            row.max_abs_delta_q,
            row.delta_p_norm,
            row.max_abs_delta_p,
        )
        if row.stiefel_tangent is not None:
            logger.debug(
                "    S-Stiefel tangent: ||delta_C S^(1/2)||_F=%.6e, ||delta_C S^(1/2)||_2=%.6e.",
                *row.stiefel_tangent,
            )

    logger.log(5, "  Frozen coordinate groups = %s", target.frozen_coordinate_groups)
    logger.log(5, "  Selected LM coordinate groups = %s", target.selected_coordinate_groups)
    logger.log(5, "  SR q = %s", target.sr_q)
    logger.log(5, "  Final delta_q = %s", step_result.q)


def _require_finite_sr_moments(
    metric_diagonal: npt.ArrayLike,
    coordinate_force: npt.ArrayLike,
    agreement_comm: MPI.Comm | None,
) -> None:
    """Reject non-finite SR inputs from their reduced moments, on every rank together.

    ``diag_q = sum_i |X_qi|^2`` and ``b_q = sum_i Re(X_qi F_i)`` are non-finite
    whenever any centered sample entry of ``X`` or ``F`` is (NaN and inf
    propagate through the sums, and ``0 * inf`` is NaN), so testing these
    already-reduced coordinate vectors is the host check of
    :func:`prepare_wide_sr_real_system` without another pass over the samples.
    Being reduced over the solve's communicator, they agree on its ranks;
    ``agreement_comm`` adds one flag ``allreduce`` when ranks outside it must
    raise too (the twist-local scopes of a twist-decomposed step).
    """
    finite = bool(np.all(np.isfinite(metric_diagonal)) and np.all(np.isfinite(coordinate_force)))
    if agreement_comm is not None:
        finite = bool(agreement_comm.allreduce(int(finite), op=MPI.MIN))
    if not finite:
        raise ValueError("wide SR sample arrays must be finite.")


def prepare_wide_sr_real_system(
    X_q_local: npt.ArrayLike,
    F_local: npt.ArrayLike,
    *,
    coordinate_groups: Sequence[Sequence[int]],
    diagonal_floor: float,
    sample_counts: Sequence[int],
    metric_diagonal: npt.ArrayLike | None = None,
    coordinate_force: npt.ArrayLike | None = None,
) -> WideSRRealSystem:
    """Freeze atomic groups and row-precondition one wide or tall real SR system.

    ``X_q_local`` and ``F_local`` stay in their published dtype (complex at a
    twist, real for a real wave function) until the complete contractions.
    For distributed callers, ``metric_diagonal`` and ``coordinate_force`` are
    the already all-reduced global values; the local contractions are used
    when they are omitted. ``sample_counts`` are the per-rank sample counts of
    the ranks the solve reduces over (``(N,)`` for one rank).

    Finiteness is checked once and collectively: a non-finite entry of ``X``
    makes its row of the globally reduced metric diagonal non-finite, and a
    non-finite ``F`` entry makes the reduced coordinate force non-finite, so
    every rank raises together; the solvers do not scan the matrix again.

    The storage of the preconditioned matrix follows the solve that will
    consume it, decided by :func:`select_sr_real_regime` from the GLOBAL
    sample count: a wide system on the host Rust route is written
    sample-major, which the Rust wide CG streams in one pass per operator
    application; a tall system stays coordinate-major, the row-major ``Z`` of
    the tall solvers.
    """
    X_q = np.asarray(X_q_local)
    F = np.asarray(F_local)
    if X_q.ndim != 2 or F.ndim != 1 or X_q.shape[1] != F.size:
        raise ValueError("wide SR requires X_q=(coordinate, sample) and a matching one-dimensional F.")
    if not np.issubdtype(X_q.dtype, np.number) or not np.issubdtype(F.dtype, np.number):
        raise ValueError("wide SR sample arrays must be numeric.")
    if not np.isfinite(diagonal_floor) or diagonal_floor < 0:
        raise ValueError("wide SR diagonal_floor must be finite and non-negative.")

    coordinate_size = int(X_q.shape[0])
    groups = tuple(tuple(int(index) for index in group) for group in coordinate_groups)
    flattened_groups = tuple(index for group in groups for index in group)
    if any(not group for group in groups) or sorted(flattened_groups) != list(range(coordinate_size)):
        raise ValueError("wide SR coordinate_groups must form an exact partition of the coordinate axis.")

    if metric_diagonal is None:
        diagonal = np.real(np.einsum("qi,qi->q", X_q, _kernel._conjugate_if_complex(X_q)))
    else:
        diagonal = np.asarray(metric_diagonal)
    if coordinate_force is None:
        force = np.real(X_q @ F)
    else:
        force = np.asarray(coordinate_force)
    if diagonal.shape != (coordinate_size,) or force.shape != (coordinate_size,):
        raise ValueError("wide SR global diagonal and coordinate force must match the coordinate axis.")
    if np.iscomplexobj(diagonal) or np.iscomplexobj(force):
        raise ValueError("wide SR metric diagonal and coordinate force must be real.")
    _require_finite_sr_moments(diagonal, force, None)

    valid = diagonal > diagonal_floor
    frozen_groups = tuple(group for group in groups if not np.all(valid[np.asarray(group, dtype=np.int64)]))
    frozen_indices = {index for group in frozen_groups for index in group}
    active_indices = np.asarray(
        [index for index in range(coordinate_size) if index not in frozen_indices],
        dtype=np.int64,
    )
    inverse_sqrt = np.zeros(coordinate_size, dtype=np.result_type(diagonal.dtype, np.float32))
    inverse_sqrt[active_indices] = 1.0 / np.sqrt(diagonal[active_indices])
    scale = inverse_sqrt[active_indices]
    regime = select_sr_real_regime(int(active_indices.size), sample_counts, complex_samples=_complex_sr_samples(X_q, F))
    if regime == "wide" and _kernel._host_rust_kernel("optimizer_wide_sr_real_cg") is not None:
        # The wide solve takes this system: hand it to the host Rust CG
        # sample-major, which it streams in one pass per operator application.
        X_preconditioned = _kernel._sample_major_preconditioned_rows(X_q, active_indices, scale)
    else:
        X_preconditioned = _owned_array_op(np.multiply, X_q[active_indices], scale[:, None])
    X_preconditioned.setflags(write=False)
    b_preconditioned = force[active_indices] * inverse_sqrt[active_indices]

    def _freeze(values: npt.ArrayLike) -> np.ndarray:
        result = np.array(values, copy=True)
        result.setflags(write=False)
        return result

    return WideSRRealSystem(
        X_preconditioned_local=X_preconditioned,
        b_preconditioned=_freeze(b_preconditioned),
        metric_diagonal=_freeze(diagonal),
        inverse_sqrt_diagonal=_freeze(inverse_sqrt),
        active_indices=_freeze(active_indices),
        frozen_coordinate_groups=frozen_groups,
    )


def restore_wide_sr_coordinates(
    q_preconditioned: npt.ArrayLike,
    prepared: WideSRRealSystem,
    *,
    coordinate_size: int,
) -> np.ndarray:
    """De-precondition survivors and restore frozen coordinates as zeros."""
    if not isinstance(prepared, WideSRRealSystem):
        raise ValueError("restore_wide_sr_coordinates requires a WideSRRealSystem.")
    q_pc = np.asarray(q_preconditioned)
    if np.iscomplexobj(q_pc) or q_pc.shape != (prepared.active_indices.size,):
        raise ValueError("wide SR preconditioned coordinates must be a matching real vector.")
    if coordinate_size != prepared.metric_diagonal.size:
        raise ValueError("wide SR coordinate_size does not match the prepared system.")
    q = np.zeros(coordinate_size, dtype=np.result_type(q_pc.dtype, prepared.inverse_sqrt_diagonal.dtype))
    q[prepared.active_indices] = q_pc * prepared.inverse_sqrt_diagonal[prepared.active_indices]
    return q


def _wide_sr_layout_fingerprint(
    coordinate_map: OptimizationCoordinateMap,
    prepared: WideSRRealSystem,
    *,
    epsilon: float,
    cg_max_iter: int,
    cg_tol: float,
) -> str:
    payload = (
        "wide-real-primal-v1",
        coordinate_map.layout_fingerprint(),
        tuple(int(index) for index in prepared.active_indices),
        "diag-correlation-survivor-only-v1",
        np.dtype(prepared.b_preconditioned.dtype).str,
        float(epsilon),
        int(cg_max_iter),
        float(cg_tol),
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _weighted_log_derivative_sum(batch: OptimizationBatch, coordinate_map: OptimizationCoordinateMap) -> np.ndarray:
    """This rank's ``sum_n w_n O_q[n]`` from the binned jackknife moments.

    The bins already hold ``sum w O`` of every source component. The
    coordinate transform is linear with real weights, so transforming the
    one summed source row gives the weighted column sum without transforming
    the sample-by-coordinate matrix (the device route does the same).
    """
    totals = []
    for block, moment in zip(coordinate_map.blocks, batch.jackknife_bin_metadata.log_source_moments, strict=True):
        components = {}
        for name in ("direct", "d_p", "d_pbar"):
            values = getattr(moment.source_sum, name)
            if values is not None:
                components[name] = np.sum(values, axis=0, keepdims=True)
        totals.append(BlockDerivativeSource.build(block=block, **components))
    weighted = coordinate_map.transform_derivatives(tuple(totals))
    if weighted.shape != (1, coordinate_map.coordinate_size):
        raise ValueError("jackknife source moments have an inconsistent coordinate axis.")
    return weighted[0]


SRTargetSystem: TypeAlias = tuple[np.ndarray, WideSRRealSystem, tuple[int, ...]]


def _prepare_sr_target_system(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    comm: MPI.Comm | None,
    diagonal_floor: float,
) -> SRTargetSystem:
    """Center and precondition one host SR system for either solver route.

    Returns the local ``F``, the prepared system and the per-rank sample
    counts of ``comm``, from which :func:`select_sr_real_regime` decides the
    route of this system once.
    """

    def reduce_sum(value: npt.ArrayLike) -> npt.ArrayLike:
        return value if comm is None else comm.allreduce(value, op=MPI.SUM)

    weights = np.asarray(batch.weights)
    local_count = int(weights.size)
    counts = (local_count,) if comm is None else tuple(int(count) for count in comm.allgather(local_count))
    global_weight = reduce_sum(np.sum(weights))
    global_weighted_O = reduce_sum(_weighted_log_derivative_sum(batch, coordinate_map))
    global_weighted_energy = reduce_sum(np.dot(weights, batch.local_energies))
    samples = build_sr_coordinate_samples(
        batch,
        coordinate_map,
        total_weight=global_weight,
        total_weighted_log_derivatives=global_weighted_O,
        total_weighted_energy=global_weighted_energy,
    )
    diagonal = reduce_sum(np.real(np.einsum("qi,qi->q", samples.X_q_local, _kernel._conjugate_if_complex(samples.X_q_local))))
    coordinate_force = reduce_sum(np.real(samples.X_q_local @ samples.F_local))
    prepared = prepare_wide_sr_real_system(
        samples.X_q_local,
        samples.F_local,
        coordinate_groups=coordinate_map.coordinate_groups,
        diagonal_floor=diagonal_floor,
        sample_counts=counts,
        metric_diagonal=diagonal,
        coordinate_force=coordinate_force,
    )
    return samples.F_local, prepared, counts


def compute_wide_sr_real_target(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    comm: MPI.Comm | None = None,
    epsilon: float,
    diagonal_floor: float,
    cg: bool,
    cg_max_iter: int,
    cg_tol: float,
    warm_start: SolverWarmStart | None,
    _system: SRTargetSystem | None = None,
) -> WideSRTargetResult:
    """Execute all wide routes on the internal, non-production target path."""

    def reduce_sum(value: npt.ArrayLike) -> npt.ArrayLike:
        return value if comm is None else comm.allreduce(value, op=MPI.SUM)

    _F_local, prepared, _counts = (
        _prepare_sr_target_system(
            _collective_transform_batch(batch, coordinate_map, comm), coordinate_map, comm=comm, diagonal_floor=diagonal_floor
        )
        if _system is None
        else _system
    )
    fingerprint = _wide_sr_layout_fingerprint(
        coordinate_map,
        prepared,
        epsilon=epsilon,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
    )
    if prepared.active_indices.size == 0:
        q_pc = np.empty(0, dtype=prepared.b_preconditioned.dtype)
        residual = 0.0
        iterations = 0
        breakdown = False
        warm_out = SolverWarmStart.build(values=q_pc, layout_fingerprint=fingerprint) if cg else None
    else:
        if (
            warm_start is not None
            and warm_start.representation == "wide_real_primal"
            and warm_start.layout_fingerprint == fingerprint
            and warm_start.values.shape == prepared.b_preconditioned.shape
        ):
            x0 = warm_start.values
        else:
            x0 = np.zeros_like(prepared.b_preconditioned)
        if cg:
            # Per-iteration coordinate-vector SUM: buffer Allreduce, never
            # mpi4py object (pickle) serialization.
            q_pc, residual, iterations, breakdown = _kernel.solve_wide_sr_real_cg(
                prepared.X_preconditioned_local,
                prepared.b_preconditioned,
                epsilon,
                max_iter=cg_max_iter,
                tol=cg_tol,
                x0=x0,
                reduce_sum=None if comm is None else (lambda values: _allreduce_sum_array(comm, values)),
            )
        else:
            q_pc = _kernel.solve_wide_sr_real_direct(
                prepared.X_preconditioned_local,
                prepared.b_preconditioned,
                epsilon,
                reduce_sum=reduce_sum,
            )
            residual = 0.0
            iterations = 0
            breakdown = False
        warm_out = SolverWarmStart.build(values=q_pc, layout_fingerprint=fingerprint) if cg else None

    q = restore_wide_sr_coordinates(q_pc, prepared, coordinate_size=coordinate_map.coordinate_size)
    return WideSRTargetResult(
        q=q,
        prepared=prepared,
        warm_start_out=warm_out,
        residual_norm=float(residual),
        num_iterations=int(iterations),
        breakdown=bool(breakdown),
        rhs_norm=_vector_norm(prepared.b_preconditioned),
    )


def _gather_tall_valid_layout(
    layout: TallSRRealLayout,
    *,
    comm: MPI.Comm | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather only valid dual entries in canonical rank-major order."""
    if comm is None:
        return np.asarray(layout.Z_valid_local), np.asarray(layout.h_valid_local)
    dtype = np.dtype(layout.Z_valid_local.dtype)
    if dtype == np.dtype(np.float64):
        mpi_dtype = MPI.DOUBLE
    elif dtype == np.dtype(np.float32):
        mpi_dtype = MPI.FLOAT
    else:
        raise ValueError(f"unsupported tall real SR MPI dtype: {dtype}")
    if comm.Get_size() == 1:
        # Nothing to gather: the local Z is the global one, in its own
        # (coordinate-major) storage, as for ``comm=None``.
        return np.asarray(layout.Z_valid_local), np.asarray(layout.h_valid_local)
    coordinate_size = layout.Z_valid_local.shape[0]
    total_valid = int(np.sum(layout.valid_counts))
    Z_flat = np.empty(total_valid * coordinate_size, dtype=dtype)
    flat_counts = np.asarray(layout.valid_counts * coordinate_size, dtype=np.int64)
    flat_displacements = np.asarray(layout.valid_displacements * coordinate_size, dtype=np.int64)
    comm.Allgatherv(
        [np.ascontiguousarray(layout.Z_valid_local.T).reshape(-1), mpi_dtype],
        [Z_flat, (flat_counts.tolist(), flat_displacements.tolist()), mpi_dtype],
    )
    h_global = np.empty(total_valid, dtype=layout.h_valid_local.dtype)
    comm.Allgatherv(
        [np.ascontiguousarray(layout.h_valid_local), mpi_dtype],
        [
            h_global,
            (layout.valid_counts.tolist(), layout.valid_displacements.tolist()),
            mpi_dtype,
        ],
    )
    return Z_flat.reshape(total_valid, coordinate_size).T, h_global


def compute_tall_sr_real_target(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    comm: MPI.Comm | None = None,
    epsilon: float,
    diagonal_floor: float,
    cg: bool,
    cg_max_iter: int,
    cg_tol: float,
    _system: SRTargetSystem | None = None,
) -> TallSRTargetResult:
    """Execute tall routes on the internal target path.

    The dual is initialized to zero inside this call and is deliberately not
    accepted or returned as persistent optimizer state.
    """

    F_local, prepared, counts = (
        _prepare_sr_target_system(
            _collective_transform_batch(batch, coordinate_map, comm), coordinate_map, comm=comm, diagonal_floor=diagonal_floor
        )
        if _system is None
        else _system
    )
    rank = 0 if comm is None else int(comm.Get_rank())
    regime = select_sr_real_regime(
        int(prepared.active_indices.size),
        counts,
        complex_samples=_complex_sr_samples(prepared.X_preconditioned_local, F_local),
    )
    if regime != "tall":
        raise ValueError("The tall SR target received a wide real-coordinate system.")
    layout = build_tall_sr_real_layout(
        prepared.X_preconditioned_local,
        F_local,
        sample_counts=counts,
        rank=rank,
    )
    Z_global, h_global = _gather_tall_valid_layout(layout, comm=comm)
    if cg:
        q_pc, residual, iterations, breakdown = _kernel.solve_tall_sr_real_cg(
            Z_global,
            h_global,
            epsilon,
            max_iter=cg_max_iter,
            tol=cg_tol,
        )
    else:
        q_pc = _kernel.solve_tall_sr_real_direct(Z_global, h_global, epsilon)
        residual = 0.0
        iterations = 0
        breakdown = False
    q = restore_wide_sr_coordinates(q_pc, prepared, coordinate_size=coordinate_map.coordinate_size)
    return TallSRTargetResult(
        q=q,
        prepared=prepared,
        dual_layout=layout,
        warm_start_out=None,
        residual_norm=float(residual),
        num_iterations=int(iterations),
        breakdown=bool(breakdown),
        # The tall CG solves for the gathered dual right-hand side h.
        rhs_norm=_global_vector_norm(layout.h_valid_local, comm),
    )


def compute_sr_real_target(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    comm: MPI.Comm | None = None,
    epsilon: float,
    diagonal_floor: float,
    cg: bool,
    cg_max_iter: int,
    cg_tol: float,
    warm_start: SolverWarmStart | None,
) -> tuple[str, WideSRTargetResult | TallSRTargetResult]:
    """Select the wide or tall route from the surviving real-coordinate dimension.

    A multi-rank entry: the real/complex route is decided once over ``comm``
    (:func:`_collective_transform_batch`).
    """
    return _compute_sr_real_target(
        _collective_transform_batch(batch, coordinate_map, comm),
        coordinate_map,
        comm=comm,
        epsilon=epsilon,
        diagonal_floor=diagonal_floor,
        cg=cg,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
        warm_start=warm_start,
    )


def _compute_sr_real_target(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    comm: MPI.Comm | None = None,
    epsilon: float,
    diagonal_floor: float,
    cg: bool,
    cg_max_iter: int,
    cg_tol: float,
    warm_start: SolverWarmStart | None,
) -> tuple[str, WideSRTargetResult | TallSRTargetResult]:
    """Select the wide or tall route from the surviving real-coordinate dimension."""

    system = _prepare_sr_target_system(batch, coordinate_map, comm=comm, diagonal_floor=diagonal_floor)
    F_local, prepared, counts = system
    regime = select_sr_real_regime(
        int(prepared.active_indices.size),
        counts,
        complex_samples=_complex_sr_samples(prepared.X_preconditioned_local, F_local),
    )
    common = dict(
        comm=comm,
        epsilon=epsilon,
        diagonal_floor=diagonal_floor,
        cg=cg,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
        _system=system,
    )
    if regime == "wide":
        result = compute_wide_sr_real_target(
            batch,
            coordinate_map,
            warm_start=warm_start,
            **common,
        )
    else:
        result = compute_tall_sr_real_target(batch, coordinate_map, **common)
    return regime, result


def _compute_device_sr_real_target_dense(
    batch: DeviceOptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    epsilon: float,
    diagonal_floor: float,
    all_real: bool,
) -> tuple[str, WideSRTargetResult | TallSRTargetResult]:
    """Dense device direct SR solve (``cg=False``) on the world mesh.

    The CG solves of the single-k device route always stream
    (:func:`_solve_device_sr_scope_streaming`); this dense route is reached
    only for the direct solves.
    """
    O_local = coordinate_map.transform_derivatives_device(batch.log_derivative_sources, all_real=all_real)
    X, F, diagonal_device, force_device, _energy = _kernel.prepare_device_sr_samples(
        O_local,
        batch.weights,
        batch.local_energies,
    )
    diagonal = np.asarray(diagonal_device)
    force = np.asarray(force_device)
    # Globally sharded moments: every process holds the same values.
    _require_finite_sr_moments(diagonal, force, None)
    valid = np.isfinite(diagonal) & (diagonal > diagonal_floor)
    groups = coordinate_map.coordinate_groups
    frozen_groups = tuple(group for group in groups if not np.all(valid[np.asarray(group, dtype=np.int64)]))
    frozen_indices = {index for group in frozen_groups for index in group}
    active = np.asarray(
        [index for index in range(coordinate_map.coordinate_size) if index not in frozen_indices],
        dtype=np.int64,
    )
    # ``X`` is the sharded global array: its sample axis is the global count.
    sample_count = int(X.shape[1])
    inverse_sqrt = np.zeros(coordinate_map.coordinate_size, dtype=np.result_type(diagonal.dtype, np.float32))
    inverse_sqrt[active] = 1.0 / np.sqrt(diagonal[active])
    regime = select_sr_real_regime(int(active.size), (sample_count,), complex_samples=_complex_sr_samples(X, F))
    # Frozen rows are scaled to exact zeros in X's own (donated) buffer.
    X_pc = _kernel.precondition_device_sr_samples(X, inverse_sqrt)
    del X
    b_pc = force[active] * inverse_sqrt[active]
    prepared = WideSRRealSystem(
        X_preconditioned_local=np.empty((active.size, 0), dtype=np.complex128),
        b_preconditioned=_freeze_array(b_pc),
        metric_diagonal=_freeze_array(diagonal),
        inverse_sqrt_diagonal=_freeze_array(inverse_sqrt),
        active_indices=_freeze_array(active),
        frozen_coordinate_groups=frozen_groups,
    )
    if active.size == 0:
        q_pc = np.empty(0, dtype=b_pc.dtype)
    elif regime == "wide":
        q_pc = _kernel.solve_device_wide_real_direct_sharded(
            X_pc,
            _scatter_active(b_pc, active, inverse_sqrt.size),
            epsilon,
            frozen_rows=_frozen_rows(active, inverse_sqrt.size),
        )[active]
    else:
        q_pc = _kernel.solve_device_tall_real_direct_sharded(X_pc, F, epsilon)[active]
    q = restore_wide_sr_coordinates(q_pc, prepared, coordinate_size=coordinate_map.coordinate_size)
    if regime == "wide":
        return regime, WideSRTargetResult(
            q=q,
            prepared=prepared,
            warm_start_out=None,
            residual_norm=0.0,
            num_iterations=0,
            breakdown=False,
            rhs_norm=_vector_norm(b_pc),
        )
    empty_layout = TallSRRealLayout(
        Z_valid_local=np.empty((active.size, 0)),
        h_valid_local=np.empty(0),
        sample_counts=(int(batch.weights.size),),
        # The real route's tall dual has one entry per sample (no imaginary block).
        valid_counts=np.asarray([(1 if all_real else 2) * int(batch.weights.size)]),
        valid_displacements=np.asarray([0]),
    )
    return regime, TallSRTargetResult(
        q=q,
        prepared=prepared,
        dual_layout=empty_layout,
        warm_start_out=None,
        residual_norm=0.0,
        num_iterations=0,
        breakdown=False,
        rhs_norm=_device_sample_vector_norm((F,)),
    )


def _frozen_rows(active: np.ndarray, size: int) -> np.ndarray:
    """The coordinate rows outside ``active``: the exact-zero rows of frozen groups."""
    return np.setdiff1d(np.arange(size, dtype=np.int64), active, assume_unique=True)


def _scatter_active(values: npt.ArrayLike, active: np.ndarray, size: int) -> np.ndarray:
    """A full-length coordinate vector holding ``values`` on ``active`` and zero on frozen rows."""
    full = np.zeros(size, dtype=np.asarray(values).dtype)
    full[active] = values
    return full


def compute_device_sr_real_target(
    batch: DeviceOptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    comm: MPI.Comm | None = None,
    epsilon: float,
    diagonal_floor: float,
    cg: bool,
    cg_max_iter: int,
    cg_tol: float,
    warm_start: SolverWarmStart | None,
    all_real: bool | None = None,
) -> tuple[str, WideSRTargetResult | TallSRTargetResult]:
    """Production single-k SR routed through the shared bounded builder.

    ``all_real`` is the device route decision of the calling optimizer entry
    (see :meth:`OptimizationCoordinateMap.transform_derivatives_device`). When
    it is ``None`` this call is the entry: the route is decided once over
    ``comm`` (:func:`_collective_transform_route`, collective).
    """
    if all_real is None:
        batch, all_real = _collective_transform_route(batch, coordinate_map, comm)
    if not cg:
        return _compute_device_sr_real_target_dense(
            batch,
            coordinate_map,
            epsilon=epsilon,
            diagonal_floor=diagonal_floor,
            all_real=all_real,
        )
    scope_comm = MPI.COMM_SELF if comm is None else comm
    indices = np.arange(coordinate_map.coordinate_size, dtype=np.int64)
    streamed = _solve_device_sr_scope_streaming(
        batch,
        coordinate_map,
        indices,
        sync_scope=None,
        comm=scope_comm,
        epsilon=epsilon,
        diagonal_floor=diagonal_floor,
        cg=cg,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
        # The single-k target reduces over the world mesh; see
        # compute_real_coordinate_optimization_target.
        device_collectives=True,
        warm_start=warm_start,
        all_real=all_real,
    )
    prepared = streamed.prepared
    if prepared is None:
        raise RuntimeError("streaming single-k SR did not publish its prepared coordinate system.")
    regime = "tall" if streamed.solver_regime.startswith("tall") else "wide"
    if regime == "wide":
        return regime, WideSRTargetResult(
            q=streamed.q,
            prepared=prepared,
            warm_start_out=streamed.warm_start_out,
            residual_norm=streamed.residual_norm,
            num_iterations=streamed.num_iterations,
            breakdown=streamed.breakdown,
            rhs_norm=streamed.rhs_norm,
        )
    counts = streamed.sample_counts
    valid_counts = np.asarray([(1 if all_real else 2) * count for count in counts], dtype=np.int64)
    valid_displacements = np.zeros(len(counts), dtype=np.int64)
    if len(counts) > 1:
        valid_displacements[1:] = np.cumsum(valid_counts[:-1])
    empty_layout = TallSRRealLayout(
        Z_valid_local=np.empty((prepared.active_indices.size, 0)),
        h_valid_local=np.empty(0),
        sample_counts=counts,
        valid_counts=_freeze_array(valid_counts),
        valid_displacements=_freeze_array(valid_displacements),
    )
    return regime, TallSRTargetResult(
        q=streamed.q,
        prepared=prepared,
        dual_layout=empty_layout,
        warm_start_out=None,
        residual_norm=streamed.residual_norm,
        num_iterations=streamed.num_iterations,
        breakdown=streamed.breakdown,
        rhs_norm=streamed.rhs_norm,
    )


def _freeze_array(values: npt.ArrayLike) -> np.ndarray:
    result = np.array(values, copy=True)
    result.setflags(write=False)
    return result


def _vector_norm(values: npt.ArrayLike) -> float:
    """Euclidean norm of a replicated SR right-hand side (0 when empty)."""
    return float(np.linalg.norm(np.asarray(values))) if np.size(values) else 0.0


def _global_vector_norm(local_values: npt.ArrayLike, comm: MPI.Comm | None) -> float:
    """Euclidean norm of a vector whose entries are distributed over ``comm``."""
    local = np.asarray(local_values)
    local_sq = float(np.real(np.vdot(local, local))) if local.size else 0.0
    total_sq = local_sq if comm is None or _comm_size(comm) == 1 else float(comm.allreduce(local_sq, op=MPI.SUM))
    return float(np.sqrt(total_sq))


def _device_sample_vector_norm(F_chunks: Sequence[jax.Array], *, comm: MPI.Comm | None = None) -> float:
    """Norm of the realified tall dual right-hand side ``[Re F; -Im F]``.

    It equals ``sqrt(sum |F|^2)`` over every chunk (and every rank of ``comm``).
    """
    local_sq = sum(float(jnp.real(jnp.vdot(F, F))) for F in F_chunks)
    total_sq = local_sq if comm is None or _comm_size(comm) == 1 else float(comm.allreduce(local_sq, op=MPI.SUM))
    return float(np.sqrt(total_sq))


def _allreduce_sum_array(comm: MPI.Comm | None, values: npt.ArrayLike) -> np.ndarray:
    """SUM a numeric NumPy array without mpi4py object serialization."""
    local = np.ascontiguousarray(values)
    if comm is None:
        return np.array(local, copy=True)
    result = np.empty_like(local)
    dtype = local.dtype
    if dtype == np.dtype(np.float64):
        mpi_dtype = MPI.DOUBLE
    elif dtype == np.dtype(np.float32):
        mpi_dtype = MPI.FLOAT
    elif dtype == np.dtype(np.complex128):
        # Keep the complex reduction independent of MPI implementation support
        # for C_DOUBLE_COMPLEX, matching the established MCMC allreduce path.
        local = local.view(np.float64)
        result_view = result.view(np.float64)
        comm.Allreduce([local, MPI.DOUBLE], [result_view, MPI.DOUBLE], op=MPI.SUM)
        return result
    else:
        raise TypeError(f"unsupported optimizer Allreduce dtype: {dtype}")
    comm.Allreduce([local, mpi_dtype], [result, mpi_dtype], op=MPI.SUM)
    return result


def _build_lm_real_pencil_from_coordinate_arrays(
    O_q: npt.ArrayLike,
    D_q: npt.ArrayLike,
    weights: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    reduce_sum: ArrayReduction | None = None,
) -> LMRealPencil:
    """Build the centered LM pencil after coordinate transformation."""
    O = np.asarray(O_q)
    D = np.asarray(D_q)
    w = np.asarray(weights)
    e = np.asarray(energies)
    if O.ndim != 2 or D.shape != O.shape or w.shape != (O.shape[0],) or e.shape != w.shape:
        raise ValueError("LM coordinate samples require matching O_q, D_q, weights, and energies.")
    if np.iscomplexobj(w) or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("LM weights must be finite, non-negative, and real.")
    if not np.all(np.isfinite(O)) or not np.all(np.isfinite(D)) or not np.all(np.isfinite(e)):
        raise ValueError("LM coordinate samples must be finite.")
    native = _kernel._host_rust_kernel("optimizer_lm_pencil_core")
    if native is not None:
        return _build_lm_real_pencil_from_coordinate_arrays_rust(native, O, D, w, e, reduce_sum=reduce_sum)
    reduce = (lambda value: value) if reduce_sum is None else reduce_sum
    total_weight = np.asarray(reduce(np.sum(w)))
    if total_weight.ndim != 0 or not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("LM total weight must be a positive finite scalar.")
    weighted_O = np.asarray(reduce(np.einsum("n,nq->q", w, O)))
    weighted_D = np.asarray(reduce(np.einsum("n,nq->q", w, D)))
    weighted_e = np.asarray(reduce(np.dot(w, e)))
    O_bar = weighted_O / total_weight
    D_bar = weighted_D / total_weight
    e_bar = weighted_e / total_weight
    dO = O - O_bar
    dD = D - D_bar
    de = e - e_bar
    conj_dO = _kernel._conjugate_if_complex(dO)
    S_c = np.asarray(reduce(np.einsum("n,na,nb->ab", w, conj_dO, dO))) / total_weight
    f_c = -2.0 * np.asarray(reduce(np.einsum("n,na,n->a", w, conj_dO, de))) / total_weight
    K_c = np.asarray(reduce(np.einsum("n,n,na,nb->ab", w, e, conj_dO, dO))) / total_weight
    B_c = np.asarray(reduce(np.einsum("n,na,nb->ab", w, _kernel._conjugate_if_complex(dD), dO))) / total_weight
    # UNCONJUGATED first-row covariance for the bra estimator (see below).
    f_bra_cov = -2.0 * np.asarray(reduce(np.einsum("n,na,n->a", w, dO, de))) / total_weight
    # NON-SYMMETRIC linear method. Every pencil element is evaluated by
    # its OWN estimator and NOT Hermitian-averaged:
    #   H_{0a} (bra) = <dO_a de> + D_bar_a      (carries the finite <de_L/dc>)
    #   H_{a0} (ket) = <dO_a* de>               (= -f_c/2, covariance form)
    #   H_{ab}       = K_c + B_c^dagger         (dE on the KET index)
    # Under exact |Psi|^2 sampling Hermiticity makes H_c Hermitian in
    # expectation and this coincides with a symmetrized
    # construction. On an imperfectly equilibrated sample the difference is
    # exactly the information the optimizer needs: a symmetrization
    # (M_c = ((K+B)+(K+B)^dagger)/2, H_{0a} = H_{a0} = -f_c/2) silently
    # discards D_bar = <de_L/dc> (up to -7.7 Ha per unit parameter on far)
    # and drives the optimization 31 mHa uphill to a spurious fixed point.
    # TurboRVB's default VMC-LM uses the same non-symmetric structure.
    M_c = K_c + B_c.conj().T
    H_0 = float(np.real(e_bar))
    coordinate_size = O.shape[1]
    # Complex at a twist; real (float64) for a real wave function's real O/D.
    # The entry's collective route decision (collective_real_route) gives O, D
    # and e the same real/complex layout on every rank of ``reduce_sum``, so
    # the local dtype is that of the reduced moments and no complex moment is
    # cast into a real pencil.
    pencil_dtype = np.result_type(O.dtype, D.dtype, e.dtype)
    H_c = np.zeros((coordinate_size + 1, coordinate_size + 1), dtype=pencil_dtype)
    S_c_aug = np.zeros_like(H_c)
    H_c[0, 0] = H_0
    H_c[1:, 0] = -0.5 * f_c
    H_c[0, 1:] = -0.5 * f_bra_cov + D_bar
    H_c[1:, 1:] = M_c
    S_c_aug[0, 0] = 1.0
    S_c_aug[1:, 1:] = S_c
    # Optimization coordinates are real, while their tangent observables may
    # be complex at a twist.  The real quadratic form is therefore the outer
    # real projection of the centered complex pencil.  Keeping the reference
    # row orthogonal makes the projective tangent metric explicit and gives
    # the same q = w[1:] / w[0] update as the centered LM.
    # H stays NON-symmetric; only the (Hermitian-by-construction) S is
    # numerically re-symmetrized.
    H_real = np.real(H_c)
    S_real = np.real(S_c_aug)
    conditioning_metric = np.real(S_c)
    S_real = 0.5 * (S_real + S_real.T)
    conditioning_metric = 0.5 * (conditioning_metric + conditioning_metric.T)
    return LMRealPencil(
        H=_freeze_array(H_real),
        S=_freeze_array(S_real),
        H_0=H_0,
        conditioning_metric=_freeze_array(conditioning_metric),
    )


def _build_lm_real_pencil_from_coordinate_arrays_rust(
    native: Callable[..., Any],
    O: np.ndarray,
    D: np.ndarray,
    w: np.ndarray,
    e: np.ndarray,
    *,
    reduce_sum: ArrayReduction | None,
) -> LMRealPencil:
    """Rust (Rayon) body of :func:`_build_lm_real_pencil_from_coordinate_arrays`.

    Same centered non-symmetric estimators and the same nine reductions in the
    same order; complex second moments enter the real pencil only through
    their real parts, which are the reduced payloads.
    """
    O_stored, _O_transposed, O_complex = _kernel._rust_row_major(np.ascontiguousarray(O))
    D_stored, _D_transposed, D_complex = _kernel._rust_row_major(np.ascontiguousarray(D))
    e_complex = bool(np.iscomplexobj(e))
    e_stored = np.ascontiguousarray(e, dtype=np.complex128 if e_complex else np.float64)
    if e_complex:
        e_stored = e_stored.view(np.float64)

    def reduce_total(local_total: float) -> float:
        total_weight = np.asarray(reduce_sum(np.float64(local_total)))
        if total_weight.ndim != 0 or not np.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("LM total weight must be a positive finite scalar.")
        return float(total_weight)

    H, S, H_0, conditioning_metric = native(
        O_stored,
        O_complex,
        D_stored,
        D_complex,
        np.ascontiguousarray(w, dtype=np.float64),
        e_stored,
        e_complex,
        O.shape[1],
        None if reduce_sum is None else reduce_total,
        None if reduce_sum is None else _kernel._rust_vector_reduction(reduce_sum),
    )
    real_dtype = np.finfo(np.result_type(O.dtype, D.dtype, e.dtype)).dtype
    return LMRealPencil(
        H=_freeze_array(np.asarray(H, dtype=real_dtype)),
        S=_freeze_array(np.asarray(S, dtype=real_dtype)),
        H_0=float(H_0),
        conditioning_metric=_freeze_array(np.asarray(conditioning_metric, dtype=real_dtype)),
    )


def build_lm_real_pencil(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    comm: MPI.Comm | None = None,
) -> LMRealPencil:
    """Transform exact O/D sources and build the full real pencil.

    A multi-rank entry: the real/complex route is decided once over ``comm``
    (:func:`_collective_transform_batch`).
    """
    return _build_lm_real_pencil(_collective_transform_batch(batch, coordinate_map, comm), coordinate_map, comm=comm)


def _build_lm_real_pencil(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    comm: MPI.Comm | None = None,
) -> LMRealPencil:
    """Transform exact O/D sources and build the full real pencil."""
    if batch.state_token != coordinate_map.state_token:
        raise ValueError("OptimizationBatch state_token does not match the current coordinate map.")
    if batch.local_energy_derivative_sources is None:
        raise ValueError("LM requires exact local-energy derivative sources.")
    O_q = coordinate_map.transform_derivatives(batch.log_derivative_sources)
    D_q = coordinate_map.transform_derivatives(batch.local_energy_derivative_sources)

    def reduce_sum(value: npt.ArrayLike) -> npt.ArrayLike:
        return value if comm is None else comm.allreduce(value, op=MPI.SUM)

    return _build_lm_real_pencil_from_coordinate_arrays(
        O_q,
        D_q,
        batch.weights,
        batch.local_energies,
        reduce_sum=reduce_sum,
    )


def _conjugate_derivative_source(
    block: VariationalParameterBlock,
    source: BlockDerivativeSource,
) -> BlockDerivativeSource:
    """Conjugate every stored source component without folding policies."""

    def conjugate(values: np.ndarray | None) -> np.ndarray | None:
        return None if values is None else np.conj(values)

    return BlockDerivativeSource.build(
        block=block,
        direct=conjugate(source.direct),
        d_p=conjugate(source.d_p),
        d_pbar=conjugate(source.d_pbar),
    )


def _conjugate_device_derivative_source(
    block: VariationalParameterBlock,
    source: DeviceBlockDerivativeSource,
) -> DeviceBlockDerivativeSource:
    def conjugate(values: DerivativeArray | None) -> DerivativeArray | None:
        return None if values is None else jnp.conj(values)

    return DeviceBlockDerivativeSource.build(
        block=block,
        direct=conjugate(source.direct),
        d_p=conjugate(source.d_p),
        d_pbar=conjugate(source.d_pbar),
    )


def estimate_coordinate_force_jackknife_device(
    batch: DeviceOptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    comm: MPI.Comm | None = None,
    all_real: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Device jackknife with only centered coordinate-vector Allreduces.

    ``all_real`` is the device route decision of the calling optimizer entry.
    When it is ``None`` this call is the entry: the route is decided once over
    ``comm`` (:func:`_collective_transform_route`, collective).
    """
    if all_real is None:
        batch, all_real = _collective_transform_route(batch, coordinate_map, comm)
    metadata = batch.jackknife_bin_metadata
    source_sums = tuple(moment.source_sum for moment in metadata.log_source_moments)
    conjugated = tuple(
        _conjugate_device_derivative_source(block, moment.conjugate_source_energy_sum)
        for block, moment in zip(coordinate_map.blocks, metadata.log_source_moments, strict=True)
    )
    O_local = coordinate_map.transform_derivatives_device(source_sums, all_real=all_real)
    OE_local = jnp.conj(coordinate_map.transform_derivatives_device(conjugated, all_real=all_real))
    weights = np.asarray(metadata.bin_weight_sums)
    energies = np.asarray(metadata.bin_weighted_energy_sums)

    def reduce_scalar(value: npt.ArrayLike) -> npt.ArrayLike:
        return value if comm is None else comm.allreduce(value, op=MPI.SUM)

    count = int(reduce_scalar(weights.size))
    total_weight = reduce_scalar(np.sum(weights))
    total_energy = reduce_scalar(np.sum(energies))
    total_O = _allreduce_sum_array(comm, np.asarray(jnp.sum(O_local, axis=0)))
    total_OE = _allreduce_sum_array(comm, np.asarray(jnp.sum(OE_local, axis=0)))

    def force(
        weight: npt.ArrayLike,
        energy: npt.ArrayLike,
        O: npt.ArrayLike,
        OE: npt.ArrayLike,
    ) -> jax.Array:
        return -2.0 * jnp.real((OE - jnp.conj(O) * (energy / weight)) / weight)

    full = np.asarray(force(total_weight, total_energy, total_O, total_OE))
    leave_weights = total_weight - weights
    mean_sum = np.zeros_like(full, dtype=np.float64)
    chunk_size = 64
    for start in range(0, weights.size, chunk_size):
        stop = min(start + chunk_size, weights.size)
        replicas = force(
            jnp.asarray(leave_weights[start:stop, None]),
            jnp.asarray(total_energy - energies[start:stop, None]),
            jnp.asarray(total_O) - O_local[start:stop],
            jnp.asarray(total_OE) - OE_local[start:stop],
        )
        mean_sum += np.asarray(jnp.sum(replicas, axis=0))
    mean = _allreduce_sum_array(comm, mean_sum) / count
    centered_sumsq = np.zeros_like(full, dtype=np.float64)
    for start in range(0, weights.size, chunk_size):
        stop = min(start + chunk_size, weights.size)
        replicas = force(
            jnp.asarray(leave_weights[start:stop, None]),
            jnp.asarray(total_energy - energies[start:stop, None]),
            jnp.asarray(total_O) - O_local[start:stop],
            jnp.asarray(total_OE) - OE_local[start:stop],
        )
        centered_sumsq += np.asarray(jnp.sum(jnp.square(replicas - jnp.asarray(mean)), axis=0))
    variance = (count - 1.0) / count * _allreduce_sum_array(comm, centered_sumsq)
    return _freeze_array(full), _freeze_array(np.sqrt(np.maximum(variance, 0.0)))


def estimate_coordinate_force_jackknife(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    comm: MPI.Comm | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild real-coordinate force and delete-one-bin standard errors.

    A multi-rank entry: the real/complex route is decided once over ``comm``
    (:func:`_collective_transform_batch`).
    """
    return _estimate_coordinate_force_jackknife(
        _collective_transform_batch(batch, coordinate_map, comm), coordinate_map, comm=comm
    )


def _estimate_coordinate_force_jackknife(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    comm: MPI.Comm | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild real-coordinate force and delete-one-bin standard errors."""
    if batch.state_token != coordinate_map.state_token:
        raise ValueError("OptimizationBatch state_token does not match the current coordinate map.")
    metadata = batch.jackknife_bin_metadata
    source_sums = tuple(moment.source_sum for moment in metadata.log_source_moments)
    native = _kernel._host_rust_kernel("optimizer_coordinate_force_jackknife")
    if native is not None:
        # The Rust port conjugates the energy moments on load (no conjugated copies)
        # and calls the same SUM reductions, in the same order, as the code below.
        energy_sums = tuple(moment.conjugate_source_energy_sum for moment in metadata.log_source_moments)
        source_specs, _ = _rust_block_specs(coordinate_map, validate_derivative_sources(coordinate_map.blocks, source_sums))
        energy_specs, _ = _rust_block_specs(coordinate_map, validate_derivative_sources(coordinate_map.blocks, energy_sums))

        def reduce_value(value: npt.ArrayLike) -> npt.ArrayLike:
            return value if comm is None else comm.allreduce(value, op=MPI.SUM)

        rust_force, rust_std = native(
            source_specs,
            energy_specs,
            coordinate_map.coordinate_size,
            np.ascontiguousarray(metadata.bin_weight_sums, dtype=np.float64),
            np.ascontiguousarray(metadata.bin_weighted_energy_sums, dtype=np.complex128),
            reduce_value,
            lambda values: _allreduce_sum_array(comm, values),
        )
        return _freeze_array(rust_force), _freeze_array(rust_std)
    conjugated_energy_sources = tuple(
        _conjugate_derivative_source(block, moment.conjugate_source_energy_sum)
        for block, moment in zip(coordinate_map.blocks, metadata.log_source_moments, strict=True)
    )
    O_sum_local = coordinate_map.transform_derivatives(source_sums)
    conjugate_O_energy_sum_local = _kernel._conjugate_if_complex(
        coordinate_map.transform_derivatives(conjugated_energy_sources)
    )
    weight_local = np.asarray(metadata.bin_weight_sums)
    energy_local = np.asarray(metadata.bin_weighted_energy_sums)

    def reduce_scalar(value: npt.ArrayLike) -> npt.ArrayLike:
        return value if comm is None else comm.allreduce(value, op=MPI.SUM)

    total_bin_count = int(reduce_scalar(int(weight_local.size)))
    if total_bin_count < 2:
        raise ValueError("coordinate-force jackknife requires at least two global bins.")
    total_weight = reduce_scalar(np.sum(weight_local))
    total_energy = reduce_scalar(np.sum(energy_local))
    total_O = _allreduce_sum_array(comm, np.sum(O_sum_local, axis=0))
    total_conjugate_O_energy = _allreduce_sum_array(comm, np.sum(conjugate_O_energy_sum_local, axis=0))
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("coordinate-force jackknife requires positive finite total weight.")

    # This is the covariance force only. A jackknife force that also carried
    # the mean local-energy derivative <d e_L / dc> would need per-bin sums
    # of w * d e_L / dc in the jackknife metadata. Until then, S/N below its
    # threshold certifies a stationary point of the COVARIANCE estimator, not
    # of the true gradient (a covariance-dead trap can read as "converged").
    def force(
        weight: npt.ArrayLike,
        energy_sum: npt.ArrayLike,
        O_sum: npt.ArrayLike,
        conjugate_O_energy_sum: npt.ArrayLike,
    ) -> np.ndarray:
        centered = conjugate_O_energy_sum - _kernel._conjugate_if_complex(O_sum) * (energy_sum / weight)
        return -2.0 * np.real(centered / weight)

    full_force = force(total_weight, total_energy, total_O, total_conjugate_O_energy)
    leave_weights = total_weight - weight_local
    if not np.all(np.isfinite(leave_weights)) or np.any(leave_weights <= 0):
        raise ValueError("coordinate-force jackknife leave-one-bin weight must remain positive.")

    # Keep peak host memory independent of the number of local bins.  In the
    # first pass only replica sums are reduced; in the second pass centered
    # squared deviations are reduced.  This is the same cancellation-resistant
    # scheme used by MCMC.get_E/get_aF/get_aT/get_gF.
    chunk_size = 64

    def replica_chunk(start: int, stop: int) -> np.ndarray:
        weights = leave_weights[start:stop]
        return force(
            weights[:, None],
            total_energy - energy_local[start:stop, None],
            total_O - O_sum_local[start:stop],
            total_conjugate_O_energy - conjugate_O_energy_sum_local[start:stop],
        )

    local_replica_sum = np.zeros_like(full_force, dtype=np.float64)
    for start in range(0, weight_local.size, chunk_size):
        local_replica_sum += np.sum(replica_chunk(start, min(start + chunk_size, weight_local.size)), axis=0)
    global_replica_sum = _allreduce_sum_array(comm, local_replica_sum)
    replicate_mean = global_replica_sum / total_bin_count

    local_centered_sumsq = np.zeros_like(full_force, dtype=np.float64)
    for start in range(0, weight_local.size, chunk_size):
        centered = replica_chunk(start, min(start + chunk_size, weight_local.size)) - replicate_mean
        local_centered_sumsq += np.sum(np.square(centered), axis=0)
    global_centered_sumsq = _allreduce_sum_array(comm, local_centered_sumsq)
    variance = (total_bin_count - 1.0) / total_bin_count * global_centered_sumsq
    return _freeze_array(full_force), _freeze_array(np.sqrt(np.maximum(variance, 0.0)))


def reduce_lm_real_pencil(
    pencil: LMRealPencil,
    direction: npt.ArrayLike,
    selected_indices: npt.ArrayLike,
) -> tuple[LMRealPencil, np.ndarray]:
    """Apply the explicit augmented reduced-basis congruence."""
    if not isinstance(pencil, LMRealPencil):
        raise ValueError("reduced LM requires an LMRealPencil.")
    g = np.asarray(direction)
    selected = np.asarray(selected_indices, dtype=np.int64)
    coordinate_size = pencil.H.shape[0] - 1
    if np.iscomplexobj(g) or g.shape != (coordinate_size,) or not np.all(np.isfinite(g)):
        raise ValueError("reduced LM direction must be a finite real coordinate vector.")
    if selected.ndim != 1 or np.unique(selected).size != selected.size:
        raise ValueError("reduced LM selected indices must be unique and one-dimensional.")
    if np.any(selected < 0) or np.any(selected >= coordinate_size):
        raise ValueError("reduced LM selected index is outside the coordinate axis.")
    A = np.zeros((coordinate_size, selected.size + 1), dtype=np.result_type(g.dtype, pencil.H.dtype))
    A[:, 0] = g
    if selected.size:
        A[selected, np.arange(1, selected.size + 1)] = 1.0
    A_aug = np.zeros((coordinate_size + 1, selected.size + 2), dtype=A.dtype)
    A_aug[0, 0] = 1.0
    A_aug[1:, 1:] = A
    H_reduced = A_aug.T @ pencil.H @ A_aug
    S_reduced = A_aug.T @ pencil.S @ A_aug
    conditioning_reduced = None
    if pencil.conditioning_metric is not None:
        conditioning_reduced = A.T @ pencil.conditioning_metric @ A
    return (
        LMRealPencil(
            # H stays NON-symmetric: the real congruence A^T H A of the
            # non-symmetric pencil preserves the bra/ket asymmetry (the
            # retained <de_L/dc>); re-symmetrizing here would discard it
            # again. S is Hermitian by construction -- its symmetrization is
            # numerical hygiene only.
            H=_freeze_array(H_reduced),
            S=_freeze_array(0.5 * (S_reduced + S_reduced.T)),
            H_0=pencil.H_0,
            conditioning_metric=None
            if conditioning_reduced is None
            else _freeze_array(0.5 * (conditioning_reduced + conditioning_reduced.T)),
        ),
        _freeze_array(A),
    )


def build_reduced_lm_real_pencil(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    direction: npt.ArrayLike,
    selected_indices: npt.ArrayLike,
    *,
    comm: MPI.Comm | None = None,
) -> tuple[LMRealPencil, np.ndarray]:
    """Build reduced O/D columns before contraction (memory-saving route).

    A multi-rank entry: the real/complex route is decided once over ``comm``
    (:func:`_collective_transform_batch`).
    """
    return _build_reduced_lm_real_pencil(
        _collective_transform_batch(batch, coordinate_map, comm), coordinate_map, direction, selected_indices, comm=comm
    )


def _build_reduced_lm_real_pencil(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    direction: npt.ArrayLike,
    selected_indices: npt.ArrayLike,
    *,
    comm: MPI.Comm | None = None,
) -> tuple[LMRealPencil, np.ndarray]:
    """Build reduced O/D columns before contraction (memory-saving route)."""
    if batch.local_energy_derivative_sources is None:
        raise ValueError("reduced LM requires exact local-energy derivative sources.")
    coordinate_size = coordinate_map.coordinate_size
    g = np.asarray(direction)
    selected = np.asarray(selected_indices, dtype=np.int64)
    if np.iscomplexobj(g) or g.shape != (coordinate_size,) or not np.all(np.isfinite(g)):
        raise ValueError("reduced LM direction must be a finite real coordinate vector.")
    if selected.ndim != 1 or np.unique(selected).size != selected.size:
        raise ValueError("reduced LM selected indices must be unique and one-dimensional.")
    if np.any(selected < 0) or np.any(selected >= coordinate_size):
        raise ValueError("reduced LM selected index is outside the coordinate axis.")
    A = np.zeros((coordinate_size, selected.size + 1), dtype=np.result_type(g.dtype, np.float32))
    A[:, 0] = g
    if selected.size:
        A[selected, np.arange(1, selected.size + 1)] = 1.0

    def reduce_sum(value: npt.ArrayLike) -> npt.ArrayLike:
        return value if comm is None else comm.allreduce(value, op=MPI.SUM)

    pencil = _build_lm_real_pencil_from_coordinate_arrays(
        coordinate_map.transform_derivatives(batch.log_derivative_sources) @ A,
        coordinate_map.transform_derivatives(batch.local_energy_derivative_sources) @ A,
        batch.weights,
        batch.local_energies,
        reduce_sum=reduce_sum,
    )
    return pencil, _freeze_array(A)


def build_reduced_lm_real_pencil_device(
    batch: DeviceOptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    direction: npt.ArrayLike,
    selected_indices: npt.ArrayLike,
    *,
    all_real: bool | None = None,
) -> tuple[LMRealPencil, np.ndarray]:
    """Contract the reduced LM basis blockwise before any full O/D allocation.

    ``all_real`` is the device route decision of the calling optimizer entry.
    This function takes no communicator; when ``all_real`` is ``None`` the
    route is this rank's own decision (:func:`_collective_transform_route`
    with ``comm=None``), so a multi-rank caller must pass ``all_real``.
    """
    if all_real is None:
        batch, all_real = _collective_transform_route(batch, coordinate_map, None)
    if batch.local_energy_derivative_sources is None:
        raise ValueError("reduced LM requires exact local-energy derivative sources.")
    coordinate_size = coordinate_map.coordinate_size
    g = np.asarray(direction)
    selected = np.asarray(selected_indices, dtype=np.int64)
    if np.iscomplexobj(g) or g.shape != (coordinate_size,) or not np.all(np.isfinite(g)):
        raise ValueError("reduced LM direction must be a finite real coordinate vector.")
    if selected.ndim != 1 or np.unique(selected).size != selected.size:
        raise ValueError("reduced LM selected indices must be unique and one-dimensional.")
    if np.any(selected < 0) or np.any(selected >= coordinate_size):
        raise ValueError("reduced LM selected index is outside the coordinate axis.")
    A = np.zeros((coordinate_size, selected.size + 1), dtype=np.result_type(g.dtype, np.float32))
    A[:, 0] = g
    if selected.size:
        A[selected, np.arange(1, selected.size + 1)] = 1.0
    O_reduced = _transform_device_derivatives_reduced_streaming(
        batch.log_derivative_sources,
        coordinate_map,
        A,
        sample_count=int(batch.weights.size),
        all_real=all_real,
    )
    D_reduced = jnp.asarray(
        coordinate_map.transform_host_derivatives_reduced(batch.local_energy_derivative_sources, A, all_real=all_real)
    )
    H, S, H_0, conditioning_metric = _kernel.build_device_lm_pencil(
        O_reduced,
        D_reduced,
        batch.weights,
        batch.local_energies,
    )
    return (
        LMRealPencil(
            H=_freeze_array(H),
            S=_freeze_array(S),
            H_0=H_0,
            conditioning_metric=_freeze_array(conditioning_metric),
        ),
        _freeze_array(A),
    )


def select_lm_coordinate_groups(
    scores: npt.ArrayLike,
    coordinate_groups: Sequence[Sequence[int]],
    *,
    coordinate_budget: int,
) -> np.ndarray:
    """Rank atomic coordinate groups by maximum score with boundary ties."""
    values = np.asarray(scores)
    if values.ndim != 1 or np.iscomplexobj(values):
        raise ValueError("LM selection scores must be a real vector.")
    groups = tuple(tuple(int(index) for index in group) for group in coordinate_groups)
    flattened = tuple(index for group in groups for index in group)
    if any(not group for group in groups) or sorted(flattened) != list(range(values.size)):
        raise ValueError("LM coordinate groups must partition the score axis.")
    if coordinate_budget == -1 or coordinate_budget >= values.size:
        return _freeze_array(np.arange(values.size, dtype=np.int64))
    if coordinate_budget <= 0:
        raise ValueError("LM coordinate budget must be positive or -1.")
    finite_scores = np.where(np.isfinite(values), values, -np.inf)
    group_scores = np.asarray([np.max(finite_scores[np.asarray(group, dtype=np.int64)]) for group in groups])
    order = sorted(range(len(groups)), key=lambda index: (-group_scores[index], index))
    chosen_groups = []
    chosen_size = 0
    boundary_score = None
    for group_index in order:
        score = group_scores[group_index]
        if chosen_size >= coordinate_budget and score != boundary_score:
            break
        chosen_groups.append(group_index)
        chosen_size += len(groups[group_index])
        boundary_score = score
    selected = [index for group_index in chosen_groups for index in groups[group_index]]
    return _freeze_array(np.asarray(selected, dtype=np.int64))


def solve_lm_real_pencil(
    pencil: LMRealPencil,
    *,
    condition_cutoff: float,
) -> LMRealResult:
    """Delegate the sole real GEP implementation to the numerical kernel.

    Raises:
        jqmc.jqmc_optimizer_kernel.LMNoRealRootError: no root of the
            conditioned pencil is numerically real, so no update is decoded;
            the optimization targets reject such an LM step.
    """
    if not isinstance(pencil, LMRealPencil):
        raise ValueError("LM solver requires an LMRealPencil.")
    q, energy, reference_weight, active = _kernel.solve_lm_real_pencil(
        pencil.H,
        pencil.S,
        pencil.H_0,
        condition_cutoff=condition_cutoff,
        conditioning_metric=pencil.conditioning_metric,
    )
    return LMRealResult(
        q=_freeze_array(q),
        predicted_energy=energy,
        reference_weight=reference_weight,
        active_indices=_freeze_array(active),
    )


def asr_coefficients_from_pencil(
    H: npt.ArrayLike,
    S: npt.ArrayLike,
    direction: npt.ArrayLike,
) -> tuple[float, float, float, float]:
    """Contract a centered real pencil along one real SR direction."""
    H_array = np.asarray(H)
    S_array = np.asarray(S)
    g = np.asarray(direction)
    if H_array.shape != S_array.shape or H_array.shape != (g.size + 1, g.size + 1):
        raise ValueError("aSR pencil and direction shapes are inconsistent.")
    if np.iscomplexobj(H_array) or np.iscomplexobj(S_array) or np.iscomplexobj(g):
        raise ValueError("aSR pencil and direction must be real.")
    overlap_scale = max(float(np.max(np.abs(S_array), initial=0.0)), 1.0)
    overlap_tolerance = get_optimizer_roundoff_tolerance(
        "lm_centered_overlap",
        np.result_type(S_array.dtype, np.float32),
        scale=overlap_scale,
    )
    if (
        np.max(np.abs(S_array[0, 1:]), initial=0.0) > overlap_tolerance
        or np.max(np.abs(S_array[1:, 0]), initial=0.0) > overlap_tolerance
    ):
        raise ValueError("aSR requires a centered pencil with a reference-orthogonal tangent block.")
    # NON-SYMMETRIC pencil: the rational model's linear coefficient is the
    # average of the bra and ket off-diagonal projections,
    # 2 gamma H_1 with H_1 = (H_{0g} + H_{g0})/2 = C_g + D_g/2 -- the bra row
    # carries the finite <de_L/dc> retained by the non-symmetric LM.
    # For a symmetric pencil this reduces bit-identically to the previous
    # H_array[0, 1:] @ g. The H_2 quadratic form picks the symmetric part of
    # the bulk automatically.
    return (
        float(H_array[0, 0]),
        float(0.5 * (H_array[0, 1:] @ g + H_array[1:, 0] @ g)),
        float(g @ H_array[1:, 1:] @ g),
        float(g @ S_array[1:, 1:] @ g),
    )


def select_asr_gamma_bounded(
    H_0: float,
    H_1: float,
    H_2: float,
    S_2: float,
    *,
    gamma_max: float,
    fallback_gamma: float,
) -> ASRBoundedResult:
    """Delegate the sole bounded aSR minimizer to the numerical kernel."""
    gamma, energy, used_fallback = _kernel.select_asr_gamma_bounded(
        H_0,
        H_1,
        H_2,
        S_2,
        gamma_max=gamma_max,
        fallback_gamma=fallback_gamma,
    )
    return ASRBoundedResult(
        gamma=gamma,
        predicted_energy=energy,
        used_fallback=used_fallback,
    )


def _select_lm_rejection_asr_fallback(
    pencil: LMRealPencil,
    basis: np.ndarray | None,
    sr_direction: np.ndarray,
    *,
    gamma_max: float,
    fallback_gamma: float,
) -> tuple[ASRBoundedResult, tuple[float, float, float, float]]:
    """Select the aSR fallback from an already-built full or reduced LM pencil."""
    direction = np.asarray(sr_direction)
    if basis is None:
        contraction_direction = direction
    else:
        reduced_basis = np.asarray(basis)
        if (
            reduced_basis.ndim != 2
            or reduced_basis.shape[0] != direction.size
            or reduced_basis.shape[1] != pencil.H.shape[0] - 1
            or not np.array_equal(reduced_basis[:, 0], direction)
        ):
            raise ValueError("reduced LM fallback basis must start with the exact SR collective direction.")
        contraction_direction = np.zeros(reduced_basis.shape[1], dtype=direction.dtype)
        contraction_direction[0] = 1.0
    coefficients = asr_coefficients_from_pencil(pencil.H, pencil.S, contraction_direction)
    result = select_asr_gamma_bounded(
        *coefficients,
        gamma_max=gamma_max,
        fallback_gamma=fallback_gamma,
    )
    return result, coefficients


def _warn_sr_cg_not_converged(
    solve: str,
    *,
    iterations: int,
    cg_max_iter: int,
    residual: float,
    target: float,
    breakdown: bool,
) -> None:
    """Log the one default-verbosity WARNING of an unconverged SR CG solve.

    The step is still applied; the line states why the CG stopped (iteration
    limit or breakdown) and how far its residual is from ``cg_tol * ||b||``.
    """
    logger.warning(
        "SR CG did not converge (%s): iterations=%d (cg_max_iter=%d), residual=%.6e, "
        "target cg_tol*||b||=%.6e, breakdown=%s. The unconverged SR step is applied.",
        solve,
        int(iterations),
        int(cg_max_iter),
        float(residual),
        float(target),
        bool(breakdown),
    )


def validate_sr_solver_options(
    *,
    epsilon: object,
    cg_tol: object,
    cg_max_iter: object,
    lm_subspace_dim: object,
) -> tuple[float, float, int, int]:
    """Return the SR/LM solver options, or raise ``ValueError`` for an invalid one.

    The single owner of their domains, applied where ``run_optimize`` parses
    ``optimizer_kwargs`` (the Python API and the CLI) and at the single-k and
    twist-decomposed target entries:

    * ``epsilon``: finite and ``>= 0`` (``0`` is the unregularized metric);
    * ``cg_tol``: finite and ``> 0`` (relative CG tolerance);
    * ``cg_max_iter``: an integer ``>= 1``;
    * ``lm_subspace_dim``: an integer ``>= -1`` (``0`` aSR, ``> 0`` reduced
      LM, ``-1`` full LM).

    Returns:
        ``(epsilon, cg_tol, cg_max_iter, lm_subspace_dim)`` as ``float``,
        ``float``, ``int``, ``int``.
    """

    def real_number(name: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"optimizer_kwargs[{name!r}] must be a real number; got {value!r}.")
        return float(value)

    def integer(name: str, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"optimizer_kwargs[{name!r}] must be an integer; got {value!r}.")
        return int(value)

    epsilon_value = real_number("epsilon", epsilon)
    if not np.isfinite(epsilon_value) or epsilon_value < 0.0:
        raise ValueError(f"optimizer_kwargs['epsilon'] must be finite and >= 0; got {epsilon!r}.")
    cg_tol_value = real_number("cg_tol", cg_tol)
    if not np.isfinite(cg_tol_value) or cg_tol_value <= 0.0:
        raise ValueError(f"optimizer_kwargs['cg_tol'] must be finite and > 0; got {cg_tol!r}.")
    cg_max_iter_value = integer("cg_max_iter", cg_max_iter)
    if cg_max_iter_value < 1:
        raise ValueError(f"optimizer_kwargs['cg_max_iter'] must be >= 1; got {cg_max_iter!r}.")
    lm_subspace_dim_value = integer("lm_subspace_dim", lm_subspace_dim)
    if lm_subspace_dim_value < -1:
        raise ValueError(
            f"optimizer_kwargs['lm_subspace_dim'] must be 0 (aSR), > 0 (reduced LM) or -1 (full LM); got {lm_subspace_dim!r}."
        )
    return epsilon_value, cg_tol_value, cg_max_iter_value, lm_subspace_dim_value


def compute_real_coordinate_optimization_target(
    batch: OptimizationBatch | DeviceOptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    comm: MPI.Comm | None = None,
    epsilon: float,
    diagonal_floor: float,
    cg: bool,
    cg_max_iter: int,
    cg_tol: float,
    warm_start: SolverWarmStart | None,
    use_lm: bool,
    lm_subspace_dim: int,
    lm_condition_cutoff: float,
    energy_standard_error: float,
    asr_gamma_max: float = OPTIMIZER_DEFAULT_ASR_GAMMA_MAX,
    asr_fallback_gamma: float = OPTIMIZER_DEFAULT_ASR_FALLBACK_GAMMA,
    lm_reference_weight_min: float = OPTIMIZER_DEFAULT_LM_REFERENCE_WEIGHT_MIN,
) -> RealCoordinateOptimizationTarget:
    """Compute one complete SR/LM/aSR target in canonical real coordinates.

    A host batch is solved on the host (mpi4py reductions over ``comm``). A
    device batch over several ranks reduces with the world-mesh JAX kernels
    (SR solves and LM pencils), so its ``comm`` must span every JAX process
    with one sample count per rank (:func:`_require_world_device_mesh`).
    """
    epsilon, cg_tol, cg_max_iter, lm_subspace_dim = validate_sr_solver_options(
        epsilon=epsilon, cg_tol=cg_tol, cg_max_iter=cg_max_iter, lm_subspace_dim=lm_subspace_dim
    )
    if isinstance(batch, DeviceOptimizationBatch) and _comm_size(comm) > 1:
        _require_world_device_mesh(comm, tuple((int(count),) for count in comm.allgather(int(batch.weights.size))))
    # One real/complex route decision over ``comm`` (#889), passed down as ``all_real``.
    batch, all_real = _collective_transform_route(batch, coordinate_map, comm)
    device_batch = isinstance(batch, DeviceOptimizationBatch)
    if device_batch:
        force, force_std = estimate_coordinate_force_jackknife_device(batch, coordinate_map, comm=comm, all_real=all_real)
    else:
        force, force_std = _estimate_coordinate_force_jackknife(batch, coordinate_map, comm=comm)
    tolerance = get_optimizer_roundoff_tolerance("force_standard_error", force_std.dtype)
    finite = np.isfinite(force_std) & (np.abs(force_std) > tolerance)
    signal_to_noise = np.zeros(force.shape, dtype=np.float64)
    signal_to_noise[finite] = np.abs(force[finite]) / force_std[finite]
    if device_batch:
        regime, sr_result = compute_device_sr_real_target(
            batch,
            coordinate_map,
            comm=comm,
            epsilon=epsilon,
            diagonal_floor=diagonal_floor,
            cg=cg,
            cg_max_iter=cg_max_iter,
            cg_tol=cg_tol,
            warm_start=warm_start,
            all_real=all_real,
        )
    else:
        regime, sr_result = _compute_sr_real_target(
            batch,
            coordinate_map,
            comm=comm,
            epsilon=epsilon,
            diagonal_floor=diagonal_floor,
            cg=cg,
            cg_max_iter=cg_max_iter,
            cg_tol=cg_tol,
            warm_start=warm_start,
        )
    sr_q = np.asarray(sr_result.q)
    sr_warm_start = sr_result.warm_start_out
    sr_iterations = sr_result.num_iterations
    sr_residual = sr_result.residual_norm
    sr_residual_tolerance = float(cg_tol) * float(sr_result.rhs_norm)
    sr_breakdown = bool(sr_result.breakdown)
    sr_converged = not cg or sr_residual <= sr_residual_tolerance
    if not sr_converged:
        # Every route reduces its CG scalars over the whole sampler, so this
        # decision and its values agree on all ranks.
        _warn_sr_cg_not_converged(
            f"regime={regime}",
            iterations=sr_iterations,
            cg_max_iter=cg_max_iter,
            residual=sr_residual,
            target=sr_residual_tolerance,
            breakdown=sr_breakdown,
        )
    sr_diagonal = sr_result.prepared.metric_diagonal
    sr_frozen_groups = sr_result.prepared.frozen_coordinate_groups
    del sr_result
    q = np.array(sr_q, copy=True)
    accepted_scale = 1.0
    predicted_energy = None
    reference_weight = None
    rejection_reason = None
    linearized_wavefunction_norm = None
    selected_groups = coordinate_map.coordinate_groups
    if use_lm:
        if lm_subspace_dim == 0:
            if device_batch:
                pencil, _basis = build_reduced_lm_real_pencil_device(
                    batch, coordinate_map, sr_q, np.empty(0, dtype=int), all_real=all_real
                )
            else:
                pencil, _basis = _build_reduced_lm_real_pencil(
                    batch,
                    coordinate_map,
                    sr_q,
                    np.empty(0, dtype=int),
                    comm=comm,
                )
            coefficients = asr_coefficients_from_pencil(pencil.H, pencil.S, np.ones(1))
            asr = select_asr_gamma_bounded(
                *coefficients,
                gamma_max=asr_gamma_max,
                fallback_gamma=asr_fallback_gamma,
            )
            q = asr.gamma * sr_q
            accepted_scale = asr.gamma
            predicted_energy = asr.predicted_energy
            linearized_wavefunction_norm = abs(asr.gamma) * np.sqrt(max(float(coefficients[-1]), 0.0))
            if asr.used_fallback:
                rejection_reason = "aSR model was invalid; configured fallback scale was used"
        else:
            if lm_subspace_dim == -1 or lm_subspace_dim >= coordinate_map.coordinate_size:
                if device_batch:
                    O_q = coordinate_map.transform_derivatives_device(batch.log_derivative_sources, all_real=all_real)
                    assert batch.local_energy_derivative_sources is not None
                    D_q = coordinate_map.transform_derivatives_device(batch.local_energy_derivative_sources, all_real=all_real)
                    H, S, H_0, conditioning_metric = _kernel.build_device_lm_pencil(
                        O_q,
                        D_q,
                        batch.weights,
                        batch.local_energies,
                    )
                    lm_pencil = LMRealPencil(
                        H=_freeze_array(H),
                        S=_freeze_array(S),
                        H_0=H_0,
                        conditioning_metric=_freeze_array(conditioning_metric),
                    )
                else:
                    lm_pencil = _build_lm_real_pencil(batch, coordinate_map, comm=comm)
                basis = None
            else:
                selected = select_lm_coordinate_groups(
                    signal_to_noise,
                    coordinate_map.coordinate_groups,
                    coordinate_budget=lm_subspace_dim,
                )
                selected_set = set(int(index) for index in selected)
                selected_groups = tuple(
                    group for group in coordinate_map.coordinate_groups if all(index in selected_set for index in group)
                )
                if device_batch:
                    lm_pencil, basis = build_reduced_lm_real_pencil_device(
                        batch,
                        coordinate_map,
                        sr_q,
                        selected,
                        all_real=all_real,
                    )
                else:
                    lm_pencil, basis = _build_reduced_lm_real_pencil(
                        batch,
                        coordinate_map,
                        sr_q,
                        selected,
                        comm=comm,
                    )
            try:
                lm = solve_lm_real_pencil(lm_pencil, condition_cutoff=lm_condition_cutoff)
            except _kernel.LMNoRealRootError:
                lm = None
            energy_limit = lm_pencil.H_0 + 3.0 * float(energy_standard_error)
            rejection_reasons = []
            if lm is None:
                rejection_reasons.append("no numerically real LM root")
            else:
                predicted_energy = lm.predicted_energy
                reference_weight = lm.reference_weight
                if lm.predicted_energy > energy_limit:
                    rejection_reasons.append("linear-method energy gate failed")
                if lm.reference_weight < lm_reference_weight_min:
                    rejection_reasons.append(
                        f"reference weight {lm.reference_weight:.8f} is below {lm_reference_weight_min:.8f}"
                    )
            if rejection_reasons:
                asr, coefficients = _select_lm_rejection_asr_fallback(
                    lm_pencil,
                    basis,
                    sr_q,
                    gamma_max=asr_gamma_max,
                    fallback_gamma=asr_fallback_gamma,
                )
                q = asr.gamma * sr_q
                accepted_scale = asr.gamma
                fallback_reason = (
                    f"aSR model invalid; configured plain-SR gamma={asr.gamma:.8f} used"
                    if asr.used_fallback
                    else f"aSR fallback gamma={asr.gamma:.8f}"
                )
                rejection_reason = "; ".join((*rejection_reasons, fallback_reason))
                predicted_energy = asr.predicted_energy
                linearized_wavefunction_norm = abs(asr.gamma) * np.sqrt(max(float(coefficients[-1]), 0.0))
            else:
                q = lm.q if basis is None else basis @ lm.q
                reduced_q = np.asarray(lm.q)
                metric_norm_sq = float(reduced_q @ lm_pencil.S[1:, 1:] @ reduced_q)
                linearized_wavefunction_norm = np.sqrt(max(metric_norm_sq, 0.0))
    if np.iscomplexobj(q) or q.shape != (coordinate_map.coordinate_size,) or not np.all(np.isfinite(q)):
        raise FloatingPointError("optimizer produced an invalid real-coordinate update.")
    return RealCoordinateOptimizationTarget(
        q=_freeze_array(q),
        sr_q=_freeze_array(sr_q),
        coordinate_force=force,
        coordinate_force_std=force_std,
        signal_to_noise=_freeze_array(signal_to_noise),
        sr_regime=regime,
        accepted_scale=float(accepted_scale),
        predicted_energy=predicted_energy,
        lm_reference_weight=reference_weight,
        warm_start_out=sr_warm_start,
        converged=sr_converged,
        num_iterations=sr_iterations,
        residual_norm=sr_residual,
        residual_tolerance=sr_residual_tolerance,
        metric_diagonal=_freeze_array(sr_diagonal),
        frozen_coordinate_groups=sr_frozen_groups,
        selected_coordinate_groups=selected_groups,
        rejection_reason=rejection_reason,
        linearized_wavefunction_norm=linearized_wavefunction_norm,
        cg_breakdown=sr_breakdown,
    )


def _coordinate_scope_indices(
    coordinate_map: OptimizationCoordinateMap,
    scope: str,
) -> np.ndarray:
    indices: list[int] = []
    for block, coordinate_slice in zip(
        coordinate_map.blocks,
        coordinate_map.block_coordinate_slices,
        strict=True,
    ):
        if block.sync_scope == scope:
            indices.extend(range(coordinate_slice.start, coordinate_slice.stop))
    return np.asarray(indices, dtype=np.int64)


def _select_planb_lm_scope_coordinates(
    signal_to_noise: np.ndarray,
    coordinate_map: OptimizationCoordinateMap,
    indices: np.ndarray,
    *,
    coordinate_budget: int,
    comm: MPI.Comm,
) -> np.ndarray | None:
    """Select one consensus LM coordinate set inside an ownership scope."""
    scope_indices = np.asarray(indices, dtype=np.int64)
    if coordinate_budget == -1 or coordinate_budget >= scope_indices.size:
        return None
    if coordinate_budget <= 0:
        raise ValueError("The scope-local LM coordinate budget must be positive or -1.")
    global_to_local = {int(index): position for position, index in enumerate(scope_indices)}
    local_coordinate_groups: list[tuple[int, ...]] = []
    for global_group in coordinate_map.coordinate_groups:
        local_group = tuple(global_to_local[index] for index in global_group if index in global_to_local)
        if local_group and len(local_group) != len(global_group):
            raise ValueError("An atomic LM coordinate group cannot cross shared/local ownership scopes.")
        if local_group:
            local_coordinate_groups.append(local_group)
    flattened_group_indices = sorted(index for group in local_coordinate_groups for index in group)
    if flattened_group_indices != list(range(scope_indices.size)):
        raise ValueError("Scope-local coordinate groups must partition the scope axis.")
    scope_groups = tuple(local_coordinate_groups)
    rank = 0 if comm is None else int(comm.Get_rank())
    selected = None
    if rank == 0:
        selected = select_lm_coordinate_groups(
            np.asarray(signal_to_noise)[scope_indices],
            scope_groups,
            coordinate_budget=coordinate_budget,
        )
    if comm is not None:
        selected = comm.bcast(selected, root=0)
    selected_array = np.asarray(selected, dtype=np.int64)
    if selected_array.ndim != 1 or np.unique(selected_array).size != selected_array.size:
        raise ValueError("Selected scope-local LM coordinates must be a unique one-dimensional vector.")
    if np.any(selected_array < 0) or np.any(selected_array >= scope_indices.size):
        raise ValueError("A selected LM coordinate is outside its ownership scope.")
    return _freeze_array(selected_array)


@dataclass(frozen=True)
class _PlanBSRScopeResult:
    q: np.ndarray
    metric_diagonal: np.ndarray
    frozen_coordinate_groups: tuple[tuple[int, ...], ...]
    residual_norm: float
    num_iterations: int
    solver_regime: str
    rhs_norm: float
    breakdown: bool
    prepared: WideSRRealSystem | None = None
    warm_start_out: SolverWarmStart | None = None
    sample_counts: tuple[int, ...] = ()


def _normalize_planb_sampler_weight(weight: float | None, sampler_comm: MPI.Comm, world_comm: MPI.Comm) -> float:
    """Normalize fixed sampler weights, counting each sampler only once."""
    value = 1.0 if weight is None else float(weight)
    valid = bool(np.isfinite(value) and value >= 0.0)
    if not world_comm.allreduce(int(valid), op=MPI.MIN):
        raise ValueError("Twist-sampler weights must be finite and non-negative.")
    minimum = sampler_comm.allreduce(value, op=MPI.MIN)
    maximum = sampler_comm.allreduce(value, op=MPI.MAX)
    if not world_comm.allreduce(int(minimum == maximum), op=MPI.MIN):
        raise ValueError("A twist-sampler weight must agree on all ranks within its sampler.")
    total = world_comm.allreduce(value / _comm_size(sampler_comm), op=MPI.SUM)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Twist-sampler weights must have a positive finite sum.")
    return value / total


def _center_planb_twist_samples(
    O: npt.ArrayLike,
    weights: npt.ArrayLike,
    energies: npt.ArrayLike,
    *,
    sampler_comm: MPI.Comm,
    world_comm: MPI.Comm,
    sampler_weight: float,
) -> tuple[Any, Any, Any]:
    """Build separately normalized shared tangents and a fixed-BZ LM model.

    Only coordinate moments cross the host boundary for device inputs. Each
    sampler contributes total sample weight W_k, irrespective of sample count
    or MPI size. Independent energy origins cancel from the energy-difference
    pencil; its common reference is the fixed BZ mean.
    """
    xp = jnp if isinstance(O, jax.Array) else np
    O, weights, energies = xp.asarray(O), xp.asarray(weights), xp.asarray(energies)
    total_weight = sampler_comm.allreduce(float(np.asarray(xp.sum(weights))), op=MPI.SUM)
    valid = bool(np.isfinite(total_weight) and total_weight > 0.0)
    if not world_comm.allreduce(int(valid), op=MPI.MIN):
        raise ValueError("Each twist sampler must have positive finite sampling weight.")
    O_mean = _allreduce_sum_array(sampler_comm, np.asarray(xp.einsum("n,nq->q", weights, O))) / total_weight
    energy_mean = sampler_comm.allreduce(np.asarray(xp.dot(weights, energies)), op=MPI.SUM) / total_weight
    reference = world_comm.allreduce(sampler_weight * float(np.real(energy_mean)) / _comm_size(sampler_comm), op=MPI.SUM)
    return (
        O - xp.asarray(O_mean),
        weights * (sampler_weight / total_weight),
        energies - xp.asarray(energy_mean) + reference,
    )


def _solve_planb_sr_scope(
    batch: OptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    indices: np.ndarray,
    *,
    comm: MPI.Comm,
    epsilon: float,
    diagonal_floor: float,
    cg: bool,
    cg_max_iter: int,
    cg_tol: float,
    normalization_comm: MPI.Comm | None = None,
    sampler_weight: float = 1.0,
    finite_comm: MPI.Comm | None = None,
) -> _PlanBSRScopeResult:
    """Solve one dense shared or twist-local diagonal block.

    ``finite_comm`` widens the non-finite-input agreement beyond ``comm``
    (:func:`_require_finite_sr_moments`).
    """
    if indices.size == 0:
        _require_finite_sr_moments((), (), finite_comm)
        return _PlanBSRScopeResult(
            q=_freeze_array(np.empty(0, dtype=np.float64)),
            metric_diagonal=_freeze_array(np.empty(0, dtype=np.float64)),
            frozen_coordinate_groups=(),
            residual_norm=0.0,
            num_iterations=0,
            breakdown=False,
            solver_regime="empty",
            rhs_norm=0.0,
        )
    weights = np.asarray(batch.weights)
    energies = np.asarray(batch.local_energies)
    O = coordinate_map.transform_derivatives(batch.log_derivative_sources)[:, indices]
    if normalization_comm is not None:
        O, weights, energies = _center_planb_twist_samples(
            O, weights, energies, sampler_comm=normalization_comm, world_comm=comm, sampler_weight=sampler_weight
        )

    def reduce(value: npt.ArrayLike) -> npt.ArrayLike:
        return comm.allreduce(value, op=MPI.SUM)

    total_weight = reduce(np.sum(weights))
    O_bar = reduce(np.einsum("n,nq->q", weights, O)) / total_weight
    energy_bar = np.real(reduce(np.dot(weights, energies)) / total_weight)
    X = (_kernel._conjugate_if_complex(O - O_bar) * np.sqrt(weights)[:, None] / np.sqrt(total_weight)).T
    F = -2.0 * np.sqrt(weights) * (energies - energy_bar) / np.sqrt(total_weight)
    diagonal = reduce(np.real(np.einsum("qi,qi->q", X, _kernel._conjugate_if_complex(X))))
    force = reduce(np.real(X @ F))
    _require_finite_sr_moments(diagonal, force, finite_comm)
    position = {int(global_index): local_index for local_index, global_index in enumerate(indices)}
    groups = tuple(
        tuple(position[index] for index in group)
        for group in coordinate_map.coordinate_groups
        if all(index in position for index in group)
    )
    sample_counts = tuple(int(count) for count in comm.allgather(int(F.size)))
    prepared = prepare_wide_sr_real_system(
        X,
        F,
        coordinate_groups=groups,
        diagonal_floor=diagonal_floor,
        sample_counts=sample_counts,
        metric_diagonal=diagonal,
        coordinate_force=force,
    )
    regime_kind = select_sr_real_regime(
        int(prepared.active_indices.size), sample_counts, complex_samples=_complex_sr_samples(X, F)
    )
    del X
    if prepared.active_indices.size == 0:
        q_pc = np.empty(0, dtype=np.float64)
        residual = 0.0
        iterations = 0
        breakdown = False
    elif regime_kind == "wide" and cg:
        q_pc, residual, iterations, breakdown = _kernel.solve_wide_sr_real_cg(
            prepared.X_preconditioned_local,
            prepared.b_preconditioned,
            epsilon,
            max_iter=cg_max_iter,
            tol=cg_tol,
            x0=np.zeros_like(prepared.b_preconditioned),
            reduce_sum=lambda values: _allreduce_sum_array(comm, values),
        )
    elif regime_kind == "wide":
        q_pc = _kernel.solve_wide_sr_real_direct(
            prepared.X_preconditioned_local,
            prepared.b_preconditioned,
            epsilon,
            reduce_sum=reduce,
        )
        residual = 0.0
        iterations = 0
        breakdown = False
    else:
        layout = build_tall_sr_real_layout(
            prepared.X_preconditioned_local,
            F,
            sample_counts=sample_counts,
            rank=int(comm.Get_rank()),
        )
        Z_global, h_global = _gather_tall_valid_layout(layout, comm=comm)
        if cg:
            q_pc, residual, iterations, breakdown = _kernel.solve_tall_sr_real_cg(
                Z_global,
                h_global,
                epsilon,
                max_iter=cg_max_iter,
                tol=cg_tol,
            )
        else:
            q_pc = _kernel.solve_tall_sr_real_direct(Z_global, h_global, epsilon)
            residual = 0.0
            iterations = 0
            breakdown = False
    return _PlanBSRScopeResult(
        q=_freeze_array(restore_wide_sr_coordinates(q_pc, prepared, coordinate_size=indices.size)),
        metric_diagonal=prepared.metric_diagonal,
        frozen_coordinate_groups=prepared.frozen_coordinate_groups,
        residual_norm=float(residual),
        num_iterations=int(iterations),
        breakdown=bool(breakdown),
        solver_regime=f"{regime_kind}-host-mpi",
        # The tall dual right-hand side is the realified F of every rank.
        rhs_norm=(_global_vector_norm(F, comm) if regime_kind == "tall" else _vector_norm(prepared.b_preconditioned)),
    )


def _estimate_planb_scope_force_device(
    batch: DeviceOptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    sync_scope: str,
    comm: MPI.Comm,
    all_real: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Jackknife one ownership scope while its binned O payload stays on device."""
    metadata = batch.jackknife_bin_metadata
    source_sums = tuple(moment.source_sum for moment in metadata.log_source_moments)
    conjugated = tuple(
        _conjugate_device_derivative_source(block, moment.conjugate_source_energy_sum)
        for block, moment in zip(coordinate_map.blocks, metadata.log_source_moments, strict=True)
    )
    O_local = coordinate_map.transform_derivatives_device(source_sums, sync_scope=sync_scope, all_real=all_real)
    OE_local = jnp.conj(coordinate_map.transform_derivatives_device(conjugated, sync_scope=sync_scope, all_real=all_real))
    weights = np.asarray(metadata.bin_weight_sums)
    energies = np.asarray(metadata.bin_weighted_energy_sums)

    def reduce_scalar(value: npt.ArrayLike) -> npt.ArrayLike:
        return comm.allreduce(value, op=MPI.SUM)

    count = int(reduce_scalar(int(weights.size)))
    if count < 2:
        raise ValueError("coordinate-force jackknife requires at least two global bins.")
    total_weight = reduce_scalar(np.sum(weights))
    total_energy = reduce_scalar(np.sum(energies))
    total_O = _allreduce_sum_array(comm, np.asarray(jnp.sum(O_local, axis=0)))
    total_OE = _allreduce_sum_array(comm, np.asarray(jnp.sum(OE_local, axis=0)))
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("coordinate-force jackknife requires positive finite total weight.")

    def force(
        weight: npt.ArrayLike,
        energy: npt.ArrayLike,
        O: npt.ArrayLike,
        OE: npt.ArrayLike,
    ) -> jax.Array:
        return -2.0 * jnp.real((OE - jnp.conj(O) * (energy / weight)) / weight)

    full = np.asarray(force(total_weight, total_energy, total_O, total_OE))
    leave_weights = total_weight - weights
    if not np.all(np.isfinite(leave_weights)) or np.any(leave_weights <= 0):
        raise ValueError("coordinate-force jackknife leave-one-bin weight must remain positive.")
    chunk_size = 64
    local_sum = np.zeros_like(full, dtype=np.float64)
    for start in range(0, weights.size, chunk_size):
        stop = min(start + chunk_size, weights.size)
        replicas = force(
            jnp.asarray(leave_weights[start:stop, None]),
            jnp.asarray(total_energy - energies[start:stop, None]),
            jnp.asarray(total_O) - O_local[start:stop],
            jnp.asarray(total_OE) - OE_local[start:stop],
        )
        local_sum += np.asarray(jnp.sum(replicas, axis=0))
    mean = _allreduce_sum_array(comm, local_sum) / count
    local_sumsq = np.zeros_like(full, dtype=np.float64)
    for start in range(0, weights.size, chunk_size):
        stop = min(start + chunk_size, weights.size)
        replicas = force(
            jnp.asarray(leave_weights[start:stop, None]),
            jnp.asarray(total_energy - energies[start:stop, None]),
            jnp.asarray(total_O) - O_local[start:stop],
            jnp.asarray(total_OE) - OE_local[start:stop],
        )
        local_sumsq += np.asarray(jnp.sum(jnp.square(replicas - jnp.asarray(mean)), axis=0))
    variance = (count - 1.0) / count * _allreduce_sum_array(comm, local_sumsq)
    return _freeze_array(full), _freeze_array(np.sqrt(np.maximum(variance, 0.0)))


def _solve_planb_sr_scope_device_dense(
    batch: DeviceOptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    indices: np.ndarray,
    *,
    sync_scope: str,
    comm: MPI.Comm,
    epsilon: float,
    diagonal_floor: float,
    cg: bool,
    cg_max_iter: int,
    cg_tol: float,
    normalization_comm: MPI.Comm | None = None,
    sampler_weight: float = 1.0,
    all_real: bool,
    finite_comm: MPI.Comm | None = None,
) -> _PlanBSRScopeResult:
    """Dense shared/local device oracle retained independently of streaming."""
    if indices.size == 0:
        _require_finite_sr_moments((), (), finite_comm)
        return _PlanBSRScopeResult(
            q=_freeze_array(np.empty(0, dtype=np.float64)),
            metric_diagonal=_freeze_array(np.empty(0, dtype=np.float64)),
            frozen_coordinate_groups=(),
            residual_norm=0.0,
            num_iterations=0,
            breakdown=False,
            solver_regime="empty",
            rhs_norm=0.0,
        )
    weights = batch.weights
    energies = batch.local_energies
    O = coordinate_map.transform_derivatives_device(batch.log_derivative_sources, sync_scope=sync_scope, all_real=all_real)
    if normalization_comm is not None:
        O, weights, energies = _center_planb_twist_samples(
            O, weights, energies, sampler_comm=normalization_comm, world_comm=comm, sampler_weight=sampler_weight
        )
    local_weight = np.asarray(jnp.sum(weights))
    local_weighted_O = np.asarray(jnp.einsum("n,nq->q", weights, O))
    local_weighted_energy = np.asarray(jnp.dot(weights, energies))
    total_weight = comm.allreduce(local_weight, op=MPI.SUM)
    total_weighted_O = _allreduce_sum_array(comm, local_weighted_O)
    total_weighted_energy = comm.allreduce(local_weighted_energy, op=MPI.SUM)
    X, F, diagonal_local, force_local, _energy = _kernel.prepare_local_device_sr_samples(
        O,
        weights,
        energies,
        total_weight=total_weight,
        total_weighted_O=total_weighted_O,
        total_weighted_energy=total_weighted_energy,
    )
    diagonal = _allreduce_sum_array(comm, np.asarray(diagonal_local))
    force = _allreduce_sum_array(comm, np.asarray(force_local))
    _require_finite_sr_moments(diagonal, force, finite_comm)
    position = {int(global_index): local_index for local_index, global_index in enumerate(indices)}
    groups = tuple(
        tuple(position[index] for index in group)
        for group in coordinate_map.coordinate_groups
        if all(index in position for index in group)
    )
    valid = np.isfinite(diagonal) & (diagonal > diagonal_floor)
    frozen_groups = tuple(group for group in groups if not np.all(valid[np.asarray(group, dtype=np.int64)]))
    frozen_indices = {index for group in frozen_groups for index in group}
    active = np.asarray([index for index in range(indices.size) if index not in frozen_indices], dtype=np.int64)
    inverse_sqrt = np.zeros(indices.size, dtype=np.result_type(diagonal.dtype, np.float32))
    inverse_sqrt[active] = 1.0 / np.sqrt(diagonal[active])
    # Frozen rows are scaled to exact zeros in X's own (donated) buffer; every
    # solve below runs on all rows, with zero right-hand side entries there.
    X_pc = _kernel.precondition_device_sr_samples(X, inverse_sqrt)
    del X
    b_pc = force[active] * inverse_sqrt[active]
    b_full = _scatter_active(b_pc, active, indices.size)
    prepared = WideSRRealSystem(
        X_preconditioned_local=np.empty((active.size, 0), dtype=np.complex128),
        b_preconditioned=_freeze_array(b_pc),
        metric_diagonal=_freeze_array(diagonal),
        inverse_sqrt_diagonal=_freeze_array(inverse_sqrt),
        active_indices=_freeze_array(active),
        frozen_coordinate_groups=frozen_groups,
    )
    sample_counts = (
        (int(weights.size),) if _comm_size(comm) == 1 else tuple(int(count) for count in comm.allgather(int(weights.size)))
    )
    regime_kind = select_sr_real_regime(int(active.size), sample_counts, complex_samples=_complex_sr_samples(X_pc, F))
    if active.size == 0:
        q_pc = np.empty(0, dtype=b_pc.dtype)
        residual = 0.0
        iterations = 0
        breakdown = False
        regime = "empty"
    elif _comm_size(comm) == 1:
        regime = f"{regime_kind}-device-local"
        if regime_kind == "wide":
            if cg:
                q_full, residual, iterations, breakdown = _kernel.solve_local_device_wide_real_cg(
                    X_pc,
                    b_full,
                    epsilon,
                    max_iter=cg_max_iter,
                    tol=cg_tol,
                    x0=np.zeros_like(b_full),
                )
            else:
                q_full = _kernel.solve_local_device_wide_real_direct(
                    X_pc, b_full, epsilon, frozen_rows=_frozen_rows(active, indices.size)
                )
                residual = 0.0
                iterations = 0
                breakdown = False
        elif cg:
            q_full, residual, iterations, breakdown = _kernel.solve_local_device_tall_real_cg(
                X_pc,
                F,
                epsilon,
                max_iter=cg_max_iter,
                tol=cg_tol,
            )
        else:
            q_full = _kernel.solve_local_device_tall_real_direct(X_pc, F, epsilon)
            residual = 0.0
            iterations = 0
            breakdown = False
        q_pc = np.asarray(q_full)[active]
    else:
        # JAX cannot call an mpi4py Allreduce from inside one compiled CG
        # loop.  Retain X on device for the common one-rank-per-twist path;
        # for a multi-rank sampler, stage only this ownership scope and use
        # the established MPI host solvers of the same regime.
        X_host = np.asarray(X_pc)
        del X_pc
        regime = f"{regime_kind}-device-staged-mpi"
        if regime_kind == "wide":

            def reduce(value: npt.ArrayLike) -> npt.ArrayLike:
                return comm.allreduce(value, op=MPI.SUM)

            if cg:
                q_full, residual, iterations, breakdown = _kernel.solve_wide_sr_real_cg(
                    X_host,
                    b_full,
                    epsilon,
                    max_iter=cg_max_iter,
                    tol=cg_tol,
                    x0=np.zeros_like(b_full),
                    reduce_sum=lambda values: _allreduce_sum_array(comm, values),
                )
            else:
                # The host direct solve factors the active block: its zero
                # frozen rows would make the shifted Gram singular at
                # epsilon = 0. The active entries are scattered back below.
                q_full = _scatter_active(
                    _kernel.solve_wide_sr_real_direct(
                        X_host if active.size == indices.size else X_host[active], b_pc, epsilon, reduce_sum=reduce
                    ),
                    active,
                    indices.size,
                )
                residual = 0.0
                iterations = 0
                breakdown = False
        else:
            layout = build_tall_sr_real_layout(
                X_host,
                np.asarray(F),
                sample_counts=sample_counts,
                rank=int(comm.Get_rank()),
            )
            Z_global, h_global = _gather_tall_valid_layout(layout, comm=comm)
            if cg:
                q_full, residual, iterations, breakdown = _kernel.solve_tall_sr_real_cg(
                    Z_global,
                    h_global,
                    epsilon,
                    max_iter=cg_max_iter,
                    tol=cg_tol,
                )
            else:
                q_full = _kernel.solve_tall_sr_real_direct(Z_global, h_global, epsilon)
                residual = 0.0
                iterations = 0
                breakdown = False
        q_pc = np.asarray(q_full)[active]
    return _PlanBSRScopeResult(
        q=_freeze_array(restore_wide_sr_coordinates(q_pc, prepared, coordinate_size=indices.size)),
        metric_diagonal=prepared.metric_diagonal,
        frozen_coordinate_groups=prepared.frozen_coordinate_groups,
        residual_norm=float(residual),
        num_iterations=int(iterations),
        breakdown=bool(breakdown),
        solver_regime=regime,
        # A tall dual right-hand side is the realified F of every rank.
        rhs_norm=_device_sample_vector_norm((F,), comm=comm) if regime.startswith("tall") else _vector_norm(b_pc),
    )


def _slice_device_derivative_sources(
    sources: Sequence[DeviceBlockDerivativeSource],
    start: int,
    stop: int,
) -> tuple[DeviceBlockDerivativeSource, ...]:
    """Move one bounded host/device sample slice to the active device."""
    return tuple(
        DeviceBlockDerivativeSource(
            name=source.name,
            direct=None if source.direct is None else jnp.asarray(source.direct[start:stop]),
            d_p=None if source.d_p is None else jnp.asarray(source.d_p[start:stop]),
            d_pbar=None if source.d_pbar is None else jnp.asarray(source.d_pbar[start:stop]),
        )
        for source in sources
    )


def _solve_device_sr_scope_streaming(
    batch: DeviceOptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    indices: np.ndarray,
    *,
    sync_scope: str | None,
    comm: MPI.Comm,
    epsilon: float,
    diagonal_floor: float,
    cg: bool,
    cg_max_iter: int,
    cg_tol: float,
    device_collectives: bool,
    warm_start: SolverWarmStart | None = None,
    all_real: bool,
    finite_comm: MPI.Comm | None = None,
) -> _PlanBSRScopeResult:
    """Solve one device SR scope without a full sample-by-coordinate allocation.

    The raw derivative publication is transformed once per bounded sample
    chunk.  Centered, diagonally preconditioned Jacobian chunks remain on the
    device and are passed as explicit operands to one compiled CG solve.
    ``sync_scope=None`` covers the ordinary single-k coordinate map; the
    twist-decomposed optimizer passes its sampler-owned scope. The dense
    implementations remain the independent numerical oracles.

    Over several ranks, ``device_collectives`` selects the reduction: the
    world-mesh JAX kernels (``comm`` must then span every JAX process, see
    :func:`_require_world_device_mesh`), or the chunks staged once to the host
    and the MPI matrix-free CG over exactly ``comm``.
    """
    if not cg:
        raise ValueError("device streaming SR requires cg=True.")
    if indices.size == 0:
        _require_finite_sr_moments((), (), finite_comm)
        return _PlanBSRScopeResult(
            q=_freeze_array(np.empty(0, dtype=np.float64)),
            metric_diagonal=_freeze_array(np.empty(0, dtype=np.float64)),
            frozen_coordinate_groups=(),
            residual_norm=0.0,
            num_iterations=0,
            breakdown=False,
            solver_regime="empty",
            rhs_norm=0.0,
        )

    weights = batch.weights
    energies = batch.local_energies
    sample_count = int(weights.size)
    chunk_size = min(sample_count, int(OPTIMIZER_DERIVATIVE_SAMPLE_BATCH))
    chunk_count = len(range(0, sample_count, chunk_size)) if chunk_size > 0 else 0
    # Every rank checks the same gathered layouts, so a bad layout raises on
    # every rank instead of one rank leaving the others blocked in the next
    # collective (issue #894).
    layouts = tuple((int(count), int(chunks)) for count, chunks in comm.allgather((sample_count, chunk_count)))
    sample_counts = tuple(count for count, _chunks in layouts)
    if min(sample_counts) <= 0:
        raise ValueError(f"device streaming SR requires samples on every rank; per-rank sample counts are {sample_counts}.")
    if chunk_size <= 0:
        raise ValueError("device streaming SR requires a positive sample batch size.")
    if device_collectives and _comm_size(comm) > 1:
        _require_world_device_mesh(comm, layouts)
    setup_start = time.perf_counter()

    local_weight = np.asarray(jnp.sum(weights))
    local_weighted_energy = np.asarray(jnp.dot(weights, energies))
    source_sums = []
    for block, moment in zip(coordinate_map.blocks, batch.jackknife_bin_metadata.log_source_moments, strict=True):
        components = {}
        for name in ("direct", "d_p", "d_pbar"):
            values = getattr(moment.source_sum, name)
            if values is not None:
                components[name] = jnp.sum(values, axis=0, keepdims=True)
        source_sums.append(DeviceBlockDerivativeSource.build(block=block, **components))
    weighted_O_total = coordinate_map.transform_derivatives_device(tuple(source_sums), sync_scope=sync_scope, all_real=all_real)
    if weighted_O_total.shape != (1, indices.size):
        raise RuntimeError("device streaming source moments have an inconsistent coordinate axis.")
    local_weighted_O = np.asarray(weighted_O_total[0])
    total_weight = comm.allreduce(local_weight, op=MPI.SUM)
    total_weighted_energy = comm.allreduce(local_weighted_energy, op=MPI.SUM)
    total_weighted_O = _allreduce_sum_array(comm, local_weighted_O)

    # Every bounded block but the last is centered straight into one
    # contiguous (coordinate x sample) head matrix (donated, in place); the
    # last, possibly shorter, block stays a separate tail. The CG then applies
    # each operator pass as one product per piece, not one per block. The
    # head is allocated once the first block's columns exist, so the device
    # peak never exceeds the per-block route: at the last block it holds the
    # same earlier samples plus that block.
    starts = tuple(range(0, sample_count, chunk_size))
    head_count = starts[-1]
    X_head = None
    F_chunks = []
    X_pieces = []
    diagonal_local = np.zeros(indices.size, dtype=np.float64)
    force_local = np.zeros(indices.size, dtype=np.float64)
    reduced_moments = dict(
        total_weight=total_weight,
        total_weighted_O=total_weighted_O,
        total_weighted_energy=total_weighted_energy,
    )
    for start in starts:
        stop = min(start + chunk_size, sample_count)
        source_chunk = _slice_device_derivative_sources(batch.log_derivative_sources, start, stop)
        O_chunk = coordinate_map.transform_derivatives_device(source_chunk, sync_scope=sync_scope, all_real=all_real)
        if stop <= head_count:
            if X_head is None:
                dtype = _kernel.local_device_sr_sample_dtype(
                    O_chunk, weights[start:stop], energies[start:stop], **reduced_moments
                )
                X_head = jnp.zeros((indices.size, head_count), dtype=dtype)
            X_head, F_chunk, diagonal_chunk, force_chunk, _energy = _kernel.prepare_local_device_sr_samples_into(
                X_head, start, O_chunk, weights[start:stop], energies[start:stop], **reduced_moments
            )
            X_head.block_until_ready()
        else:
            X_tail, F_chunk, diagonal_chunk, force_chunk, _energy = _kernel.prepare_local_device_sr_samples(
                O_chunk, weights[start:stop], energies[start:stop], **reduced_moments
            )
            X_tail.block_until_ready()
        del O_chunk
        F_chunks.append(F_chunk)
        diagonal_local += np.asarray(diagonal_chunk)
        force_local += np.asarray(force_chunk)
    if X_head is not None:
        X_pieces.append(X_head)
        F_pieces = [jnp.concatenate(F_chunks[:-1])] if len(F_chunks) > 2 else [F_chunks[0]]
    else:
        F_pieces = []
    X_pieces.append(X_tail)
    F_pieces.append(F_chunks[-1])
    del X_head, X_tail

    diagonal = _allreduce_sum_array(comm, diagonal_local)
    force = _allreduce_sum_array(comm, force_local)
    _require_finite_sr_moments(diagonal, force, finite_comm)
    position = {int(global_index): local_index for local_index, global_index in enumerate(indices)}
    groups = tuple(
        tuple(position[index] for index in group)
        for group in coordinate_map.coordinate_groups
        if all(index in position for index in group)
    )
    valid = np.isfinite(diagonal) & (diagonal > diagonal_floor)
    frozen_groups = tuple(group for group in groups if not np.all(valid[np.asarray(group, dtype=np.int64)]))
    frozen_indices = {index for group in frozen_groups for index in group}
    active = np.asarray([index for index in range(indices.size) if index not in frozen_indices], dtype=np.int64)
    inverse_sqrt = np.zeros(indices.size, dtype=np.result_type(diagonal.dtype, np.float32))
    inverse_sqrt[active] = 1.0 / np.sqrt(diagonal[active])
    b_pc = force[active] * inverse_sqrt[active]
    prepared = WideSRRealSystem(
        X_preconditioned_local=np.empty((active.size, 0), dtype=np.complex128),
        b_preconditioned=_freeze_array(b_pc),
        metric_diagonal=_freeze_array(diagonal),
        inverse_sqrt_diagonal=_freeze_array(inverse_sqrt),
        active_indices=_freeze_array(active),
        frozen_coordinate_groups=frozen_groups,
    )
    # The warm start persists only for the full single-k coordinate system.
    fingerprint = (
        _wide_sr_layout_fingerprint(
            coordinate_map,
            prepared,
            epsilon=epsilon,
            cg_max_iter=cg_max_iter,
            cg_tol=cg_tol,
        )
        if indices.size == coordinate_map.coordinate_size
        else None
    )
    if active.size == 0:
        setup_elapsed = time.perf_counter() - setup_start
        solve_start = time.perf_counter()
        q_pc = np.empty(0, dtype=b_pc.dtype)
        residual = 0.0
        iterations = 0
        breakdown = False
        regime = "empty"
    else:
        # Frozen rows are scaled to exact zeros in each piece's own (donated)
        # buffer; the CG runs on every row with zero right-hand side and
        # carry entries there, which keep their solution entries zero.
        X_pc_chunks = []
        for X_piece in X_pieces:
            X_pc = _kernel.precondition_device_sr_samples(X_piece, inverse_sqrt)
            X_pc.block_until_ready()
            X_pc_chunks.append(X_pc)
        del X_piece
        X_pieces.clear()
        setup_elapsed = time.perf_counter() - setup_start
        solve_start = time.perf_counter()
        regime_kind = select_sr_real_regime(
            int(active.size),
            sample_counts,
            complex_samples=any(_complex_sr_samples(X, F) for X, F in zip(X_pc_chunks, F_pieces, strict=True)),
        )
        staged = _comm_size(comm) > 1 and not device_collectives
        if staged:
            # JAX cannot reduce over an mpi4py subcommunicator inside one
            # compiled CG loop. Stage the chunks once into one host matrix,
            # release them on the device, and solve with the MPI CG of the
            # same regime over ``comm`` (as the dense sibling does).
            X_host = np.empty((indices.size, sample_count), dtype=np.result_type(*(X.dtype for X in X_pc_chunks)))
            column = 0
            for X_pc in X_pc_chunks:
                X_host[:, column : column + int(X_pc.shape[1])] = np.asarray(X_pc)
                column += int(X_pc.shape[1])
            del X_pc
            X_pc_chunks.clear()
        if regime_kind == "wide":
            if (
                fingerprint is not None
                and warm_start is not None
                and warm_start.layout_fingerprint == fingerprint
                and warm_start.values.shape == b_pc.shape
            ):
                x0 = warm_start.values
            else:
                x0 = np.zeros_like(b_pc)
            b_full = _scatter_active(b_pc, active, indices.size)
            x0_full = _scatter_active(x0, active, indices.size)
            if staged:
                regime = "wide-device-staged-mpi"
                q_full, residual, iterations, breakdown = _kernel.solve_wide_sr_real_cg(
                    X_host,
                    b_full,
                    epsilon,
                    max_iter=cg_max_iter,
                    tol=cg_tol,
                    x0=x0_full,
                    reduce_sum=lambda values: _allreduce_sum_array(comm, values),
                )
            elif _comm_size(comm) == 1:
                regime = "wide-device-local-streaming"
                q_full, residual, iterations, breakdown = _kernel.solve_local_device_wide_real_cg_chunks(
                    tuple(X_pc_chunks),
                    b_full,
                    epsilon,
                    max_iter=cg_max_iter,
                    tol=cg_tol,
                    x0=x0_full,
                )
            else:
                regime = "wide-device-sharded-streaming"
                q_full, residual, iterations, breakdown = _kernel.solve_device_wide_real_cg_chunks_sharded(
                    tuple(X_pc_chunks),
                    b_full,
                    epsilon,
                    max_iter=cg_max_iter,
                    tol=cg_tol,
                    x0=x0_full,
                )
        elif staged:
            regime = "tall-device-staged-mpi"
            layout = build_tall_sr_real_layout(
                X_host,
                np.asarray(jnp.concatenate(tuple(F_chunks))),
                sample_counts=sample_counts,
                rank=int(comm.Get_rank()),
            )
            Z_global, h_global = _gather_tall_valid_layout(layout, comm=comm)
            q_full, residual, iterations, breakdown = _kernel.solve_tall_sr_real_cg(
                Z_global,
                h_global,
                epsilon,
                max_iter=cg_max_iter,
                tol=cg_tol,
            )
            del layout, Z_global
        elif _comm_size(comm) == 1:
            regime = "tall-device-local-streaming"
            q_full, residual, iterations, breakdown = _kernel.solve_local_device_tall_real_cg_chunks(
                tuple(X_pc_chunks),
                tuple(F_pieces),
                epsilon,
                max_iter=cg_max_iter,
                tol=cg_tol,
            )
        else:
            regime = "tall-device-sharded-streaming"
            q_full, residual, iterations, breakdown = _kernel.solve_device_tall_real_cg_chunks_sharded(
                tuple(X_pc_chunks),
                tuple(F_pieces),
                epsilon,
                max_iter=cg_max_iter,
                tol=cg_tol,
            )
        if staged:
            del X_host
        q_pc = np.asarray(q_full)[active]
    solve_elapsed = time.perf_counter() - solve_start
    logger.debug(
        "  Streaming SR timing: setup=%.3f sec, CG=%.3f sec, iterations=%d, sec/iteration=%.6f.",
        setup_elapsed,
        solve_elapsed,
        iterations,
        solve_elapsed / max(iterations, 1),
    )
    warm_out = (
        SolverWarmStart.build(values=q_pc, layout_fingerprint=fingerprint)
        if fingerprint is not None and regime.startswith("wide")
        else None
    )
    return _PlanBSRScopeResult(
        q=_freeze_array(restore_wide_sr_coordinates(q_pc, prepared, coordinate_size=indices.size)),
        metric_diagonal=prepared.metric_diagonal,
        frozen_coordinate_groups=prepared.frozen_coordinate_groups,
        residual_norm=float(residual),
        num_iterations=int(iterations),
        breakdown=bool(breakdown),
        solver_regime=regime,
        # The tall streaming routes solve for the realified F of every rank.
        rhs_norm=(_device_sample_vector_norm(tuple(F_chunks), comm=comm) if regime.startswith("tall") else _vector_norm(b_pc)),
        prepared=prepared,
        warm_start_out=warm_out,
        sample_counts=sample_counts,
    )


def _solve_planb_sr_scope_device_streaming(
    batch: DeviceOptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    indices: np.ndarray,
    *,
    sync_scope: str,
    comm: MPI.Comm,
    epsilon: float,
    diagonal_floor: float,
    cg: bool,
    cg_max_iter: int,
    cg_tol: float,
    all_real: bool,
    finite_comm: MPI.Comm | None = None,
) -> _PlanBSRScopeResult:
    """Adapt the shared streaming SR implementation to a twist-local scope.

    ``comm`` is the twist's sampler communicator, a strict subset of the JAX
    world, so the world-mesh device collectives are never used here: several
    ranks reduce through the host-staged MPI CG over ``comm`` (issue #894).
    """
    if sync_scope != "sampler":
        raise ValueError("Twist-decomposed streaming SR is restricted to the sampler-owned scope.")
    return _solve_device_sr_scope_streaming(
        batch,
        coordinate_map,
        indices,
        sync_scope=sync_scope,
        comm=comm,
        epsilon=epsilon,
        diagonal_floor=diagonal_floor,
        cg=cg,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
        device_collectives=False,
        all_real=all_real,
        finite_comm=finite_comm,
    )


def _solve_planb_sr_scope_device(
    batch: DeviceOptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    indices: np.ndarray,
    *,
    sync_scope: str,
    comm: MPI.Comm,
    epsilon: float,
    diagonal_floor: float,
    cg: bool,
    cg_max_iter: int,
    cg_tol: float,
    normalization_comm: MPI.Comm | None = None,
    sampler_weight: float = 1.0,
    all_real: bool,
    finite_comm: MPI.Comm | None = None,
) -> _PlanBSRScopeResult:
    """Route only the large sampler-local CG scope through bounded streaming."""
    solve = _solve_planb_sr_scope_device_streaming if sync_scope == "sampler" and cg else _solve_planb_sr_scope_device_dense
    normalization_kwargs = (
        {"normalization_comm": normalization_comm, "sampler_weight": sampler_weight} if normalization_comm is not None else {}
    )
    return solve(
        batch,
        coordinate_map,
        indices,
        sync_scope=sync_scope,
        comm=comm,
        epsilon=epsilon,
        diagonal_floor=diagonal_floor,
        cg=cg,
        cg_max_iter=cg_max_iter,
        cg_tol=cg_tol,
        all_real=all_real,
        finite_comm=finite_comm,
        **normalization_kwargs,
    )


def _transform_device_derivatives_reduced_streaming(
    sources: Sequence[DeviceBlockDerivativeSource],
    coordinate_map: OptimizationCoordinateMap,
    right: npt.ArrayLike,
    *,
    sample_count: int,
    sample_batch_size: int = OPTIMIZER_DERIVATIVE_SAMPLE_BATCH,
    all_real: bool,
) -> jax.Array:
    """Contract a reduced basis without publishing full-batch coordinate columns."""
    if sample_count <= 0:
        raise ValueError("reduced device contraction requires a positive sample count.")
    if sample_batch_size <= 0:
        raise ValueError("reduced device contraction requires a positive sample batch size.")
    reduced_chunks = []
    for start in range(0, sample_count, sample_batch_size):
        stop = min(start + sample_batch_size, sample_count)
        source_chunk = _slice_device_derivative_sources(sources, start, stop)
        reduced = coordinate_map.transform_derivatives_device(source_chunk, right=right, all_real=all_real)
        reduced.block_until_ready()
        reduced_chunks.append(reduced)
    return jnp.concatenate(tuple(reduced_chunks), axis=0)


def _build_planb_lm_pencil_device(
    batch: DeviceOptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    indices: np.ndarray,
    direction: np.ndarray,
    selected_scope: np.ndarray | None,
    *,
    comm: MPI.Comm,
    normalization_comm: MPI.Comm | None = None,
    sampler_weight: float = 1.0,
    all_real: bool,
) -> tuple[LMRealPencil, np.ndarray | None]:
    """Build a scope-local LM pencil after device-side O/D contraction."""
    if batch.local_energy_derivative_sources is None:
        raise ValueError("Twist-decomposed LM/aSR requires exact local-energy derivative sources.")
    scope_size = int(indices.size)
    if selected_scope is None:
        basis_scope = np.eye(scope_size, dtype=np.float64)
        reduced_basis = None
    else:
        selected = np.asarray(selected_scope, dtype=np.int64)
        basis_scope = np.zeros((scope_size, selected.size + 1), dtype=np.float64)
        basis_scope[:, 0] = direction
        if selected.size:
            basis_scope[selected, np.arange(1, selected.size + 1)] = 1.0
        reduced_basis = _freeze_array(basis_scope)
    basis_full = np.zeros((coordinate_map.coordinate_size, basis_scope.shape[1]), dtype=basis_scope.dtype)
    basis_full[indices] = basis_scope
    O_reduced = _transform_device_derivatives_reduced_streaming(
        batch.log_derivative_sources,
        coordinate_map,
        basis_full,
        sample_count=int(batch.weights.size),
        all_real=all_real,
    )
    D_reduced = coordinate_map.transform_host_derivatives_reduced(
        batch.local_energy_derivative_sources, basis_full, all_real=all_real
    )
    weights, energies = batch.weights, batch.local_energies
    if normalization_comm is not None:
        O_reduced, weights, energies = _center_planb_twist_samples(
            O_reduced,
            weights,
            energies,
            sampler_comm=normalization_comm,
            world_comm=comm,
            sampler_weight=sampler_weight,
        )

    def reduce_sum(value: npt.ArrayLike) -> npt.ArrayLike:
        return comm.allreduce(value, op=MPI.SUM)

    pencil = _build_lm_real_pencil_from_coordinate_arrays(
        np.asarray(O_reduced),
        D_reduced,
        np.asarray(weights),
        np.asarray(energies),
        reduce_sum=reduce_sum,
    )
    return pencil, reduced_basis


def _warn_twist_sr_cg_not_converged(
    scope_reports: Sequence[tuple[str, str, int, float, float, bool]],
    *,
    sampler_comm: MPI.Comm,
    world_comm: MPI.Comm,
    cg_max_iter: int,
) -> None:
    """Log at most one CG non-convergence WARNING per twist ownership scope.

    ``scope_reports`` holds ``(scope, regime, iterations, residual, target,
    breakdown)`` of this rank's shared and twist-local CG solves.  The
    twist-local solves differ between samplers and only the world root logs,
    so one world ``allgather`` (collective: every rank calls this) gives every
    rank the same picture.  Each unconverged scope is reported once, with the
    number of samplers whose solve did not converge and the values of the
    worst one (largest ``residual / target``).
    """
    # Each rank carries 1/|sampler| so that every sampler is counted once.
    gathered = world_comm.allgather((1.0 / _comm_size(sampler_comm), tuple(scope_reports)))
    sampler_total = int(round(sum(share for share, _reports in gathered)))
    for scope in ("world", "sampler"):
        failed = []
        for share, reports in gathered:
            for report_scope, regime, iterations, residual, target, breakdown in reports:
                if report_scope == scope and not residual <= target:
                    ratio = residual / target if target > 0.0 else np.inf
                    failed.append((ratio, share, regime, iterations, residual, target, breakdown))
        if not failed:
            continue
        failed_samplers = int(round(sum(entry[1] for entry in failed)))
        _ratio, _share, regime, iterations, residual, target, breakdown = max(failed, key=lambda entry: entry[0])
        label = "shared" if scope == "world" else "twist-local"
        _warn_sr_cg_not_converged(
            f"twist-decomposed {label} scope, regime={regime}, {failed_samplers}/{sampler_total} sampler(s), worst shown",
            iterations=iterations,
            cg_max_iter=cg_max_iter,
            residual=residual,
            target=target,
            breakdown=breakdown,
        )


def compute_twist_decomposed_real_coordinate_optimization_target(
    batch: OptimizationBatch | DeviceOptimizationBatch,
    coordinate_map: OptimizationCoordinateMap,
    *,
    sampler_comm: MPI.Comm,
    world_comm: MPI.Comm,
    sampler_weight: float | None = None,
    epsilon: float,
    diagonal_floor: float,
    cg: bool,
    cg_max_iter: int,
    cg_tol: float,
    use_lm: bool,
    lm_subspace_dim: int,
    lm_condition_cutoff: float,
    energy_standard_error: float | None = None,
    world_energy_standard_error: float | None = None,
    sampler_energy_standard_error: float | None = None,
    asr_gamma_max: float = OPTIMIZER_DEFAULT_ASR_GAMMA_MAX,
    asr_fallback_gamma: float = OPTIMIZER_DEFAULT_ASR_FALLBACK_GAMMA,
    lm_reference_weight_min: float = OPTIMIZER_DEFAULT_LM_REFERENCE_WEIGHT_MIN,
) -> RealCoordinateOptimizationTarget:
    """Compute the optimization target for shared and twist-local parameter blocks."""
    epsilon, cg_tol, cg_max_iter, lm_subspace_dim = validate_sr_solver_options(
        epsilon=epsilon, cg_tol=cg_tol, cg_max_iter=cg_max_iter, lm_subspace_dim=lm_subspace_dim
    )
    if energy_standard_error is not None:
        if world_energy_standard_error is not None or sampler_energy_standard_error is not None:
            raise ValueError(
                "energy_standard_error cannot be combined with the scope-specific "
                "world_energy_standard_error / sampler_energy_standard_error values."
            )
        world_energy_standard_error = float(energy_standard_error)
        sampler_energy_standard_error = float(energy_standard_error)
    elif world_energy_standard_error is None or sampler_energy_standard_error is None:
        raise ValueError(
            "Twist-decomposed optimization requires both world_energy_standard_error and sampler_energy_standard_error."
        )
    world_energy_standard_error = float(world_energy_standard_error)
    sampler_energy_standard_error = float(sampler_energy_standard_error)
    # Twists differ per sampler: decide the real/complex route over the world,
    # once, and pass it down to every device scope as ``all_real``.
    batch, all_real = _collective_transform_route(batch, coordinate_map, world_comm)
    device_batch = isinstance(batch, DeviceOptimizationBatch)
    separate_twists = _comm_size(world_comm) > _comm_size(sampler_comm)
    normalized_weight = _normalize_planb_sampler_weight(sampler_weight, sampler_comm, world_comm) if separate_twists else 1.0
    normalization_kwargs = {"normalization_comm": sampler_comm, "sampler_weight": normalized_weight} if separate_twists else {}
    shared = _coordinate_scope_indices(coordinate_map, "world")
    local = _coordinate_scope_indices(coordinate_map, "sampler")
    if shared.size + local.size != coordinate_map.coordinate_size:
        raise ValueError("Shared and twist-local ownership must partition the coordinate axis.")
    if device_batch:
        force = np.zeros(coordinate_map.coordinate_size, dtype=np.float64)
        force_std = np.zeros(coordinate_map.coordinate_size, dtype=np.float64)
        for scope, indices, scope_comm in (
            ("world", shared, world_comm),
            ("sampler", local, sampler_comm),
        ):
            if indices.size == 0:
                continue
            scope_force, scope_std = _estimate_planb_scope_force_device(
                batch,
                coordinate_map,
                sync_scope=scope,
                comm=sampler_comm if scope == "world" and separate_twists else scope_comm,
                all_real=all_real,
            )
            if scope == "world" and separate_twists:
                scope_force = _allreduce_sum_array(world_comm, normalized_weight * scope_force / _comm_size(sampler_comm))
                scope_std = np.sqrt(
                    _allreduce_sum_array(world_comm, np.square(normalized_weight * scope_std) / _comm_size(sampler_comm))
                )
            force[indices] = scope_force
            force_std[indices] = scope_std
    else:
        sampler_force, sampler_std = _estimate_coordinate_force_jackknife(
            batch,
            coordinate_map,
            comm=sampler_comm,
        )
        force = np.array(sampler_force, copy=True)
        force_std = np.array(sampler_std, copy=True)
        if shared.size and separate_twists:
            force[shared] = _allreduce_sum_array(
                world_comm, normalized_weight * sampler_force[shared] / _comm_size(sampler_comm)
            )
            force_std[shared] = np.sqrt(
                _allreduce_sum_array(world_comm, np.square(normalized_weight * sampler_std[shared]) / _comm_size(sampler_comm))
            )
    tolerance = get_optimizer_roundoff_tolerance("force_standard_error", force_std.dtype)
    finite = np.isfinite(force_std) & (np.abs(force_std) > tolerance)
    signal_to_noise = np.zeros(force.shape, dtype=np.float64)
    signal_to_noise[finite] = np.abs(force[finite]) / force_std[finite]
    sr_q = np.zeros(coordinate_map.coordinate_size, dtype=np.float64)
    metric_diagonal = np.zeros(coordinate_map.coordinate_size, dtype=np.float64)
    frozen_groups: list[tuple[int, ...]] = []
    residual_norm = 0.0
    residual_tolerance = 0.0
    num_iterations = 0
    cg_breakdown = False
    solver_regimes: list[str] = []
    scope_cg_reports: list[tuple[str, str, int, float, float, bool]] = []
    for scope, indices, scope_comm in (("world", shared, world_comm), ("sampler", local, sampler_comm)):
        # The shared moments are world-reduced already; a twist-local scope
        # must agree over the world so that every twist raises together.
        finite_comm = world_comm if scope == "sampler" and separate_twists else None
        if device_batch:
            scope_result = _solve_planb_sr_scope_device(
                batch,
                coordinate_map,
                indices,
                sync_scope=scope,
                comm=scope_comm,
                epsilon=epsilon,
                diagonal_floor=diagonal_floor,
                cg=cg,
                cg_max_iter=cg_max_iter,
                cg_tol=cg_tol,
                all_real=all_real,
                finite_comm=finite_comm,
                **(normalization_kwargs if scope == "world" else {}),
            )
        else:
            scope_result = _solve_planb_sr_scope(
                batch,
                coordinate_map,
                indices,
                comm=scope_comm,
                epsilon=epsilon,
                diagonal_floor=diagonal_floor,
                cg=cg,
                cg_max_iter=cg_max_iter,
                cg_tol=cg_tol,
                finite_comm=finite_comm,
                **(normalization_kwargs if scope == "world" else {}),
            )
        sr_q[indices] = scope_result.q
        metric_diagonal[indices] = scope_result.metric_diagonal
        frozen_groups.extend(
            tuple(int(indices[local_index]) for local_index in group) for group in scope_result.frozen_coordinate_groups
        )
        # Report the scope farthest from its own relative CG threshold
        # cg_tol * ||b_scope|| (issue #859), so the aggregate is converged
        # exactly when every scope is.
        scope_tolerance = float(cg_tol) * float(scope_result.rhs_norm)
        if not solver_regimes or scope_result.residual_norm - scope_tolerance > residual_norm - residual_tolerance:
            residual_norm = scope_result.residual_norm
            residual_tolerance = scope_tolerance
        num_iterations = max(num_iterations, scope_result.num_iterations)
        cg_breakdown = cg_breakdown or scope_result.breakdown
        solver_regimes.append(f"{scope}:{scope_result.solver_regime}")
        scope_cg_reports.append(
            (
                scope,
                scope_result.solver_regime,
                scope_result.num_iterations,
                scope_result.residual_norm,
                scope_tolerance,
                scope_result.breakdown,
            )
        )
    if cg:
        _warn_twist_sr_cg_not_converged(
            scope_cg_reports, sampler_comm=sampler_comm, world_comm=world_comm, cg_max_iter=cg_max_iter
        )
    q = np.array(sr_q, copy=True)
    # The gamma of every scope whose SR direction was scaled (aSR, or the aSR
    # fallback of a rejected LM candidate), as ``accepted_scale = gamma`` on
    # the single-k path; an accepted LM scope contributes none.
    scope_scales: list[float] = []
    predicted_energies: list[float] = []
    reference_weights: list[float] = []
    rejection_reasons: list[str] = []
    linearized_norm_sq = 0.0
    # Only the host LM/aSR pencils below need O and D; plain SR does not.
    O_all = None if device_batch or not use_lm else coordinate_map.transform_derivatives(batch.log_derivative_sources)
    D_all = (
        None
        if device_batch or not use_lm or batch.local_energy_derivative_sources is None
        else coordinate_map.transform_derivatives(batch.local_energy_derivative_sources)
    )
    selected_by_scope: dict[str, np.ndarray | None] = {}
    if use_lm and lm_subspace_dim > 0:
        for scope, indices, scope_comm in (
            ("world", shared, world_comm),
            ("sampler", local, sampler_comm),
        ):
            if indices.size:
                selected_by_scope[scope] = _select_planb_lm_scope_coordinates(
                    signal_to_noise,
                    coordinate_map,
                    indices,
                    coordinate_budget=lm_subspace_dim,
                    comm=scope_comm,
                )
    selected_set = set(range(coordinate_map.coordinate_size))
    if use_lm and lm_subspace_dim > 0:
        selected_set.clear()
        for scope, indices in (("world", shared), ("sampler", local)):
            selected_scope = selected_by_scope.get(scope)
            if selected_scope is None:
                selected_set.update(int(index) for index in indices)
            else:
                selected_set.update(int(indices[local_index]) for local_index in selected_scope)
    selected_groups = tuple(
        group for group in coordinate_map.coordinate_groups if all(index in selected_set for index in group)
    )
    if use_lm:
        if batch.local_energy_derivative_sources is None:
            raise ValueError("Twist-decomposed LM/aSR requires exact local-energy derivative sources.")
        for scope, indices, scope_comm in (
            ("world", shared, world_comm),
            ("sampler", local, sampler_comm),
        ):
            if indices.size == 0:
                continue

            def reduce(value: npt.ArrayLike, scope_comm: MPI.Comm = scope_comm) -> npt.ArrayLike:
                return scope_comm.allreduce(value, op=MPI.SUM)

            direction = sr_q[indices]
            if lm_subspace_dim == 0 and not np.any(direction):
                # The one-dimensional aSR trial is just the reference state.
                # Do not contract a zero basis (or manufacture a direction).
                continue
            if device_batch:
                if lm_subspace_dim == 0:
                    selected_scope = np.empty(0, dtype=np.int64)
                else:
                    selected_scope = selected_by_scope.get(scope)
                pencil, basis = _build_planb_lm_pencil_device(
                    batch,
                    coordinate_map,
                    indices,
                    direction,
                    selected_scope,
                    comm=scope_comm,
                    all_real=all_real,
                    **(normalization_kwargs if scope == "world" else {}),
                )
            else:
                assert O_all is not None and D_all is not None
                O_scope = O_all[:, indices]
                D_scope = D_all[:, indices]
                scope_weights, scope_energies = batch.weights, batch.local_energies
                if scope == "world" and separate_twists:
                    O_scope, scope_weights, scope_energies = _center_planb_twist_samples(
                        O_scope,
                        scope_weights,
                        scope_energies,
                        sampler_comm=sampler_comm,
                        world_comm=world_comm,
                        sampler_weight=normalized_weight,
                    )
                pencil = _build_lm_real_pencil_from_coordinate_arrays(
                    O_scope,
                    D_scope,
                    scope_weights,
                    scope_energies,
                    reduce_sum=reduce,
                )
                basis = None
            if lm_subspace_dim == 0:
                contraction_direction = np.ones(1) if device_batch else direction
                coefficients = asr_coefficients_from_pencil(pencil.H, pencil.S, contraction_direction)
                asr = select_asr_gamma_bounded(
                    *coefficients,
                    gamma_max=asr_gamma_max,
                    fallback_gamma=asr_fallback_gamma,
                )
                q[indices] = asr.gamma * direction
                scope_scales.append(asr.gamma)
                predicted_energies.append(asr.predicted_energy)
                linearized_norm_sq += max(float(asr.gamma**2 * coefficients[-1]), 0.0)
                if asr.used_fallback:
                    rejection_reasons.append("aSR model was invalid in one shared/local scope")
                continue
            lm_pencil = pencil
            selected_scope = selected_by_scope.get(scope)
            if not device_batch and selected_scope is not None:
                lm_pencil, basis = reduce_lm_real_pencil(pencil, direction, selected_scope)
            try:
                lm = solve_lm_real_pencil(lm_pencil, condition_cutoff=lm_condition_cutoff)
            except _kernel.LMNoRealRootError:
                lm = None
            scope_error = world_energy_standard_error if scope == "world" else sampler_energy_standard_error
            scope_reasons = []
            if lm is None:
                scope_reasons.append("no numerically real LM root")
            else:
                reference_weights.append(lm.reference_weight)
                if lm.predicted_energy > pencil.H_0 + 3.0 * scope_error:
                    scope_reasons.append("linear-method energy gate failed")
                if lm.reference_weight < lm_reference_weight_min:
                    scope_reasons.append(f"reference weight {lm.reference_weight:.8f} is below {lm_reference_weight_min:.8f}")
            if scope_reasons:
                asr, coefficients = _select_lm_rejection_asr_fallback(
                    lm_pencil,
                    basis,
                    direction,
                    gamma_max=asr_gamma_max,
                    fallback_gamma=asr_fallback_gamma,
                )
                q[indices] = asr.gamma * direction
                scope_scales.append(asr.gamma)
                fallback_reason = (
                    f"aSR model invalid; configured plain-SR gamma={asr.gamma:.8f} used"
                    if asr.used_fallback
                    else f"aSR fallback gamma={asr.gamma:.8f}"
                )
                rejection_reasons.append("shared/local scope: " + "; ".join((*scope_reasons, fallback_reason)))
                linearized_norm_sq += max(float(asr.gamma**2 * coefficients[-1]), 0.0)
                predicted_energies.append(asr.predicted_energy)
            else:
                q[indices] = lm.q if basis is None else basis @ lm.q
                reduced_q = np.asarray(lm.q)
                linearized_norm_sq += max(float(reduced_q @ lm_pencil.S[1:, 1:] @ reduced_q), 0.0)
                predicted_energies.append(lm.predicted_energy)
    # One reported scale: the scope gamma of smallest magnitude, with its sign
    # (a gamma may be negative), i.e. the most conservative scaled step.  It
    # equals the gamma itself when one scope, or all scopes alike, was scaled.
    accepted_scale = float(min(scope_scales, key=abs)) if scope_scales else 1.0
    if not np.all(np.isfinite(q)):
        raise FloatingPointError("Twist-decomposed optimizer produced a non-finite real-coordinate update.")
    return RealCoordinateOptimizationTarget(
        q=_freeze_array(q),
        sr_q=_freeze_array(sr_q),
        coordinate_force=_freeze_array(force),
        coordinate_force_std=_freeze_array(force_std),
        signal_to_noise=_freeze_array(signal_to_noise),
        sr_regime="twist-decomposed[" + ",".join(solver_regimes) + "]",
        accepted_scale=accepted_scale,
        predicted_energy=min(predicted_energies) if predicted_energies else None,
        lm_reference_weight=min(reference_weights) if reference_weights else None,
        warm_start_out=None,
        converged=not cg or residual_norm <= residual_tolerance,
        num_iterations=num_iterations,
        residual_norm=residual_norm,
        residual_tolerance=residual_tolerance,
        metric_diagonal=_freeze_array(metric_diagonal),
        frozen_coordinate_groups=tuple(frozen_groups),
        selected_coordinate_groups=selected_groups,
        rejection_reason="; ".join(rejection_reasons) if rejection_reasons else None,
        linearized_wavefunction_norm=(np.sqrt(linearized_norm_sq) if use_lm else None),
        cg_breakdown=cg_breakdown,
    )


class OptimizationSampler(Protocol):
    """Normalized sampling/estimator contract required by ``run_optimize``.

    MCMC and future GFMC drivers implement this protocol structurally.  The
    optimizer intentionally does not know their native method names or raw
    storage layouts.
    """

    comm: Any
    world_comm: Any
    rank: int
    size: int

    @property
    def hamiltonian_data(self) -> Hamiltonian_data: ...

    @hamiltonian_data.setter
    def hamiltonian_data(self, value: Hamiltonian_data) -> None: ...

    @property
    def optimization_step_index(self) -> int: ...

    @property
    def wavefunction_state_token(self) -> tuple[str, int]: ...

    def derivative_capabilities(self) -> tuple[bool, bool]: ...

    def configure_optimization(self, *, method: str, hyperparameters: dict[str, Any]) -> None: ...

    def select_parameters(self, flags: dict[str, bool]) -> None: ...

    def sample(self, *, num_steps: int, max_time: int) -> None: ...

    def estimate_energy(self, *, num_warmup_steps: int, num_bin_blocks: int) -> EnergyEstimate: ...

    def validate_wavefunction_candidate(self, candidate: Any) -> None: ...

    def variational_blocks(self, parameters: ParameterSelection) -> list[Any]: ...

    def optimization_batch(
        self,
        *,
        num_warmup_steps: int,
        num_bin_blocks: int,
        blocks: list[Any],
        need_local_energy_derivatives: bool,
    ) -> OptimizationBatch | DeviceOptimizationBatch: ...

    def advance_optimization_steps(self, count: int) -> None: ...

    def advance_wavefunction_revision(self) -> None: ...


def _comm_size(comm: Any) -> int:
    """Return an MPI-like communicator size without assuming mpi4py."""
    get_size = getattr(comm, "Get_size", None)
    if get_size is not None:
        return int(get_size())
    return int(getattr(comm, "size", 1))


def _require_world_device_mesh(comm: Any, layouts: Sequence[tuple[int, ...]]) -> None:
    """Raise on every rank unless the world-mesh device SR kernels fit ``comm``.

    Those kernels (``_kernel._get_sr_mesh``) reduce with JAX collectives over
    every JAX process and assemble each process's local sample block into one
    global array without a global shape. They are exact only when ``comm``
    spans every JAX process and every rank has the same positive layout: its
    sample count and, on the streaming route, the chunk count that selects the
    compiled kernel. ``layouts`` is the per-rank layout gathered over ``comm``
    (sample count first), so every rank reaches the same verdict.
    """
    if _comm_size(comm) != jax.process_count():
        raise ValueError(
            f"The device SR collectives reduce over all {jax.process_count()} JAX processes, but the communicator has "
            f"{_comm_size(comm)} ranks; a sampler subcommunicator must use the host-staged MPI route."
        )
    if len(set(layouts)) != 1 or layouts[0][0] <= 0:
        raise ValueError(f"The device SR collectives require one positive per-rank layout on every rank; got {tuple(layouts)}.")


def _collective_step_acceptance(local_ok: bool, twist_decomposed: bool, world_comm: Any) -> bool:
    """Return true only when every twist rank can continue the step."""
    if not twist_decomposed:
        return bool(local_ok)
    return bool(world_comm.allreduce(int(bool(local_ok)), op=MPI.MIN))


def _collective_candidate_acceptance(local_ok: bool, sampler_comm: Any, twist_decomposed: bool, world_comm: Any) -> bool:
    """Reject a candidate on every sampler rank before any rank publishes it."""
    sampler_ok = bool(sampler_comm.allreduce(int(bool(local_ok)), op=MPI.MIN))
    return _collective_step_acceptance(sampler_ok, twist_decomposed, world_comm)


def run_optimize(
    sampler: OptimizationSampler,
    config: OptimizationConfig,
) -> OptimizationStopReason | None:
    """Optimize a wavefunction using samples supplied through a common contract.

    This function is deliberately agnostic to whether samples came from MCMC,
    GFMC-t, GFMC-n, or another driver.  Sampler-specific method names and raw
    storage layouts are normalized by each class's ``OptimizationSampler``
    implementation.

    Args:
        sampler: Normalized provider of samples and estimators.
        config: Optimization controls independent of the sampler algorithm.

    Returns:
        ``"max_time"`` or ``"stop_flag"`` when the loop was left early because
        ``config.max_time`` was exceeded or ``stop = true`` was found in
        ``external_control_opt.toml``; ``None`` otherwise. The value is the
        same on every rank of ``sampler.world_comm`` (both decisions are taken
        on its root and broadcast). The twist-averaged driver uses this
        reason to stop after the current chunk's checkpoint.

    Notes:
        - Persists the SR optimizer method + hyperparameters across calls.
        - Owns ``external_control_opt.toml`` for direct invocations. Chunked
          callers can manage its lifetime across invocations.
        - Updates :class:`Hamiltonian_data` in-place and increments the optimization counter.

        Flag pair ``opt_geminal`` / ``fix_geminal_rank`` controls the
        geminal optimisation:

        * ``parameters.opt_geminal=True``: enable geminal opt.
        * ``config.fix_geminal_rank=int``: rank-truncated MO opt. When the
          current geminal is ``AGP_data`` (loaded from TREXIO / QE /
          full AO basis), it is auto-converted to ``AGPn_data`` of
          rank ``int`` at the top of ``run_optimize`` via
          :func:`jqmc.determinant.to_AGPn`. The rest of the opt loop
          then runs uniformly through ``get_variational_blocks`` /
          ``apply_block_update`` -- the S-Grassmannian polar retraction
          built into :meth:`AGPn_data.apply_block_update` keeps
          ``C^H S C = I`` invariant. When the current geminal is
          already ``AGPn_data`` / ``SD_data`` (already MO-basis), the
          rank is structural; mismatches are warned (AGPn) or raised
          (SD).
        * ``config.fix_geminal_rank=None`` (default): full-rank AO-ambient
          optimisation on ``AGP_data``.
    """
    parameters = config.parameters
    num_sampling_steps = config.num_sampling_steps
    num_opt_steps = config.num_opt_steps
    wavefunction_dump_frequency = config.wavefunction_dump_frequency
    # ``dump_wavefunction=False`` suppresses every ``hamiltonian_data_opt_step_*.h5``
    # write of this invocation.  The twist-averaged driver sets it so the
    # rank-0-twist-only file is never produced; it dumps the complete
    # multi-twist combined artifact itself, at the same cadence.
    dump_wavefunction = bool(config.dump_wavefunction)
    max_time = config.max_time
    num_warmup_steps = config.num_warmup_steps
    num_bin_blocks = config.num_bin_blocks
    if not isinstance(config.use_device_optimizer, bool):
        raise TypeError(f"use_device_optimizer must be a bool; got {config.use_device_optimizer!r}.")
    # The device reduction route follows from the communicators: a sampler
    # that spans the world reduces with the world-mesh device kernels, a
    # twist sampler (a strict subcommunicator) with the host-staged MPI CG.
    use_device_optimizer = config.use_device_optimizer
    fix_geminal_rank = config.fix_geminal_rank
    twist_decomposed = _comm_size(sampler.comm) < _comm_size(sampler.world_comm)

    if fix_geminal_rank is not None and fix_geminal_rank < 1:
        raise ValueError(f"fix_geminal_rank must be a positive integer; got {fix_geminal_rank}.")

    # ``fix_geminal_rank`` + AGP_data -> auto-promote to AGPn_data at
    # the requested rank. The whole optimisation loop then runs
    # class-uniformly via ``get_variational_blocks`` /
    # ``apply_block_update``. When the
    # geminal is already AGPn / SD, the rank is structural: mismatches
    # are warned (AGPn) or raised (SD) since silent re-truncation
    # mid-pipeline would discard variational state.
    if fix_geminal_rank is not None and parameters.opt_geminal:
        from .determinant import AGP_data, AGPn_data, SD_data, to_AGPn

        current_geminal = sampler.hamiltonian_data.wavefunction_data.geminal_data
        if isinstance(current_geminal, AGP_data):
            if any([parameters.opt_lambda_basis_exp, parameters.opt_lambda_basis_coeff]):
                raise ValueError(
                    "fix_geminal_rank cannot be combined with "
                    "opt_lambda_basis_exp / opt_lambda_basis_coeff. "
                    "Changing the AO basis invalidates the overlap "
                    "matrix that the AGP -> AGPn SVD relies on."
                )
            new_geminal = to_AGPn(current_geminal, n=int(fix_geminal_rank))
            logger.info(
                "fix_geminal_rank=%d: converted AGP_data -> AGPn_data "
                "(num_mo=%d, time_reversal=%s) for class-uniform MO "
                "opt path.",
                int(fix_geminal_rank),
                int(new_geminal.num_mo),
                new_geminal.time_reversal,
            )
            wf_data = sampler.hamiltonian_data.wavefunction_data
            new_wf_data = type(wf_data)(
                jastrow_data=wf_data.jastrow_data,
                geminal_data=new_geminal,
            )
            sampler.hamiltonian_data = Hamiltonian_data(
                structure_data=sampler.hamiltonian_data.structure_data,
                wavefunction_data=new_wf_data,
                coulomb_potential_data=sampler.hamiltonian_data.coulomb_potential_data,
            )
        elif isinstance(current_geminal, AGPn_data):
            if int(fix_geminal_rank) != int(current_geminal.num_mo):
                logger.warning(
                    "fix_geminal_rank=%d differs from current AGPn_data "
                    "rank=%d; ignoring (rank is structural on AGPn / SD; "
                    "call to_AGPn explicitly upstream to re-truncate).",
                    int(fix_geminal_rank),
                    int(current_geminal.num_mo),
                )
        elif isinstance(current_geminal, SD_data):
            if int(fix_geminal_rank) != int(current_geminal.num_electron_dn):
                raise ValueError(
                    f"fix_geminal_rank={int(fix_geminal_rank)} does not "
                    f"match SD_data.num_electron_dn="
                    f"{int(current_geminal.num_electron_dn)}. SD has "
                    f"structural rank = N_occ; pass an AGPn_data instead "
                    f"if a different rank is needed."
                )

    optimizer_kwargs = dict(config.optimizer_kwargs)
    optimizer_mode = optimizer_kwargs.pop("method", "sr")
    if not isinstance(optimizer_mode, str):
        raise TypeError("optimizer_kwargs['method'] must be a string if provided.")
    optimizer_mode = optimizer_mode.lower()
    if optimizer_mode != "sr":
        raise ValueError(
            f"Unsupported optimizer '{optimizer_mode}'. SR (natural gradient) / linear method is the "
            "only supported wavefunction optimizer; set optimizer_kwargs['method'] = 'sr'."
        )

    # SR parameters (including LM options)
    delta = float(optimizer_kwargs.get("delta", 1.0e-3))
    sr_cg_flag = bool(optimizer_kwargs.get("cg_flag", True))
    use_lm = bool(optimizer_kwargs.get("use_lm", False))
    # Rejected here, before any sampling, for the Python API and the CLI alike.
    epsilon, sr_cg_tol, sr_cg_max_iter, lm_subspace_dim = validate_sr_solver_options(
        epsilon=optimizer_kwargs.get("epsilon", 1.0e-3),
        cg_tol=optimizer_kwargs.get("cg_tol", OPTIMIZER_DEFAULT_CG_TOL),
        cg_max_iter=optimizer_kwargs.get("cg_max_iter", OPTIMIZER_DEFAULT_CG_MAX_ITER),
        lm_subspace_dim=optimizer_kwargs.get("lm_subspace_dim", 0),
    )
    lm_cond = _kernel.validate_lm_condition_cutoff(optimizer_kwargs.get("lm_cond", OPTIMIZER_DEFAULT_LM_CONDITION_CUTOFF))
    lm_reference_weight_min = float(optimizer_kwargs.get("lm_reference_weight_min", OPTIMIZER_DEFAULT_LM_REFERENCE_WEIGHT_MIN))
    asr_gamma_max = float(optimizer_kwargs.get("asr_gamma_max", OPTIMIZER_DEFAULT_ASR_GAMMA_MAX))
    asr_fallback_gamma = float(optimizer_kwargs.get("asr_fallback_gamma", OPTIMIZER_DEFAULT_ASR_FALLBACK_GAMMA))

    # use_lm requires derivative computations
    if use_lm:
        has_log_derivative, has_local_energy_derivative = sampler.derivative_capabilities()
        if not has_log_derivative:
            raise RuntimeError("use_lm requires compute_log_WF_param_deriv=True.")
        if not has_local_energy_derivative:
            raise RuntimeError("use_lm requires comput_e_L_param_deriv=True.")

    _check_mpi4py_jax_distribution_consistency()
    if twist_decomposed and config.twist_index is None:
        raise ValueError("A twist-decomposed optimization needs config.twist_index (set by the twist-averaged driver).")

    optimizer_hparams: dict[str, float | int | str] = {
        "method": optimizer_mode,
        "delta": delta,
        "epsilon": epsilon,
        "cg_flag": sr_cg_flag,
        "cg_max_iter": sr_cg_max_iter,
        "cg_tol": sr_cg_tol,
        "use_lm": use_lm,
        "lm_subspace_dim": lm_subspace_dim,
        "lm_cond": lm_cond,
        "lm_reference_weight_min": lm_reference_weight_min,
        "asr_gamma_max": asr_gamma_max,
        "asr_fallback_gamma": asr_fallback_gamma,
        "use_device_optimizer": use_device_optimizer,
    }

    sampler.configure_optimization(
        method=optimizer_mode,
        hyperparameters=optimizer_hparams,
    )

    # optimizer info
    logger.info(f"The chosen optimizer is '{optimizer_mode}'.")
    logger.info("  The homemade 'SR (aka natural gradient)' optimizer is used for wavefunction optimization.")
    if use_lm and lm_subspace_dim != 0:
        logger.info(
            "  use_lm=True, lm_subspace_dim=%d: Linear Method with SR collective variable.",
            lm_subspace_dim,
        )
        if twist_decomposed and lm_subspace_dim > 0:
            logger.debug(
                "  Twist-decomposed LM applies independent coordinate budgets: world/shared=%d and sampler/local=%d per twist.",
                lm_subspace_dim,
                lm_subspace_dim,
            )
    elif use_lm:
        logger.info("  use_lm=True, lm_subspace_dim=0: accelerated SR (aSR) gamma scaling.")
    # Name of the method that runs every step, for the update and timer logs.
    _log_label = "LM" if (use_lm and lm_subspace_dim != 0) else "aSR" if use_lm else "SR"
    logger.info("  Hyperparameters: %s", ", ".join(f"{k}={v}" for k, v in sorted(optimizer_hparams.items())))
    logger.info("")

    sampler.select_parameters(parameters.gradient_flag_kwargs())

    if config._manage_external_control:
        _init_external_control_file(_OPTIMIZER_CONTROL_FILE, sampler.world_comm)
    sampler.comm.Barrier()

    # timer
    optimization_total_start = time.perf_counter()

    # timer_counter
    timer_opt_total = 0.0
    timer_sampling = 0.0
    timer_get_E = 0.0
    timer_get_gF = 0.0
    timer_optimizer = 0.0
    timer_param_update = 0.0
    timer_MPI_barrier = 0.0
    num_opt_done = 0
    stop_reason: OptimizationStopReason | None = None

    sr_warm_start: SolverWarmStart | None = None

    _log_memory_usage("optimize_pre_loop", comm=sampler.comm)

    i_opt_offset = sampler.optimization_step_index
    # Totals over every twist, gathered once per call: the coordinate map is
    # rebuilt each step from the same wavefunction structure, so its block
    # sizes cannot change within a call (a resumed or chained call gathers
    # them again from its own first step).
    all_twist_counts: tuple[int, int] | None = None

    def compute_step_in_batch_scope(
        blocks: list[Any],
        coordinate_map: OptimizationCoordinateMap,
        world_energy_standard_error: float,
        sampler_energy_standard_error: float,
        warm_start: SolverWarmStart | None,
    ) -> tuple[RealCoordinateOptimizationTarget, OptimizationStepResult, str, float, _TwistStepDiagnostics | None]:
        """Build and consume one sample batch inside a bounded scope.

        The fourth return value is the wall time spent building the batch
        (the per-sample derivative data behind the generalized forces); the
        caller books it as the gF time and the remainder as the optimizer time.
        The last one is the diagnostics merged over every twist (world root of
        a twist-decomposed step only, else ``None``).
        """
        batch_start = time.perf_counter()
        batch = sampler.optimization_batch(
            num_warmup_steps=num_warmup_steps,
            num_bin_blocks=num_bin_blocks,
            blocks=blocks,
            need_local_energy_derivatives=use_lm,
        )
        batch_elapsed = time.perf_counter() - batch_start
        if batch.state_token != coordinate_map.state_token:
            raise ValueError("OptimizationBatch state_token does not match the current coordinate map.")
        optimizer_backend = (
            "CPU/MPI"
            if not isinstance(batch, DeviceOptimizationBatch)
            else "device-local/MPI"
            if twist_decomposed
            else "device-collective"
        )
        _log_optimization_batch_layout(batch, coordinate_map, sampler.size)
        if twist_decomposed:
            target = compute_twist_decomposed_real_coordinate_optimization_target(
                batch,
                coordinate_map,
                sampler_comm=sampler.comm,
                world_comm=sampler.world_comm,
                sampler_weight=config.sampler_weight,
                epsilon=epsilon,
                diagonal_floor=min_S_diag_abs,
                cg=sr_cg_flag,
                cg_max_iter=sr_cg_max_iter,
                cg_tol=sr_cg_tol,
                use_lm=use_lm,
                lm_subspace_dim=lm_subspace_dim,
                lm_condition_cutoff=lm_cond,
                world_energy_standard_error=world_energy_standard_error,
                sampler_energy_standard_error=sampler_energy_standard_error,
                lm_reference_weight_min=lm_reference_weight_min,
                asr_gamma_max=asr_gamma_max,
                asr_fallback_gamma=asr_fallback_gamma,
            )
        else:
            target = compute_real_coordinate_optimization_target(
                batch,
                coordinate_map,
                comm=sampler.comm,
                epsilon=epsilon,
                diagonal_floor=min_S_diag_abs,
                cg=sr_cg_flag,
                cg_max_iter=sr_cg_max_iter,
                cg_tol=sr_cg_tol,
                warm_start=warm_start,
                use_lm=use_lm,
                lm_subspace_dim=lm_subspace_dim,
                lm_condition_cutoff=lm_cond,
                energy_standard_error=world_energy_standard_error,
                lm_reference_weight_min=lm_reference_weight_min,
                asr_gamma_max=asr_gamma_max,
                asr_fallback_gamma=asr_fallback_gamma,
            )
        step_error: Exception | None = None
        try:
            step_result = finalize_optimization_step(
                target,
                coordinate_map,
                learning_rate=delta,
                energy_standard_error=world_energy_standard_error,
            )
        except Exception as exc:  # re-raised below, after the world gather
            if not twist_decomposed:
                raise
            step_result, step_error = None, exc
        twist_diagnostics = None
        if twist_decomposed:
            # Every world rank has just left the world-collective target above
            # (the only earlier exit, a non-finite energy, is agreed over the
            # world before the batch is built) and the local, collective-free
            # finalize, whose failure is held until after the gather; every
            # later rejection comes after this point. So all ranks enter this
            # gather once per step.
            twist_diagnostics = _gather_twist_step_diagnostics(
                target,
                step_result,
                coordinate_map,
                sampler_comm=sampler.comm,
                world_comm=sampler.world_comm,
                twist_index=int(config.twist_index),
            )
        if step_error is not None:
            raise step_error
        assert step_result is not None
        _log_optimization_diagnostics(
            target,
            step_result,
            coordinate_map,
            use_lm=use_lm,
            lm_subspace_dim=lm_subspace_dim,
            cg=sr_cg_flag,
            twist_diagnostics=twist_diagnostics,
        )
        return target, step_result, optimizer_backend, batch_elapsed, twist_diagnostics

    # Main optimization loop.
    for i_opt in range(num_opt_steps):
        logger.info("=" * num_sep_line)
        logger.info(f"Optimization step = {i_opt + 1 + i_opt_offset}/{num_opt_steps + i_opt_offset}.")
        logger.info("=" * num_sep_line)

        logger.info(f"Sampling steps this iteration = {num_sampling_steps}.")
        logger.info(f"Warmup steps = {num_warmup_steps}.")
        logger.info(f"Bin blocks = {num_bin_blocks}.")
        logger.info("")

        # Generate the next sampler batch. The batch is never cut short by
        # max_time: a truncated batch has too few samples for the energy and
        # the generalized forces of this step. max_time is enforced between
        # optimization steps below, as documented for control.max_time: a
        # step predicted to end after max_time is not started.
        start = time.perf_counter()
        sampler.sample(num_steps=num_sampling_steps, max_time=float("inf"))
        end = time.perf_counter()
        timer_sampling += end - start

        # get E
        start = time.perf_counter()
        sampler_energy_estimate = sampler.estimate_energy(
            num_warmup_steps=num_warmup_steps,
            num_bin_blocks=num_bin_blocks,
        )
        if config.energy_aggregator is None:
            energy_estimate = sampler_energy_estimate
        else:
            energy_estimate = config.energy_aggregator(sampler_energy_estimate)
            if not isinstance(energy_estimate, EnergyEstimate):
                raise TypeError("energy_aggregator must return an EnergyEstimate.")
        E = energy_estimate.mean
        E_std = energy_estimate.standard_error
        end = time.perf_counter()
        timer_get_E += end - start
        logger.info("Total Energy before update of wavefunction.")
        logger.info("-" * num_sep_line)
        logger.info(f"E = {E:.5f} +- {E_std:.5f} Ha")
        logger.info("-" * num_sep_line)
        logger.info("")

        # Abort optimization if energy is NaN/Inf -- the wavefunction is
        # corrupted and further updates would be meaningless.
        energy_is_finite = bool(np.isfinite(E))
        if not _collective_step_acceptance(energy_is_finite, twist_decomposed, sampler.world_comm):
            if twist_decomposed:
                logger.error(
                    "Non-finite twist-averaged energy detected (E_TA=%s). Rejecting the world-coupled optimization step.",
                    E,
                )
            else:
                logger.error("Non-finite energy detected (E=%s). Rejecting the optimization step.", E)
            break

        # collect variational parameter blocks from the wavefunction data
        blocks = sampler.variational_blocks(parameters)
        coordinate_map = OptimizationCoordinateMap.from_blocks(blocks)

        # flatten index mapping for the blocks
        offsets = []
        start = 0
        for block in blocks:
            offsets.append((block, start, start + block.size))
            start += block.size
        parameter_counts = _variational_parameter_counts(coordinate_map)
        if twist_decomposed and i_opt == 0:
            # World collective on the first step of the call; every rank is here
            # (the non-finite-energy exit above is agreed over the world).
            all_twist_counts = _gather_all_twist_counts(
                coordinate_map, sampler_comm=sampler.comm, world_comm=sampler.world_comm
            )

        logger.info(f"Number of variational parameters = {parameter_counts['values']}.")
        if all_twist_counts is not None:
            logger.info(f"Number of variational parameters (all twists) = {all_twist_counts[0]}.")
        logger.info(f"Number of real optimization coordinates = {parameter_counts['coordinates']}.")
        if all_twist_counts is not None:
            logger.info(f"Number of real optimization coordinates (all twists) = {all_twist_counts[1]}.")

        #############################
        # real-coordinate optimization step
        #############################
        start = time.perf_counter()
        target, step_result, optimizer_backend, batch_elapsed, twist_diagnostics = compute_step_in_batch_scope(
            blocks,
            coordinate_map,
            E_std,
            sampler_energy_estimate.standard_error,
            sr_warm_start,
        )
        sr_warm_start = target.warm_start_out
        f = target.coordinate_force
        f_std = target.coordinate_force_std
        signal_to_noise_f = target.signal_to_noise
        theta = np.concatenate([np.asarray(update.delta).reshape(-1) for update in step_result.updates])
        end = time.perf_counter()
        # The batch build and the force/solver stage are fused in one call;
        # book each interval exactly once so that misc stays non-negative.
        timer_get_gF += batch_elapsed
        timer_optimizer += (end - start) - batch_elapsed

        logger.info("-" * num_sep_line)
        if twist_diagnostics is None:
            f_argmax = int(np.argmax(np.abs(f)))
            logger.info("Max generalized force = %s Ha/a.u.", _format_value_and_error(f[f_argmax], f_std[f_argmax]))
            logger.info("Max generalized force signal-to-noise = %.3f.", np.max(signal_to_noise_f))
        else:
            # Maxima over the shared coordinates and every twist's local ones.
            logger.info(
                "Max generalized force = %s Ha/a.u.",
                _format_value_and_error(twist_diagnostics.max_force, twist_diagnostics.max_force_std),
            )
            logger.info("Max generalized force signal-to-noise = %.3f.", twist_diagnostics.max_signal_to_noise)
            for twist, twist_max_signal_to_noise in twist_diagnostics.twist_local_max_signal_to_noise:
                logger.info("  Twist %d local max generalized force signal-to-noise = %.3f.", twist, twist_max_signal_to_noise)
        logger.info(
            "SR solver path: regime=%s, method=%s, backend=%s",
            target.sr_regime,
            "CG" if sr_cg_flag else "direct",
            optimizer_backend,
        )
        logger.info("-" * num_sep_line)

        logger.info(f"theta.size = {theta.size}.")
        logger.info(f"np.count_nonzero(theta) = {np.count_nonzero(theta)}.")
        logger.info(f"max. and min. of |theta| are {np.max(np.abs(theta))} and {np.min(np.abs(theta))}.")

        # Common log helpers
        _log_delta = 1.0

        # Guard against NaN/Inf components before applying the update and
        # report which blocks contain problematic entries.
        non_finite_mask = ~np.isfinite(theta)
        theta_is_finite = not bool(np.any(non_finite_mask))
        if not _collective_step_acceptance(theta_is_finite, twist_decomposed, sampler.world_comm):
            bad_blocks: list[str] = []
            for block, start, end in offsets:
                block_mask = non_finite_mask[start:end]
                if np.any(block_mask):
                    count_bad = int(np.count_nonzero(block_mask))
                    bad_blocks.append(f"{block.name}({count_bad}/{block.size})")
            logger.error(
                "Detected non-finite entries in %s update vector on at least "
                "one twist; rejecting the world-coupled step. Local blocks=%s",
                _log_label,
                ", ".join(bad_blocks) if bad_blocks else "unknown",
            )
            break

        # Emit per-block statistics so we can confirm that parameters are
        # actually updated (or detect suspiciously large steps).
        for block, start, end in offsets:
            block_theta = theta[start:end]
            if not np.any(block_theta):
                logger.info(
                    "  [%s update] - block=%s size=%d  theta=ALL ZERO (no update)",
                    _log_label,
                    block.name,
                    block.size,
                )
                continue
            block_norm = float(np.linalg.norm(block_theta))
            block_max = float(np.max(np.abs(block_theta)))
            block_delta_max = float(_log_delta * block_max)
            logger.info(
                "  [%s update] - block=%s size=%d  ||delta_p||=%.3e  max|delta_p|=%.3e  max|applied update|=%.3e",
                _log_label,
                block.name,
                block.size,
                block_norm,
                block_max,
                block_delta_max,
            )

        start = time.perf_counter()
        logger.info(f"Updating parameters with optimizer '{optimizer_mode}'.")
        logger.devel(f"dX.shape for MPI-rank={sampler.rank} is {theta.shape}")

        structure_data = sampler.hamiltonian_data.structure_data
        coulomb_potential_data = sampler.hamiltonian_data.coulomb_potential_data

        wavefunction_data_old = sampler.hamiltonian_data.wavefunction_data
        update_error: Exception | None = None
        try:
            wavefunction_data = wavefunction_data_old.apply_optimization_updates(
                blocks=blocks,
                updates=step_result.updates,
                expected_state_token=coordinate_map.state_token,
            )
            sampler.validate_wavefunction_candidate(wavefunction_data)
        except Exception as exc:  # candidate construction; no sampler state mutated
            update_error = exc
            wavefunction_data = None

        update_accepted = _collective_candidate_acceptance(
            update_error is None, sampler.comm, twist_decomposed, sampler.world_comm
        )
        if not update_accepted:
            if not twist_decomposed:
                # Every rank rejects, including peers whose own walkers passed.
                # Raising on only the failing rank would strand others at the
                # publication barrier after mutating their accepted state.
                if update_error is not None:
                    raise update_error
                raise ValueError("Candidate wavefunction was rejected on another sampler rank before publication.")
            logger.error(
                "Wave-function update/retraction failed on at least one twist; "
                "rejecting the world-coupled step before mutation. Local error=%r",
                update_error,
            )
            break
        assert wavefunction_data is not None

        hamiltonian_data = Hamiltonian_data(
            structure_data=structure_data,
            wavefunction_data=wavefunction_data,
            coulomb_potential_data=coulomb_potential_data,
        )
        logger.info("Wavefunction has been updated. Optimization loop is done.")
        logger.info("")
        sampler.hamiltonian_data = hamiltonian_data
        sampler.advance_wavefunction_revision()
        end = time.perf_counter()
        timer_param_update += end - start

        # MPI barrier after all optimization operation (timed)
        start = time.perf_counter()
        sampler.comm.Barrier()
        end = time.perf_counter()
        timer_MPI_barrier += end - start

        num_opt_done += 1

        # check max time
        optimization_current = time.perf_counter()

        # if max_time exceeded, or the next step would end after it, break the
        # loop. The global root decides and broadcasts, as MCMC.run does:
        # per-rank clocks differ, and a rank that broke alone would deadlock
        # the next collective of the others.
        world_comm = sampler.world_comm
        next_step_follows = (i_opt + 1) < num_opt_steps or config._more_steps_follow
        should_stop = (
            _optimization_budget_exhausted(
                optimization_current - optimization_total_start,
                num_opt_done,
                max_time,
                next_step_follows=next_step_follows,
            )
            if world_comm.Get_rank() == 0
            else False
        )
        should_stop = world_comm.bcast(should_stop, root=0)
        if should_stop:
            elapsed = optimization_current - optimization_total_start
            if elapsed <= max_time:
                logger.info(
                    f"The next optimization step is predicted to end after max_time: elapsed = {elapsed:.1f} sec., "
                    f"mean step = {elapsed / num_opt_done:.1f} sec."
                )
            logger.info(f"Stopping... max_time = {max_time} sec. exceeds.")
            stop_reason = "max_time"
            # dump WF (global rank 0 only; sampler.rank is the per-twist
            # intra-comm rank, so under twist averaging every twist-group
            # leader would write the same file and collide on HDF5 locking)
            if dump_wavefunction and sampler.world_comm.Get_rank() == 0:
                hamiltonian_data_filename = f"hamiltonian_data_opt_step_{i_opt + 1 + i_opt_offset}.h5"
                logger.info(f"Hamiltonian data is dumped as an HDF5 file: {hamiltonian_data_filename}.")
                hamiltonian_data_to_dump = sampler.hamiltonian_data
                hamiltonian_data_to_dump.save_to_hdf5(hamiltonian_data_filename)
            logger.info("Break the optimization loop.")
            break

        # MPI barrier before checking toml file
        sampler.comm.Barrier()

        # check the control file (stop flag); shared across all ranks / twists
        stop_flag = _read_external_stop_flag(_OPTIMIZER_CONTROL_FILE, world_comm)
        if stop_flag:
            logger.info(f"Stopping... stop_flag in {_OPTIMIZER_CONTROL_FILE} is true.")
            stop_reason = "stop_flag"
            # dump WF (global rank 0 only; see note above)
            if dump_wavefunction and sampler.world_comm.Get_rank() == 0:
                hamiltonian_data_filename = f"hamiltonian_data_opt_step_{i_opt + 1 + i_opt_offset}.h5"
                logger.info(f"Hamiltonian data is dumped as an HDF5 file: {hamiltonian_data_filename}.")
                hamiltonian_data_to_dump = sampler.hamiltonian_data
                hamiltonian_data_to_dump.save_to_hdf5(hamiltonian_data_filename)
            logger.info("Break the optimization loop.")
            logger.info("")
            break

        # dump WF (global rank 0 only; see note above)
        if dump_wavefunction and sampler.world_comm.Get_rank() == 0:
            if (i_opt + 1) % wavefunction_dump_frequency == 0 or (i_opt + 1) == num_opt_steps:
                hamiltonian_data_filename = f"hamiltonian_data_opt_step_{i_opt + 1 + i_opt_offset}.h5"
                logger.info(f"Hamiltonian data is dumped as an HDF5 file: {hamiltonian_data_filename}.")
                logger.info("")
                hamiltonian_data_to_dump = sampler.hamiltonian_data
                hamiltonian_data_to_dump.save_to_hdf5(hamiltonian_data_filename)

    # update WF opt counter
    # Only accepted wave-function updates advance the persistent counter.
    # In particular, a zero-step call and an abort before parameter update are
    # no-ops.
    sampler.advance_optimization_steps(num_opt_done)

    optimization_total_end = time.perf_counter()
    timer_opt_total += optimization_total_end - optimization_total_start
    timer_misc = timer_opt_total - (
        timer_sampling + timer_get_E + timer_get_gF + timer_optimizer + timer_param_update + timer_MPI_barrier
    )

    # average among MPI processes
    ave_timer_opt_total = sampler.comm.allreduce(timer_opt_total, op=MPI.SUM) / sampler.size
    if num_opt_done > 0:
        ave_timer_sampling = sampler.comm.allreduce(timer_sampling, op=MPI.SUM) / sampler.size / num_opt_done
        ave_timer_get_E = sampler.comm.allreduce(timer_get_E, op=MPI.SUM) / sampler.size / num_opt_done
        ave_timer_get_gF = sampler.comm.allreduce(timer_get_gF, op=MPI.SUM) / sampler.size / num_opt_done
        ave_timer_optimizer = sampler.comm.allreduce(timer_optimizer, op=MPI.SUM) / sampler.size / num_opt_done
        ave_timer_param_update = sampler.comm.allreduce(timer_param_update, op=MPI.SUM) / sampler.size / num_opt_done
        ave_timer_MPI_barrier = sampler.comm.allreduce(timer_MPI_barrier, op=MPI.SUM) / sampler.size / num_opt_done
        ave_timer_misc = sampler.comm.allreduce(timer_misc, op=MPI.SUM) / sampler.size / num_opt_done
    else:
        ave_timer_sampling = 0.0
        ave_timer_get_E = 0.0
        ave_timer_get_gF = 0.0
        ave_timer_optimizer = 0.0
        ave_timer_param_update = 0.0
        ave_timer_MPI_barrier = 0.0
        ave_timer_misc = 0.0

    logger.info(f"Total elapsed time for optimization {num_opt_done} steps. = {ave_timer_opt_total:.6f} sec.")
    logger.info(f"Elapsed times per optimization step, averaged over {num_opt_done} steps.")
    logger.info(f"  Time for sampling = {ave_timer_sampling:.6f} sec.")
    logger.info(f"  Time for computing E = {ave_timer_get_E:.6f} sec.")
    logger.info(f"  Time for computing generalized forces (gF) = {ave_timer_get_gF:.6f} sec.")
    logger.info(f"  Time for optimizer ({_log_label}) = {ave_timer_optimizer:.6f} sec.")
    logger.info(f"  Time for parameter update = {ave_timer_param_update:.6f} sec.")
    logger.info(f"  Time for MPI barrier = {ave_timer_MPI_barrier:.6f} sec.")
    logger.info(f"  Time for misc. (others) = {ave_timer_misc:.6f} sec.")
    logger.info("")

    # remove the control file (world root owns it; EAFP-guarded)
    sampler.comm.Barrier()
    if config._manage_external_control:
        _remove_external_control_file(_OPTIMIZER_CONTROL_FILE, sampler.world_comm)

    _log_memory_usage("optimize_post_loop", comm=sampler.comm)

    return stop_reason
