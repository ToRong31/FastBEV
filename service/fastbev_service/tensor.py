from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


MAGIC_NUMBER = 0x33FF1101


class TensorFormatError(ValueError):
    pass


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: np.dtype
    shape: tuple[int, ...]


DTYPE_BY_CODE = {
    1: np.dtype("<i4"),
    2: np.dtype("<f2"),
    3: np.dtype("<f4"),
    4: np.dtype("<i8"),
    5: np.dtype("<u8"),
    6: np.dtype("<u4"),
    7: np.dtype("i1"),
    8: np.dtype("u1"),
}


def _read_all(source: bytes | bytearray | memoryview | BinaryIO) -> bytes:
    if isinstance(source, bytes):
        return source
    if isinstance(source, bytearray):
        return bytes(source)
    if isinstance(source, memoryview):
        return source.tobytes()
    return source.read()


def load_tensor(source: bytes | bytearray | memoryview | BinaryIO) -> np.ndarray:
    data = _read_all(source)
    if len(data) < 12:
        raise TensorFormatError("tensor file is too short to contain a header")

    magic, ndim, dtype_code = np.frombuffer(data[:12], dtype="<i4", count=3)
    if int(magic) != MAGIC_NUMBER:
        raise TensorFormatError("invalid tensor magic number")
    if int(ndim) < 0 or int(ndim) > 16:
        raise TensorFormatError(f"invalid tensor rank: {int(ndim)}")
    if int(dtype_code) not in DTYPE_BY_CODE:
        raise TensorFormatError(f"unsupported tensor dtype code: {int(dtype_code)}")

    dims_offset = 12
    dims_end = dims_offset + int(ndim) * 4
    if len(data) < dims_end:
        raise TensorFormatError("tensor file ended before shape metadata")

    dims = tuple(int(v) for v in np.frombuffer(data[dims_offset:dims_end], dtype="<i4", count=int(ndim)))
    if any(dim < 0 for dim in dims):
        raise TensorFormatError(f"invalid tensor shape: {dims}")

    dtype = DTYPE_BY_CODE[int(dtype_code)]
    numel = int(np.prod(dims, dtype=np.int64)) if dims else 1
    payload_bytes = numel * dtype.itemsize
    payload_end = dims_end + payload_bytes
    if len(data) != payload_end:
        raise TensorFormatError(
            f"tensor payload size mismatch: expected {payload_bytes} bytes, got {len(data) - dims_end}"
        )

    return np.frombuffer(data[dims_end:payload_end], dtype=dtype, count=numel).reshape(dims).copy()


def load_tensor_file(path: str | Path) -> np.ndarray:
    with open(path, "rb") as handle:
        return load_tensor(handle)


def validate_tensor(array: np.ndarray, spec: TensorSpec) -> np.ndarray:
    expected_dtype = np.dtype(spec.dtype)
    if array.dtype != expected_dtype:
        raise ValueError(f"{spec.name} must be dtype {expected_dtype}, got {array.dtype}")
    if tuple(array.shape) != spec.shape:
        raise ValueError(f"{spec.name} must have shape {spec.shape}, got {tuple(array.shape)}")
    return np.ascontiguousarray(array)
