"""Shared helpers for datasets stored in the IDX binary format (MNIST, EMNIST):
downloading with resume-skip, and parsing the format itself.

IDX format: a 4-byte header (2 zero bytes, 1 dtype-code byte, 1 ndims byte),
followed by `ndims` big-endian uint32 dimension sizes, followed by raw data.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

_IDX_DTYPES: dict[int, type[np.generic]] = {
    0x08: np.uint8,
    0x09: np.int8,
    0x0B: np.int16,
    0x0C: np.int32,
    0x0D: np.float32,
    0x0E: np.float64,
}


def download(url: str, dest: Path) -> None:
    """No-op if dest already exists — datasets are never re-downloaded/overwritten
    once fetched. Downloads to a .part file first so a partial download can't be
    mistaken for a complete one."""
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp_path)  # noqa: S310 — fixed, project-controlled URLs
    tmp_path.rename(dest)


def read_idx_gzip(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        header = f.read(4)
        data_type_code, num_dims = header[2], header[3]
        dtype = _IDX_DTYPES[data_type_code]
        shape = tuple(struct.unpack(">I", f.read(4))[0] for _ in range(num_dims))
        data = np.frombuffer(f.read(), dtype=dtype)
    return data.reshape(shape)
