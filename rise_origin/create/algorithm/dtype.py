# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The dtype vocabulary of the algorithm code, based on numpy types."""

import numpy as np

__all__ = [
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "uint64",
    "int64",
    "float16",
    "float32",
    "float64",
    "make_torch_dtype",
]

uint8 = np.uint8
int8 = np.int8
uint16 = np.uint16
int16 = np.int16
uint32 = np.uint32
int32 = np.int32
uint64 = np.uint64
int64 = np.int64
float16 = np.float16
float32 = np.float32
float64 = np.float64


def make_torch_dtype(dtype: object) -> object:
    """Return the torch dtype matching a vocabulary dtype.

    Args:
        dtype: The dtype to translate, anything ``np.dtype`` accepts.
    """
    import torch

    return getattr(torch, np.dtype(dtype).name)
