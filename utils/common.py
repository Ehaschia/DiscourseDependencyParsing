"""data type helper and constants"""

from typing import Sequence

import numpy as np

# constant
STOP = 0  # decision
GO = 1  # decision
NOCHILD = 0  # valence
HASCHILD = 1  # valence

# float

npf = np.float32

# int
npi = np.int64


# zeros


def npizeros(*args, **kwargs):
    return np.zeros(*args, **kwargs, dtype=npi)


def npfzeros(*args, **kwargs):
    return np.zeros(*args, **kwargs, dtype=npf)


# fill


def npifull(*args, **kwargs):
    return np.full(*args, **kwargs, dtype=npi)


def npffull(*args, **kwargs):
    return np.full(*args, **kwargs, dtype=npf)


# empty


def npiempty(*args, **kwargs):
    return np.empty(*args, **kwargs, dtype=npi)


def npfempty(*args, **kwargs):
    return np.empty(*args, **kwargs, dtype=npf)


def npasarray(a, dtype=None) -> np.ndarray:
    if dtype is None:
        if isinstance(a, np.ndarray):
            is_int = np.issubdtype(a.dtype, np.integer)
        elif isinstance(a, Sequence):
            _a = a[0]
            while isinstance(_a, Sequence):
                _a = _a[0]
            is_int = type(_a) == int \
                     or np.issubdtype(_a.dtype, np.integer)
        else:
            raise ValueError(f"Cannot deal with type {type(a)}")
        dtype = npi if is_int else npf
    return np.asarray(a, dtype)
