"""Rotation of GPAW real-spherical-harmonic orbital blocks about z.

The per-degree rotation matrices are taken directly from GPAW's own
``gpaw.rotation.rotation(l, U)`` so that the orbital convention matches
the LCAO basis exactly (verified in tests/test_blocks.py by requiring
symmetry-equivalent pairs to give identical frame-aligned blocks). Only
rotations about the out-of-plane z axis are used, since the monolayer
normal is fixed and pair frames differ from the global frame by an
in-plane rotation.
"""
import numpy as np
from functools import lru_cache
from gpaw.rotation import rotation as _gpaw_rotation

from .blocks import L_LIST, NAO


def _Uz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@lru_cache(maxsize=8192)
def _dz_l_cached(l, tkey):
    return _gpaw_rotation(l, _Uz(tkey * 1e-9))


def dz_l(l, t):
    return _gpaw_rotation(l, _Uz(t))


@lru_cache(maxsize=8192)
def _atom_rotation_cached(Z, tkey):
    t = tkey * 1e-9
    n = NAO[Z]
    D = np.zeros((n, n))
    o = 0
    for l in L_LIST[Z]:
        B = _gpaw_rotation(l, _Uz(t))
        m = 2 * l + 1
        D[o:o + m, o:o + m] = B
        o += m
    assert o == n
    return D


def atom_rotation(Z, t):
    """Block-diagonal z-rotation matrix for all dzp orbitals of element Z."""
    return _atom_rotation_cached(Z, int(round(t * 1e9)))


def rotate_block(H, Zi, Zj, t):
    """Rotate an orbital block when the structure is rotated by +t about z."""
    Di = atom_rotation(Zi, t)
    Dj = atom_rotation(Zj, t)
    return Di @ H @ Dj.T
