"""Robust generalized eigensolver with canonical orthogonalization.

The LCAO overlap S is mildly overcomplete, so it has eigenvalues close
to zero. Small errors in a learned Hamiltonian in that near-null space
are amplified enormously by a naive generalized eigensolver. Canonical
orthogonalization removes the offending directions: S is diagonalized,
directions with eigenvalue below a threshold are dropped, and H is
solved in the remaining well-conditioned subspace. This is the standard
remedy used inside electronic-structure codes.
"""
import numpy as np
from scipy.linalg import eigh


def gen_eigh(H, S, thresh=1e-4, eigvals_only=True):
    H = 0.5 * (H + H.conj().T)
    S = 0.5 * (S + S.conj().T)
    s, U = eigh(S)
    keep = s > thresh
    X = U[:, keep] / np.sqrt(s[keep])
    Hp = X.conj().T @ H @ X
    if eigvals_only:
        return eigh(Hp, eigvals_only=True)
    w, Vp = eigh(Hp)
    return w, X @ Vp
