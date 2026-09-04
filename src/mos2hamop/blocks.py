"""Real-space Hamiltonian blocks from k-space LCAO matrices.

GPAW returns H(k) and S(k) on the Monkhorst-Pack grid of the supercell.
The inverse Fourier sum over that grid gives the real-space matrices
H(R) on the Born-von-Karman lattice (supercell repeated by the k-grid),
from which per-atom-pair orbital blocks are cut. The transform and its
phase convention are verified numerically in tests/test_blocks.py by
round-tripping H(R) -> H(k) and by checking Hermiticity and decay.
"""
import numpy as np

NAO = {42: 29, 16: 13}  # GPAW official dzp basis size: Mo, S
# orbital angular momenta per radial function, in exact GPAW LCAO order.
# Mo dzp (29): 4s,5s,4p,5p,4d (sz) | 4s,5s,4p,5p,4d (dz) | p polarization
# S  dzp (13): 3s,3p (sz) | 3s,3p (dz) | d polarization
L_LIST = {42: [0, 0, 1, 1, 2, 0, 0, 1, 1, 2, 1],
          16: [0, 1, 0, 1, 2]}


def orbital_offsets(numbers):
    offs, n = [], 0
    for Z in numbers:
        offs.append(n)
        n += NAO[Z]
    return np.array(offs), n


def realspace_matrices(H_kMM, S_kMM, kpts, kgrid):
    """H(R), S(R) for R on the BvK lattice defined by the k-grid.

    kpts: scaled k-points (fractional, supercell BZ). kgrid: (n1, n2).
    Returns dict {(r1, r2): (H_R, S_R)} with r_i in 0..n_i-1, and
    H(k) = sum_R exp(+2 pi i k.R) H(R) reproduces the input.
    """
    n1, n2 = kgrid
    nk = len(kpts)
    assert nk == n1 * n2
    out = {}
    for r1 in range(n1):
        for r2 in range(n2):
            R = np.array([r1, r2, 0.0])
            phase = np.exp(-2j * np.pi * (kpts @ R))
            H_R = np.tensordot(phase, H_kMM, axes=(0, 0)) / nk
            S_R = np.tensordot(phase, S_kMM, axes=(0, 0)) / nk
            out[(r1, r2)] = (H_R, S_R)
    return out


def reconstruct_k(mats_R, kpt):
    """H(k), S(k) at an arbitrary fractional k from the R-space matrices."""
    H = S = None
    for (r1, r2), (H_R, S_R) in mats_R.items():
        ph = np.exp(2j * np.pi * (kpt[0] * r1 + kpt[1] * r2))
        H = ph * H_R if H is None else H + ph * H_R
        S = ph * S_R if S is None else S + ph * S_R
    return H, S


def pair_blocks(mats_R, positions, numbers, cell, kgrid, rcut=11.0):
    """Cut per-pair orbital blocks H_ij(R), S_ij(R) within rcut.

    Pair vector convention: d = (pos_j + R_cart) - pos_i, folded into the
    minimum image of the BvK supercell. Yields dicts with i, j, d (Cartesian
    displacement), Zi, Zj, H (nio x njo), S.
    """
    offs, nao = orbital_offsets(numbers)
    n1, n2 = kgrid
    samples = []
    for (r1, r2), (H_R, S_R) in mats_R.items():
        for i in range(len(numbers)):
            for j in range(len(numbers)):
                # unique minimum image of atom j in cell R relative to i;
                # every representative m == r (mod n) must be tried, so the
                # +n image is included, not only r and r-n
                best = None
                for m1 in (r1 - n1, r1, r1 + n1):
                    for m2 in (r2 - n2, r2, r2 + n2):
                        R_cart = m1 * cell[0] + m2 * cell[1]
                        d = positions[j] + R_cart - positions[i]
                        dist = np.linalg.norm(d)
                        if best is None or dist < best[1] - 1e-9:
                            best = (d, dist, m1, m2)
                d, dist, m1, m2 = best
                if dist > rcut:
                    continue
                kind = 'onsite' if (r1 == 0 and r2 == 0 and i == j) else 'pair'
                oi, oj = offs[i], offs[j]
                blk_H = H_R[oi:oi + NAO[numbers[i]], oj:oj + NAO[numbers[j]]]
                blk_S = S_R[oi:oi + NAO[numbers[i]], oj:oj + NAO[numbers[j]]]
                samples.append(dict(i=i, j=j, d=d, dist=dist,
                                    Zi=numbers[i], Zj=numbers[j],
                                    kind=kind, H=blk_H, S=blk_S))
    return samples
