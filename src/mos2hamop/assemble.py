"""Assemble ML-predicted H and S matrices for arbitrary MoS2 structures.

Given a periodic structure (any supercell of the monolayer with
vacancies, strain, or displacements), every atom pair within the
interaction cutoff is featurized in its pair frame, its H and S blocks
are predicted, rotated back to the global frame, and placed into
real-space block lists compatible with kubo.bloch_matrices and with the
principal-layer partitioning of negf.py. Hermiticity is enforced by
averaging each block with the transpose of its reversed pair.
"""
import numpy as np

from .blocks import NAO
from .features import pair_descriptor, frame_angle
from .mlmodel import type_key, rotate_out_of_frame, select_features

RCUT = 11.0


def orbital_offsets(numbers):
    offs, n = [], 0
    for Z in numbers:
        offs.append(n)
        n += NAO[Z]
    return np.array(offs), n


def enumerate_pairs(positions, numbers, cell, images, min_image=False):
    """All (i, j, image, d) pairs within RCUT. images: list of (m1, m2).

    min_image=True (Gamma-sampled torus; images must be [(0, 0)]): each
    ordered pair (i, j) is taken once, at its minimum-image displacement
    over the 3x3 cell neighborhood, matching blocks.pair_blocks with
    kgrid=(1, 1), so predicted blocks correspond one-to-one to the
    folded DFT reference blocks.
    """
    pairs = []
    if min_image:
        assert list(images) == [(0, 0)]
        for i in range(len(numbers)):
            for j in range(len(numbers)):
                if i == j:
                    pairs.append((i, i, (0, 0), np.zeros(3), 0.0, 'onsite'))
                    continue
                best = None
                for m1 in (-1, 0, 1):
                    for m2 in (-1, 0, 1):
                        d = (positions[j] + m1 * cell[0] + m2 * cell[1]
                             - positions[i])
                        dist = np.linalg.norm(d)
                        if best is None or dist < best[1] - 1e-9:
                            best = (d, dist)
                d, dist = best
                if dist < RCUT:
                    pairs.append((i, j, (0, 0), d, dist, 'pair'))
        return pairs
    for (m1, m2) in images:
        R = m1 * cell[0] + m2 * cell[1]
        for i in range(len(numbers)):
            d_all = positions + R - positions[i]
            dist = np.linalg.norm(d_all, axis=1)
            for j in np.nonzero(dist < RCUT)[0]:
                if m1 == 0 and m2 == 0 and i == j:
                    pairs.append((i, i, (0, 0), np.zeros(3), 0.0, 'onsite'))
                else:
                    pairs.append((i, int(j), (m1, m2), d_all[j],
                                  float(dist[j]), 'pair'))
    return pairs


def predict_blocks(positions, numbers, cell, images, models, kgrid_env,
                   refs=None, exact_S=None, min_image=False):
    """Predict all blocks. Returns {(m1, m2): [(oi, oj, ni, nj, d, H, S)]}.

    models: {(kind, Zi, Zj): (BlockModel_H, BlockModel_S)}.
    refs: optional {(kind, Zi, Zj): (DistanceReference_H, DistanceReference_S)}
    added back to the MLP residual in the pair frame before rotation.
    kgrid_env: images used for the environment torus in the descriptor
    (pass the periodicity of the target cell, e.g. (1, 1) for a big
    supercell whose own cell is larger than the environment radius).
    """
    offs, nao = orbital_offsets(numbers)
    pairs = enumerate_pairs(positions, numbers, cell, images,
                            min_image=min_image)
    # batch descriptors per type
    per_type = {}
    for idx, (i, j, img, d, dist, kind) in enumerate(pairs):
        Zi, Zj = numbers[i], numbers[j]
        key = type_key(kind, Zi, Zj)
        store_swapped = False
        if key is None:  # S-Mo: predict the reversed pair, transpose later
            key = ('pair', Zj, Zi)
            store_swapped = True
        per_type.setdefault(key, []).append((idx, store_swapped))
    sample_like = lambda i, j, d, dist: dict(i=i, j=j, d=d, dist=dist)
    out = {}
    X_cache = {}
    for key, entries in per_type.items():
        X = []
        for idx, swapped in entries:
            i, j, img, d, dist, kind = pairs[idx]
            if swapped:
                s = sample_like(j, i, -d, dist)
            else:
                s = sample_like(i, j, d, dist)
            X.append(pair_descriptor(s, positions, numbers, cell, kgrid_env))
        X = np.array(X, dtype=np.float32)
        mH, mS = models[key]
        Xf = select_features(key[0], X)
        Hp = mH.predict(Xf)
        Sp = mS.predict(Xf)
        if refs is not None and key in refs:
            rH, rS = refs[key]
            Hp = Hp + np.array([rH.value(dd)[0] for dd in X[:, 0]])
            Sp = Sp + np.array([rS.value(dd)[0] for dd in X[:, 0]])
        ni_o, nj_o = mH.ni, mH.nj
        for (idx, swapped), hrow, srow in zip(entries, Hp, Sp):
            i, j, img, d, dist, kind = pairs[idx]
            Hb = hrow.reshape(ni_o, nj_o)
            Sb = srow.reshape(ni_o, nj_o)
            if swapped:
                theta = frame_angle(-d)
                Hb = rotate_out_of_frame(Hb, numbers[j], numbers[i], theta).T
                Sb = rotate_out_of_frame(Sb, numbers[j], numbers[i], theta).T
            else:
                theta = frame_angle(d)
                Hb = rotate_out_of_frame(Hb, numbers[i], numbers[j], theta)
                Sb = rotate_out_of_frame(Sb, numbers[i], numbers[j], theta)
            if exact_S is not None:
                ek = (i, j, tuple(np.round(d, 4)))
                if ek in exact_S:
                    Sb = exact_S[ek]
            out.setdefault(img, []).append(
                (offs[i], offs[j], NAO[numbers[i]], NAO[numbers[j]],
                 d, Hb, Sb))
    return out, nao


def hermitize(blocks_by_img):
    """Average each block with the transpose of its reversed partner."""
    index = {}
    for img, lst in blocks_by_img.items():
        for n, (oi, oj, ni, nj, d, Hb, Sb) in enumerate(lst):
            index[(img, oi, oj, tuple(np.round(d, 6)))] = (img, n)
    for img, lst in blocks_by_img.items():
        for n, (oi, oj, ni, nj, d, Hb, Sb) in enumerate(lst):
            rev = ((-img[0], -img[1]), oj, oi, tuple(np.round(-d, 6)))
            if rev in index:
                img2, n2 = index[rev]
                Hb2 = blocks_by_img[img2][n2][5]
                Sb2 = blocks_by_img[img2][n2][6]
                Hs = 0.5 * (Hb + Hb2.T)
                Ss = 0.5 * (Sb + Sb2.T)
                lst[n] = (oi, oj, ni, nj, d, Hs, Ss)
    return blocks_by_img


def flatten(blocks_by_img):
    lst = []
    for img, blocks in blocks_by_img.items():
        lst.extend(blocks)
    return lst
