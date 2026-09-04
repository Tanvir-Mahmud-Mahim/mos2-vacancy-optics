"""Spectral validation of the learned Hamiltonian on Gamma-point 5x5 cells.

In the 5x5 supercell sampled at the zone centre, the largest
minimum-image pair distance (about 9.3 A) lies inside the 11 A learning range,
so every Hamiltonian and overlap block of the periodic torus is local by
construction and the range obstruction of range_analysis.py is absent.
This script trains the block models on the Gamma 5x5 training set
(samples_train55, built by build_samples.py), then, for every held-out
test structure, assembles the fully learned H and S from the geometry
alone and compares eigenvalues, the gap, and the Kubo sub-gap absorption
A_sub against the exact DFT result computed with identical settings.
Everything is a genuine held-out prediction: the test structures never
enter training.
"""
import os, sys, glob, json, pickle, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.blocks import realspace_matrices, pair_blocks, orbital_offsets
from mos2hamop.mlmodel import (BlockModel, BLOCK_TYPES, select_features)
from mos2hamop.reference import DistanceReference
from mos2hamop.assemble import predict_blocks, hermitize
from mos2hamop.eigsolve import gen_eigh
from mos2hamop.kubo import sigma_xx

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
ETA = 0.08
KGRID = (1, 1)
# Overlap-eigenvalue threshold for canonical orthogonalization. The dzp
# LCAO basis is strongly overcomplete: the overlap has hundreds of
# eigenvalues below 0.1, and eigenvalue errors of a learned Hamiltonian
# in that near-null subspace are amplified by 1/s. Both the reference
# and the learned operators are therefore read out in the
# well-conditioned subspace s > THRESH, and the (small) effect of the
# projection on the reference observables is recorded alongside.
THRESH = 0.1
EXACT_THRESH = 1e-4   # reproduces the stored DFT eigenvalues to ~2 meV


def train():
    files = sorted(glob.glob(os.path.join(ROOT, 'samples_train55', '*.npz')))
    print(len(files), 'gamma-5x5 sample files', flush=True)
    models, refs = {}, {}
    for kind, Zi, Zj in BLOCK_TYPES:
        tag = f'{kind}_{Zi}_{Zj}'
        X, YH, YS = [], [], []
        for fn in files:
            d = np.load(fn)
            if 'X_' + tag not in d:
                continue
            X.append(d['X_' + tag]); YH.append(d['H_' + tag])
            YS.append(d['S_' + tag])
        X = np.concatenate(X); YH = np.concatenate(YH); YS = np.concatenate(YS)
        from mos2hamop.blocks import NAO
        ni, nj = NAO[Zi], NAO[Zj]
        dist = X[:, 0]
        onsite = (kind == 'onsite')
        refH = DistanceReference(max(dist.min()-.01, 0), dist.max()+.01, 24,
                                 ni, nj, onsite); refH.fit(dist, YH)
        refS = DistanceReference(max(dist.min()-.01, 0), dist.max()+.01, 24,
                                 ni, nj, onsite); refS.fit(dist, YS)
        rH = np.array([refH.value(x)[0] for x in dist])
        rS = np.array([refS.value(x)[0] for x in dist])
        Xf = select_features(kind, X)
        t0 = time.time()
        mH = BlockModel(Xf.shape[1], ni, nj)
        mH.fit(Xf, YH - rH, epochs=350, seed=1, patience=25,
               log=lambda *a: None)
        mS = BlockModel(Xf.shape[1], ni, nj)
        mS.fit(Xf, YS - rS, epochs=250, seed=2, patience=25,
               log=lambda *a: None)
        print(f'{tag}: {len(X)} blocks, trained in {time.time()-t0:.0f}s',
              flush=True)
        models[(kind, Zi, Zj)] = (mH, mS)
        refs[(kind, Zi, Zj)] = (refH, refS)
    pickle.dump({k: (a.state(), b.state()) for k, (a, b) in models.items()},
                open(os.path.join(ROOT, 'models55.pkl'), 'wb'))
    pickle.dump({k: (a.state(), b.state()) for k, (a, b) in refs.items()},
                open(os.path.join(ROOT, 'refs55.pkl'), 'wb'))
    return models, refs


def load_models():
    models = {}
    for key, (stH, stS) in pickle.load(
            open(os.path.join(ROOT, 'models55.pkl'), 'rb')).items():
        mH = BlockModel(len(stH['x_mean']), stH['ni'], stH['nj']); mH.load(stH)
        mS = BlockModel(len(stS['x_mean']), stS['ni'], stS['nj']); mS.load(stS)
        models[key] = (mH, mS)
    refs = {k: (DistanceReference.load(a), DistanceReference.load(b))
            for k, (a, b) in pickle.load(
                open(os.path.join(ROOT, 'refs55.pkl'), 'rb')).items()}
    return models, refs


def flat_from_dft(d):
    mats = realspace_matrices(d['H_kMM'], d['S_kMM'], d['kpts'], KGRID)
    mr = {k: (H.real, S.real) for k, (H, S) in mats.items()}
    sam = pair_blocks(mr, d['positions'], d['numbers'], d['cell'], KGRID,
                      rcut=100.0)
    offs, nao = orbital_offsets(d['numbers'])
    return [(offs[s['i']], offs[s['j']], s['H'].shape[0], s['H'].shape[1],
             s['d'], s['H'], s['S']) for s in sam], nao


def flat_from_ml(d, models, refs):
    # Gamma-only torus: one block per ordered pair, at its minimum-image
    # displacement, exactly the convention of the folded DFT reference
    bbi, nao = predict_blocks(d['positions'], d['numbers'], d['cell'],
                              [(0, 0)], models, KGRID, refs,
                              min_image=True)
    bbi = hermitize(bbi)
    return [b for lst in bbi.values() for b in lst], nao


def spectrum(flat, nao, d, omega, thresh):
    cell = d['cell']
    icell = 2 * np.pi * np.linalg.inv(cell).T
    ks = [kf @ icell for kf in d['kpts']]
    wk = np.ones(len(ks)) / len(ks)
    area = float(np.abs(np.linalg.det(cell[:2, :2])))
    num = d['numbers']
    nocc = int(round((14 * (num == 42).sum() + 6 * (num == 16).sum()) / 2))
    # place the chemical potential mid-gap by electron count, identically
    # for the DFT and the learned spectrum
    from mos2hamop.kubo import bloch_matrices
    H, S, _, _ = bloch_matrices(flat, nao, ks[0], None)
    w = gen_eigh(H, S, thresh=thresh, eigvals_only=True)
    mu = 0.5 * (w[nocc - 1] + w[nocc])
    gap = float(w[nocc] - w[nocc - 1])
    sig, ne, e_all = sigma_xx(flat, nao, ks, wk, area, omega, mu=mu, eta=ETA,
                              s_thresh=thresh)
    return sig, w, mu, gap, nocc


def main():
    if not os.path.exists(os.path.join(ROOT, 'models55.pkl')):
        train()
    models, refs = load_models()
    omega = np.linspace(0.05, 3.0, 90)

    # pristine gap of the gamma-5x5 reference (exact basis), for the
    # A_sub window, identical for every spectrum
    dp = np.load(sorted(glob.glob(
        os.path.join(ROOT, 'train55', 'g55_prist_000.npz')))[0])
    flat_p, nao_p = flat_from_dft(dp)
    _, wp, mup, gap0, _ = spectrum(flat_p, nao_p, dp, omega, EXACT_THRESH)
    print(f'pristine gamma-5x5 gap {gap0:.3f} eV', flush=True)

    def a_sub(sig):
        m = (omega > 0.15) & (omega < gap0 - 0.2)
        return float(np.trapezoid(sig[m], omega[m]))

    out = []
    save = {'omega': omega, 'gap0': gap0, 'thresh': THRESH}
    for fn in sorted(glob.glob(os.path.join(ROOT, 'test55', '*.npz'))):
        name = os.path.basename(fn)[:-4]
        d = np.load(fn)
        nvac = int(50 - (d['numbers'] == 16).sum())
        t0 = time.time()
        flat_d, nao = flat_from_dft(d)
        # exact-basis reference observables
        sig_x, w_x, mu_x, gap_x, nocc = spectrum(flat_d, nao, d, omega,
                                                 EXACT_THRESH)
        # reference and learned spectra in the common well-conditioned
        # subspace (same threshold for both)
        sig_d, w_d, mu_d, gap_d, _ = spectrum(flat_d, nao, d, omega, THRESH)
        flat_m, nao_m = flat_from_ml(d, models, refs)
        sig_m, w_m, mu_m, gap_m, _ = spectrum(flat_m, nao_m, d, omega, THRESH)
        n = min(len(w_d), len(w_m))
        win = (w_d[:n] > mu_d - 2) & (w_d[:n] < mu_d + 2)
        mae_win = float(np.abs((w_m[:n] - mu_m) - (w_d[:n] - mu_d))[win].mean())
        r = dict(name=name, nvac=nvac,
                 gap_exact=gap_x, gap_dft=gap_d, gap_ml=gap_m,
                 a_sub_exact=a_sub(sig_x), a_sub_dft=a_sub(sig_d),
                 a_sub_ml=a_sub(sig_m),
                 eig_mae_win_meV=mae_win * 1e3)
        out.append(r)
        save[f'exact_{name}'] = sig_x
        save[f'dft_{name}'] = sig_d
        save[f'ml_{name}'] = sig_m
        print(f'{name}: nv={nvac} gap exact {gap_x:.3f} DFT {gap_d:.3f} '
              f'ML {gap_m:.3f} | A_sub exact {r["a_sub_exact"]:.2f} '
              f'DFT {r["a_sub_dft"]:.2f} ML {r["a_sub_ml"]:.2f} | '
              f'eig MAE {mae_win*1e3:.0f} meV ({time.time()-t0:.0f}s)',
              flush=True)

    ad = np.array([r['a_sub_dft'] for r in out])
    am = np.array([r['a_sub_ml'] for r in out])
    ax_ = np.array([r['a_sub_exact'] for r in out])
    pear = float(np.corrcoef(ad, am)[0, 1]) if len(ad) > 2 else float('nan')
    pear_x = float(np.corrcoef(ax_, am)[0, 1]) if len(ax_) > 2 else float('nan')
    json.dump(dict(results=out, gap0=gap0, thresh=THRESH, pearson=pear,
                   pearson_exact=pear_x),
              open(os.path.join(ROOT, 'spectral_validation.json'), 'w'),
              indent=1)
    np.savez_compressed(os.path.join(ROOT, 'spectral_validation.npz'), **save)
    print(f'A_sub ML vs projected DFT pearson: {pear:.3f}; '
          f'vs exact DFT: {pear_x:.3f}')
    print('wrote spectral_validation.json / .npz')


if __name__ == '__main__':
    main()
