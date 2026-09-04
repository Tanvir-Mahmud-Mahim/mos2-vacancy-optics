"""Held-out validation data for the learned Hamiltonian blocks.

An 85/15 split by structure is used: models are re-fit on the training
split and evaluated on the held-out blocks, so the reported errors are
genuine generalization errors. Saves matrix-element parity and the
error as a function of interatomic distance.
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.mlmodel import BlockModel, BLOCK_TYPES, select_features
from mos2hamop.blocks import NAO
from mos2hamop.reference import DistanceReference

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')


def main():
    files = sorted(glob.glob(os.path.join(ROOT, 'samples_train', '*.npz')))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(files))
    ntr = max(1, int(0.85 * len(files)))
    tr = [files[i] for i in idx[:ntr]]; te = [files[i] for i in idx[ntr:]]
    if not te:
        te = [files[idx[-1]]]
    parity_dft, parity_ml, parity_dist = [], [], []
    report = {}
    for kind, Zi, Zj in BLOCK_TYPES:
        tag = f'{kind}_{Zi}_{Zj}'
        def load(fl):
            X, Y = [], []
            for f in fl:
                d = np.load(f)
                if 'X_' + tag in d:
                    X.append(d['X_' + tag]); Y.append(d['H_' + tag])
            return (np.concatenate(X), np.concatenate(Y)) if X else (None, None)
        Xtr, Ytr = load(tr); Xte, Yte = load(te)
        if Xtr is None or Xte is None:
            continue
        ni, nj = NAO[Zi], NAO[Zj]
        dtr = Xtr[:, 0]
        ref = DistanceReference(dtr.min()-.01, dtr.max()+.01, 24, ni, nj,
                                kind == 'onsite')
        ref.fit(dtr, Ytr)
        rtr = np.array([ref.value(x)[0] for x in dtr])
        Xtr_f = select_features(kind, Xtr); Xte_f = select_features(kind, Xte)
        m = BlockModel(Xtr_f.shape[1], ni, nj)
        m.fit(Xtr_f, Ytr - rtr, epochs=300, seed=1, patience=25)
        rte = np.array([ref.value(x)[0] for x in Xte[:, 0]])
        pred = m.predict(Xte_f) + rte
        rmse = float(np.sqrt(np.mean((pred - Yte) ** 2)))
        report[tag] = dict(rmse_meV=rmse * 1e3, n_test=int(len(Xte)))
        print(f'{tag}: held-out RMSE {rmse*1e3:.1f} meV ({len(Xte)} blocks)',
              flush=True)
        # parity sample (subset), and distance of each block
        sel = rng.choice(len(Yte), min(4000, len(Yte)), replace=False)
        parity_dft.append(Yte[sel].ravel())
        parity_ml.append(pred[sel].ravel())
        parity_dist.append(np.repeat(Xte[sel, 0], Yte.shape[1]))
    np.savez_compressed(os.path.join(ROOT, 'ml_parity.npz'),
                        dft=np.concatenate(parity_dft),
                        ml=np.concatenate(parity_ml),
                        dist=np.concatenate(parity_dist))
    json.dump(report, open(os.path.join(ROOT, 'ml_report.json'), 'w'), indent=1)
    print('ML validation done')


if __name__ == '__main__':
    main()
