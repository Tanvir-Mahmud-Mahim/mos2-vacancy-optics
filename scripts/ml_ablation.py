"""Ablation and state-of-the-art comparison for the learned Hamiltonian.

Every variant is trained and evaluated on the *same* 85/15 by-structure
split used in ml_validation.py, so all numbers are genuine held-out
generalization errors on identical test blocks. Five variants isolate the
two physical ingredients of the model (the distance reference and the
equivariant environment descriptor):

  A  global-mean block            single mean block, no distance, no MLP
  B  distance two-center (SOTA)   distance-binned reference only (a
                                  conventional Slater-Koster-type
                                  two-center tight-binding parameterization)
  C  environment MLP, no ref      full descriptor MLP on the raw block
  D  reference + scalar MLP       distance reference + MLP on scalars only
                                  (no equivariant environment)
  E  reference + environment MLP  the proposed model (full descriptor
                                  residual on the distance reference)

The headline comparison is E versus B: how much the learned model beats
the conventional distance-parameterized tight-binding baseline.
"""
import os, sys, glob, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.mlmodel import BlockModel, BLOCK_TYPES
from mos2hamop.blocks import NAO
from mos2hamop.reference import DistanceReference

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
SCALAR_COLS = 4          # [dist, dx, dz, dz*dz] frame invariants
EPOCHS = 220
PATIENCE = 22


def load(fl, tag):
    X, Y = [], []
    for f in fl:
        d = np.load(f)
        if 'X_' + tag in d:
            X.append(d['X_' + tag]); Y.append(d['H_' + tag])
    return (np.concatenate(X), np.concatenate(Y)) if X else (None, None)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)) * 1e3)   # meV


def main():
    files = sorted(glob.glob(os.path.join(ROOT, 'samples_train', '*.npz')))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(files))
    ntr = max(1, int(0.85 * len(files)))
    tr = [files[i] for i in idx[:ntr]]
    te = [files[i] for i in idx[ntr:]] or [files[idx[-1]]]

    variants = ['A_globalmean', 'B_distanceTB', 'C_envMLP_noref',
                'D_ref_scalarMLP', 'E_ref_envMLP']
    per = {v: {} for v in variants}

    for kind, Zi, Zj in BLOCK_TYPES:
        tag = f'{kind}_{Zi}_{Zj}'
        Xtr, Ytr = load(tr, tag); Xte, Yte = load(te, tag)
        if Xtr is None or Xte is None:
            continue
        ni, nj = NAO[Zi], NAO[Zj]
        onsite = (kind == 'onsite')
        dtr, dte = Xtr[:, 0], Xte[:, 0]
        n_test = int(len(Xte))
        print(f'\n=== {tag}: {len(Xtr)} train, {n_test} test blocks ===',
              flush=True)

        # A: single global-mean block (one bin)
        refA = DistanceReference(0, 1, 1, ni, nj, True)
        refA.fit(dtr, Ytr)
        predA = np.repeat(refA.ref[0][None, :], n_test, axis=0)
        per['A_globalmean'][tag] = dict(rmse_meV=rmse(predA, Yte), n_test=n_test)

        # B: distance-binned two-center reference (conventional TB, SOTA)
        refB = DistanceReference(dtr.min()-.01, dtr.max()+.01, 24, ni, nj, onsite)
        refB.fit(dtr, Ytr)
        predB = np.array([refB.value(x)[0] for x in dte])
        per['B_distanceTB'][tag] = dict(rmse_meV=rmse(predB, Yte), n_test=n_test)

        # C: environment MLP on the raw block (no reference)
        t0 = time.time()
        mC = BlockModel(Xtr.shape[1], ni, nj)
        mC.fit(Xtr, Ytr, epochs=EPOCHS, seed=1, patience=PATIENCE, log=lambda *a: None)
        predC = mC.predict(Xte)
        per['C_envMLP_noref'][tag] = dict(rmse_meV=rmse(predC, Yte), n_test=n_test)
        print(f'  C envMLP-noref  {per["C_envMLP_noref"][tag]["rmse_meV"]:.1f} meV'
              f'  ({time.time()-t0:.0f}s)', flush=True)

        # D: distance reference + MLP on scalar invariants only (no environment)
        rtr = np.array([refB.value(x)[0] for x in dtr])
        rte = np.array([refB.value(x)[0] for x in dte])
        t0 = time.time()
        mD = BlockModel(SCALAR_COLS, ni, nj)
        mD.fit(Xtr[:, :SCALAR_COLS], Ytr - rtr, epochs=EPOCHS, seed=1,
               patience=PATIENCE, log=lambda *a: None)
        predD = mD.predict(Xte[:, :SCALAR_COLS]) + rte
        per['D_ref_scalarMLP'][tag] = dict(rmse_meV=rmse(predD, Yte), n_test=n_test)
        print(f'  D ref+scalarMLP {per["D_ref_scalarMLP"][tag]["rmse_meV"]:.1f} meV'
              f'  ({time.time()-t0:.0f}s)', flush=True)

        # E: distance reference + full environment MLP (proposed)
        t0 = time.time()
        mE = BlockModel(Xtr.shape[1], ni, nj)
        mE.fit(Xtr, Ytr - rtr, epochs=EPOCHS, seed=1, patience=PATIENCE,
               log=lambda *a: None)
        predE = mE.predict(Xte) + rte
        per['E_ref_envMLP'][tag] = dict(rmse_meV=rmse(predE, Yte), n_test=n_test)
        print(f'  E ref+envMLP    {per["E_ref_envMLP"][tag]["rmse_meV"]:.1f} meV'
              f'  ({time.time()-t0:.0f}s)', flush=True)

    # overall count-weighted RMSE per variant (pairs + onsite)
    overall = {}
    pair_overall = {}
    for v in variants:
        tot = sum(d['n_test'] for d in per[v].values())
        ss = sum(d['n_test'] * d['rmse_meV']**2 for d in per[v].values())
        overall[v] = float(np.sqrt(ss / tot))
        ptot = sum(d['n_test'] for t, d in per[v].items() if t.startswith('pair'))
        pss = sum(d['n_test'] * d['rmse_meV']**2
                  for t, d in per[v].items() if t.startswith('pair'))
        pair_overall[v] = float(np.sqrt(pss / ptot)) if ptot else 0.0

    out = dict(per_type=per, overall_meV=overall, pair_overall_meV=pair_overall,
               n_train_struct=len(tr), n_test_struct=len(te))
    json.dump(out, open(os.path.join(ROOT, 'ablation.json'), 'w'), indent=1)
    print('\n==== overall count-weighted RMSE (meV) ====')
    for v in variants:
        print(f'  {v:18s} all {overall[v]:7.1f}   pairs {pair_overall[v]:7.1f}')
    gain = per['B_distanceTB']  # noqa
    print('\nheadline: pair RMSE', round(pair_overall['B_distanceTB'], 1),
          '->', round(pair_overall['E_ref_envMLP'], 1), 'meV',
          f'({pair_overall["B_distanceTB"]/pair_overall["E_ref_envMLP"]:.1f}x)')
    print('wrote', os.path.join(ROOT, 'ablation.json'))


if __name__ == '__main__':
    main()
