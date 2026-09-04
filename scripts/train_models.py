"""Train the block MLPs on residuals off a distance-binned reference."""
import os, sys, glob, time, pickle
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.mlmodel import BlockModel, BLOCK_TYPES, select_features
from mos2hamop.blocks import NAO
from mos2hamop.reference import DistanceReference

root = os.path.join(os.path.dirname(__file__), '..', 'data')
files = sorted(glob.glob(os.path.join(root, 'samples_train', '*_samples.npz')))
print(len(files), 'sample files')

models, refs, report = {}, {}, {}
for kind, Zi, Zj in BLOCK_TYPES:
    tag = f'{kind}_{Zi}_{Zj}'
    X, YH, YS = [], [], []
    for fn in files:
        d = np.load(fn)
        if 'X_' + tag not in d:
            continue
        X.append(d['X_' + tag]); YH.append(d['H_' + tag]); YS.append(d['S_' + tag])
    X = np.concatenate(X); YH = np.concatenate(YH); YS = np.concatenate(YS)
    ni, nj = NAO[Zi], NAO[Zj]
    dist = X[:, 0]
    onsite = (kind == 'onsite')
    dmin = max(dist.min() - 0.01, 0.0); dmax = dist.max() + 0.01
    refH = DistanceReference(dmin, dmax, 24, ni, nj, onsite); refH.fit(dist, YH)
    refS = DistanceReference(dmin, dmax, 24, ni, nj, onsite); refS.fit(dist, YS)
    rH = np.array([refH.value(dd)[0] for dd in dist])
    rS = np.array([refS.value(dd)[0] for dd in dist])
    Xf = select_features(kind, X)
    print(f'{tag}: {len(X)} samples, block {ni}x{nj}, {Xf.shape[1]} features, '
          f'ref removes H std {YH.std():.3f}->{(YH - rH).std():.3f} eV')
    t0 = time.time()
    mH = BlockModel(Xf.shape[1], ni, nj); vH = mH.fit(Xf, YH - rH, epochs=350, seed=1, patience=25)
    mS = BlockModel(Xf.shape[1], ni, nj); vS = mS.fit(Xf, YS - rS, epochs=250, seed=2, patience=25)
    rmsH = np.sqrt(np.mean((mH.predict(Xf) - (YH - rH)) ** 2))
    rmsS = np.sqrt(np.mean((mS.predict(Xf) - (YS - rS)) ** 2))
    print(f'{tag}: residual train RMSE H {rmsH*1e3:.1f} meV, S {rmsS:.2e}, '
          f'{time.time()-t0:.0f} s')
    models[(kind, Zi, Zj)] = (mH, mS)
    refs[(kind, Zi, Zj)] = (refH, refS)
    report[tag] = dict(n=int(len(X)), rmsH=float(rmsH), rmsS=float(rmsS))

pickle.dump({k: (a.state(), b.state()) for k, (a, b) in models.items()},
            open(os.path.join(root, 'models.pkl'), 'wb'))
pickle.dump({k: (a.state(), b.state()) for k, (a, b) in refs.items()},
            open(os.path.join(root, 'refs.pkl'), 'wb'))
pickle.dump(report, open(os.path.join(root, 'train_report.pkl'), 'wb'))
print('saved models and references')
