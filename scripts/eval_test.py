"""Evaluate trained models on held-out structures: matrix and eigenvalue errors."""
import os, sys, glob, pickle, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.eigsolve import gen_eigh
from mos2hamop.blocks import realspace_matrices, orbital_offsets
from mos2hamop.mlmodel import BlockModel
from mos2hamop.reference import DistanceReference
from mos2hamop.assemble import predict_blocks, hermitize
from mos2hamop.kubo import bloch_matrices

root = os.path.join(os.path.dirname(__file__), '..', 'data')
with open(os.path.join(root, 'models.pkl'), 'rb') as f:
    states = pickle.load(f)
models = {}
for key, (stH, stS) in states.items():
    mH = BlockModel(len(stH['x_mean']), stH['ni'], stH['nj']); mH.load(stH)
    mS = BlockModel(len(stS['x_mean']), stS['ni'], stS['nj']); mS.load(stS)
    models[key] = (mH, mS)
refs = None
_rp = os.path.join(root, 'refs.pkl')
if os.path.exists(_rp):
    _rs = pickle.load(open(_rp, 'rb'))
    refs = {k: (DistanceReference.load(a), DistanceReference.load(b))
            for k, (a, b) in _rs.items()}

def load_flat_dft(d):
    """Reference real-space blocks straight from the DFT matrices."""
    from mos2hamop.blocks import realspace_matrices, pair_blocks
    mats = realspace_matrices(d['H_kMM'], d['S_kMM'], d['kpts'], (2, 2))
    mr = {k: (H.real, S.real) for k, (H, S) in mats.items()}
    return pair_blocks(mr, d['positions'], d['numbers'], d['cell'], (2, 2),
                       rcut=11.0)


results = {}
parity_H_ml, parity_H_dft = [], []
eig_ml_stack, eig_dft_stack, ef_stack = [], [], []
for fn in sorted(glob.glob(os.path.join(root, 'test', '*.npz'))):
    name = os.path.basename(fn)[:-4]
    d = np.load(fn)
    pos, num, cell = d['positions'], d['numbers'], d['cell']
    images = [(m1, m2) for m1 in (-1, 0, 1) for m2 in (-1, 0, 1)]
    blocks_by_img, nao = predict_blocks(pos, num, cell, images, models, (2, 2), refs)
    blocks_by_img = hermitize(blocks_by_img)
    flat = [b for lst in blocks_by_img.values() for b in lst]
    # eigenvalues at the DFT k-points (fractional -> cartesian)
    icell = 2 * np.pi * np.linalg.inv(cell).T
    errs = []
    ev_ml_all, ev_dft_all = [], []
    for ik, kfrac in enumerate(d['kpts']):
        kcart = kfrac @ icell
        H, S, _, _ = bloch_matrices(flat, nao, kcart, cell)
        # canonical orthogonalization: the learned overlap can be mildly
        # non-positive-definite in the near-null overcomplete directions
        w = gen_eigh(H, S, thresh=1e-4, eigvals_only=True)
        ev_dft = d['eigenvalues'][ik]
        n = len(ev_dft)
        errs.append(np.abs(w[:n] - ev_dft))
        ev_ml_all.append(w[:n]); ev_dft_all.append(ev_dft)
    errs = np.array(errs)
    ef = float(d['efermi'])
    ev_dft_all = np.array(ev_dft_all); ev_ml_all = np.array(ev_ml_all)
    win = (ev_dft_all > ef - 2) & (ev_dft_all < ef + 2)
    results[name] = dict(
        mae_all=float(errs.mean()), mae_win=float(errs[win].mean()),
        max_win=float(errs[win].max()), efermi=ef, natoms=int(len(num)),
        nvac=int(32 - (num == 16).sum()) if (num == 42).sum() == 16 else -1)
    print(name, {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in results[name].items()}, flush=True)
    eig_ml_stack.append(ev_ml_all - ef); eig_dft_stack.append(ev_dft_all - ef)
    ef_stack.append(ef)
    # matrix-element parity: ML vs DFT nearest-neighbour Mo-S blocks
    dft_samples = load_flat_dft(d)
    from mos2hamop.features import frame_angle
    from mos2hamop.mlmodel import type_key, rotate_into_frame
    # collect ML predictions in the same (i,j,d) order as DFT samples
    ml_lookup = {}
    for (oi, oj, ni, nj, dd, Hb, Sb) in flat:
        ml_lookup[(oi, oj, tuple(np.round(dd, 4)))] = Hb
    from mos2hamop.assemble import orbital_offsets as _oo
    offs, _ = _oo(num)
    for s in dft_samples:
        if s['dist'] > 4.0:
            continue
        key = (offs[s['i']], offs[s['j']], tuple(np.round(s['d'], 4)))
        if key in ml_lookup:
            parity_H_dft.append(s['H'].ravel())
            parity_H_ml.append(ml_lookup[key].ravel())

json.dump(results, open(os.path.join(root, 'eval_test.json'), 'w'), indent=1)
if parity_H_dft:
    np.savez_compressed(os.path.join(root, 'eval_parity.npz'),
                        H_dft=np.concatenate(parity_H_dft),
                        H_ml=np.concatenate(parity_H_ml))
np.savez_compressed(os.path.join(root, 'eval_eigs.npz'),
                    ml=np.concatenate([e.ravel() for e in eig_ml_stack]),
                    dft=np.concatenate([e.ravel() for e in eig_dft_stack]))
print('saved eval arrays')
