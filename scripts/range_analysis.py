"""Why the spectral read-out needs larger training cells: a quantitative
diagnosis of the pair-range limitation.

Three facts are established, all from the exact DFT blocks (no learning
involved), so they bound any local block-learning approach:

1. Truncating the exact Hamiltonian and overlap at the 11 A pair range
   used for block learning collapses the supercell gap (4x4 and 5x5).
2. Restoring every pair of the Born-von-Karman torus (max 14.8 A in the
   4x4 cell with the 2x2 mesh) restores the eigenvalues to a few meV, so
   nothing beyond the torus is needed.
3. The far blocks (11 A to the torus edge) are Born-von-Karman alias
   sums, not functions of the local pair geometry: a frame-aligned,
   distance-binned reference fitted on the training set captures almost
   none of their variance. They are therefore not learnable from local
   geometry at this cell size, and the remedy is training data whose
   torus contains all physical tails inside the learnable local range
   (larger cells or denser k-meshes), not a better local model.
"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.eigsolve import gen_eigh
from mos2hamop.blocks import realspace_matrices, pair_blocks, orbital_offsets, NAO
from mos2hamop.features import frame_angle
from mos2hamop.mlmodel import rotate_into_frame
from mos2hamop.reference import DistanceReference
from mos2hamop.kubo import bloch_matrices

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
R_LEARN = 11.0


def load_full(fn, kgrid=(2, 2)):
    d = np.load(fn)
    mats = realspace_matrices(d['H_kMM'], d['S_kMM'], d['kpts'], kgrid)
    mr = {k: (H.real, S.real) for k, (H, S) in mats.items()}
    full = pair_blocks(mr, d['positions'], d['numbers'], d['cell'], kgrid,
                       rcut=100.0)
    return d, full


def spectrum_stats(d, sam, label):
    offs, nao = orbital_offsets(d['numbers'])
    flat = [(offs[s['i']], offs[s['j']], s['H'].shape[0], s['H'].shape[1],
             s['d'], s['H'], s['S']) for s in sam]
    icell = 2 * np.pi * np.linalg.inv(d['cell']).T
    ev_dft = d['eigenvalues']; ef = float(d['efermi'])
    num = d['numbers']
    nocc = int(round((14 * (num == 42).sum() + 6 * (num == 16).sum()) / 2))
    gaps, stds = [], []
    for ik, kf in enumerate(d['kpts']):
        H, S, _, _ = bloch_matrices(flat, nao, kf @ icell, None)
        w = gen_eigh(H, S, thresh=1e-4, eigvals_only=True)
        n = min(len(w), ev_dft.shape[1])
        gaps.append(w[nocc] - w[nocc - 1])
        m = (ev_dft[ik][:n] > ef - 3) & (ev_dft[ik][:n] < ef + 3)
        stds.append((w[:n] - ev_dft[ik][:n])[m].std())
    g_dft = float(np.mean([ev_dft[ik][nocc] - ev_dft[ik][nocc - 1]
                           for ik in range(len(d['kpts']))]))
    print(f'  {label:26s} gap {np.mean(gaps):.3f} eV (DFT {g_dft:.3f}), '
          f'near-gap eigenvalue std {np.mean(stds)*1e3:.1f} meV', flush=True)


def main():
    print('== 1+2: truncation collapse and required range (4x4 test cell) ==')
    d, full = load_full(os.path.join(ROOT, 'test', 'test_vac1_001.npz'))
    dists = [s['dist'] for s in full]
    print(f'  torus max pair distance {max(dists):.2f} A, {len(full)} pairs')
    spectrum_stats(d, full, 'all torus pairs')
    spectrum_stats(d, [s for s in full if s['dist'] <= R_LEARN],
                   f'pairs <= {R_LEARN:.0f} A')

    print('== 1b: same collapse in the 5x5 cell ==')
    d5, full5 = load_full(os.path.join(ROOT, 'sep', 'sep5_00.npz'))
    spectrum_stats(d5, full5, 'all torus pairs')
    spectrum_stats(d5, [s for s in full5 if s['dist'] <= R_LEARN],
                   f'pairs <= {R_LEARN:.0f} A')

    print('== 3: the far blocks are not functions of local geometry ==')
    acc = {}
    files = sorted(glob.glob(os.path.join(ROOT, 'samples_train',
                                          '*_samples.npz')))
    names = [os.path.basename(f)[:-len('_samples.npz')] for f in files]
    for name in names:
        fn = os.path.join(ROOT, 'train', name + '.npz')
        if not os.path.exists(fn):
            continue
        dt, ft = load_full(fn)
        for s in ft:
            if s['dist'] <= R_LEARN or s['kind'] != 'pair':
                continue
            th = frame_angle(s['d'])
            Hf = rotate_into_frame(s['H'], s['Zi'], s['Zj'], th)
            acc.setdefault((s['Zi'], s['Zj']), []).append((s['dist'],
                                                          Hf.ravel()))
    for (Zi, Zj), rows in sorted(acc.items()):
        dist = np.array([r[0] for r in rows])
        Y = np.array([r[1] for r in rows])
        if Y.std() < 1e-6:
            print(f'  far ({Zi},{Zj}): {len(rows)} blocks, negligible '
                  f'(std {Y.std()*1e3:.3f} meV)')
            continue
        ref = DistanceReference(dist.min()-.01, dist.max()+.01, 14,
                                NAO[Zi], NAO[Zj])
        ref.fit(dist, Y)
        res = np.sqrt(np.mean((Y - np.array([ref.value(x)[0]
                                             for x in dist])) ** 2))
        print(f'  far ({Zi},{Zj}): {len(rows)} blocks, std '
              f'{Y.std()*1e3:.0f} meV, distance-reference residual '
              f'{res*1e3:.0f} meV '
              f'({100*(1-res**2/Y.std()**2):.0f}% variance explained)')


if __name__ == '__main__':
    main()
