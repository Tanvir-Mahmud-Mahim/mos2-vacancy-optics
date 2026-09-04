"""Numerical validation of the real-space transform and z rotations."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.blocks import realspace_matrices, reconstruct_k, pair_blocks
from mos2hamop.rotations import rotate_block
from mos2hamop.features import frame_angle
from mos2hamop.mlmodel import rotate_into_frame

d = np.load(os.path.join(os.path.dirname(__file__), '..',
                         'data', 'train', 'prist_000.npz'))
H_kMM, S_kMM, kpts = d['H_kMM'], d['S_kMM'], d['kpts']
mats = realspace_matrices(H_kMM, S_kMM, kpts, (2, 2))

# 1) round trip
err = 0.0
for ik, k in enumerate(kpts):
    Hk, Sk = reconstruct_k(mats, k)
    err = max(err, abs(Hk - H_kMM[ik]).max(), abs(Sk - S_kMM[ik]).max())
print('roundtrip max err:', err)

# 2) reality of H(R)
im = max(abs(H.imag).max() for H, S in mats.values())
print('max imag part of H(R):', im)

# 3) Hermiticity across blocks: H_ij(d) = H_ji(-d)^T
mats_real = {k: (H.real, S.real) for k, (H, S) in mats.items()}
samples = pair_blocks(mats_real, d['positions'], d['numbers'], d['cell'], (2, 2))
bykey = {}
for s in samples:
    bykey[(s['i'], s['j'], tuple(np.round(s['d'], 5)))] = s
herr = 0.0
for (i, j, dd), s in bykey.items():
    rev = bykey.get((j, i, tuple(np.round(-np.array(dd), 5))))
    if rev is not None:
        herr = max(herr, abs(s['H'] - rev['H'].T).max())
print('Hermiticity pair err:', herr)

# 4) rotation convention: symmetry-equivalent Mo-S pairs must give the
# same frame-aligned block (pristine cell has C3 symmetry about each Mo)
mo0 = 0
Zn = d['numbers']
cands = [s for s in samples if s['i'] == mo0 and s['Zj'] == 16
         and abs(s['dist'] - min(x['dist'] for x in samples
                                 if x['i'] == mo0 and x['Zj'] == 16)) < 1e-4
         and s['d'][2] > 0]
print('nearest top-S neighbors of Mo0:', len(cands),
      [np.round(s['d'][:2], 3) for s in cands])
frames = []
for s in cands:
    th = frame_angle(s['d'])
    frames.append(rotate_into_frame(s['H'], 42, 16, th))
for a in range(1, len(frames)):
    print(f'frame-aligned block diff vs first: '
          f'{abs(frames[a] - frames[0]).max():.3e}')
# also distant pair type Mo-Mo
candm = [s for s in samples if s['i'] == mo0 and s['Zj'] == 42
         and 3.0 < s['dist'] < 3.3]
framem = []
for s in candm:
    th = frame_angle(s['d'])
    framem.append(rotate_into_frame(s['H'], 42, 42, th))
for a in range(1, len(framem)):
    print(f'Mo-Mo aligned diff vs first: {abs(framem[a]-framem[0]).max():.3e}')
print('block scale (max |H| Mo-S):', abs(frames[0]).max())
