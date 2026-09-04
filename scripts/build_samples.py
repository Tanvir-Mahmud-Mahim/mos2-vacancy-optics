"""Convert DFT npz files into ML training arrays per block type."""
import os, sys, glob, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.blocks import realspace_matrices, pair_blocks, NAO
from mos2hamop.features import pair_descriptor, frame_angle
from mos2hamop.mlmodel import type_key, rotate_into_frame

which = sys.argv[1] if len(sys.argv) > 1 else 'train'
indir = os.path.join(os.path.dirname(__file__), '..', 'data', which)
outdir = os.path.join(os.path.dirname(__file__), '..', 'data', 'samples_' + which)
os.makedirs(outdir, exist_ok=True)
KGRID = (2, 2)

for fn in sorted(glob.glob(os.path.join(indir, '*.npz'))):
    name = os.path.basename(fn)[:-4]
    out = os.path.join(outdir, name + '_samples.npz')
    if os.path.exists(out):
        print('skip', name); continue
    t0 = time.time()
    d = np.load(fn)
    mats_R = realspace_matrices(d['H_kMM'], d['S_kMM'], d['kpts'], KGRID)
    # sanity: imaginary parts of H(R) must vanish (real basis functions)
    im = max(abs(H.imag).max() for H, S in mats_R.values())
    samples = pair_blocks({k: (H.real, S.real) for k, (H, S) in mats_R.items()},
                          d['positions'], d['numbers'], d['cell'], KGRID)
    buckets = {}
    for s in samples:
        key = type_key(s['kind'], s['Zi'], s['Zj'])
        if key is None:
            continue
        theta = frame_angle(s['d'])
        Hf = rotate_into_frame(s['H'], s['Zi'], s['Zj'], theta)
        Sf = rotate_into_frame(s['S'], s['Zi'], s['Zj'], theta)
        x = pair_descriptor(s, d['positions'], d['numbers'], d['cell'], KGRID)
        b = buckets.setdefault(key, dict(X=[], H=[], S=[], meta=[]))
        b['X'].append(x); b['H'].append(Hf.ravel()); b['S'].append(Sf.ravel())
        b['meta'].append([s['i'], s['j'], s['dist']])
    save = dict(imag_max=im)
    for key, b in buckets.items():
        tag = f'{key[0]}_{key[1]}_{key[2]}'
        save['X_' + tag] = np.array(b['X'], dtype=np.float32)
        save['H_' + tag] = np.array(b['H'], dtype=np.float64)
        save['S_' + tag] = np.array(b['S'], dtype=np.float64)
        save['meta_' + tag] = np.array(b['meta'])
    np.savez_compressed(out, **save)
    print(f'{name}: {len(samples)} pair samples, imag_max={im:.2e}, '
          f'{time.time()-t0:.0f} s', flush=True)
print('done')
