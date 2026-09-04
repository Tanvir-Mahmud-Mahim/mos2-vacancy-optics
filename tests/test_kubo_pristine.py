"""Kubo optics of pristine MoS2 from dense-k DFT blocks; check absorption edge."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.blocks import realspace_matrices, pair_blocks, orbital_offsets
from mos2hamop.kubo import sigma_xx

d = np.load('data/kubo_prim.npz')
pos, num, cell = d['positions'], d['numbers'], d['cell']
mats = realspace_matrices(d['H_kMM'], d['S_kMM'], d['kpts'], (12, 12))
mats_real = {k: (H.real, S.real) for k, (H, S) in mats.items()}
samples = pair_blocks(mats_real, pos, num, cell, (12, 12), rcut=14.0)
offs, nao = orbital_offsets(num)
flat = [(offs[s['i']], offs[s['j']], s['H'].shape[0], s['H'].shape[1],
         s['d'], s['H'], s['S']) for s in samples]

icell = 2*np.pi*np.linalg.inv(cell).T
nk = 18
ks = [np.array([i/nk, j/nk, 0.])@icell for i in range(nk) for j in range(nk)]
wk = np.ones(len(ks))/len(ks)
area = float(np.abs(np.linalg.det(cell[:2, :2])))
ef = float(d['efermi'])
omega = np.linspace(0.1, 3.2, 64)
sig, ne, e_all = sigma_xx(flat, nao, ks, wk, area, omega, mu=ef, eta=0.06)
gaps = [e[e>ef].min()-e[e<ef].max() for e in e_all]
print('area=%.2f A^2  min direct gap over mesh=%.3f eV' % (area, min(gaps)))
for w, s in zip(omega[::6], sig[::6]):
    print('  %.2f eV  sigma=%.4f' % (w, s))
print('max sigma below 1.4 eV:', round(sig[omega<1.4].max(), 4))
print('mean sigma 1.9-2.6 eV:', round(sig[(omega>1.9)&(omega<2.6)].mean(), 4))
