"""Controlled two-vacancy separation series in a 5x5 cell.

Two sulfur vacancies are placed at a controlled separation, from an
adjacent divacancy out to the maximum separation the cell allows, to
isolate how the optical fingerprint depends on vacancy arrangement at
fixed vacancy count. A 5x5 cell gives room to separate the pair.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from mos2hamop.structures import supercell, sulfur_indices
from mos2hamop.dftrun import run_structure

OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'sep')
os.makedirs(OUT, exist_ok=True)
NX = NY = 5


def make_pair(sep_rank, rattle=0.02, seed=7):
    atoms = supercell(NX, NY)
    top = sulfur_indices(atoms, 'top')
    pos = atoms.positions
    # anchor near the middle
    cen = pos[:, :2].mean(0)
    i0 = top[np.argmin(np.linalg.norm(pos[top, :2] - cen, axis=1))]
    r0 = pos[i0, :2]
    # sort other top S by distance from the anchor
    others = [t for t in top if t != i0]
    dists = [np.linalg.norm(pos[t, :2] - r0) for t in others]
    order = np.argsort(dists)
    idx = order[min(sep_rank, len(order) - 1)]
    i1 = others[idx]
    sep = float(dists[idx])
    rng = np.random.default_rng(seed)
    for k in sorted([i0, i1], reverse=True):
        del atoms[k]
    atoms.positions += rng.normal(0, rattle, atoms.positions.shape)
    return atoms, sep


if __name__ == '__main__':
    ranks = [0, 2, 5, 9, 14]     # adjacent -> far
    if len(sys.argv) > 1:
        ranks = [int(x) for x in sys.argv[1].split(',')]
    for r in ranks:
        name = f'sep5_{r:02d}'
        if os.path.exists(os.path.join(OUT, name + '.npz')):
            print('skip', name); continue
        atoms, sep = make_pair(r)
        t0 = time.time()
        e = run_structure(atoms, name, OUT)
        print(f'{name}: sep={sep:.2f} A, E={e:.2f} eV, {time.time()-t0:.0f}s',
              flush=True)
    print('separation series done')
