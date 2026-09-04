"""Analyse the controlled two-vacancy separation series: A_sub vs separation."""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.blocks import realspace_matrices, pair_blocks, orbital_offsets
from mos2hamop.kubo import sigma_xx
from mos2hamop.structures import supercell, sulfur_indices

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')


def optics(d, omega, eta=0.08):
    mats = realspace_matrices(d['H_kMM'], d['S_kMM'], d['kpts'], (2, 2))
    mr = {k: (H.real, S.real) for k, (H, S) in mats.items()}
    samples = pair_blocks(mr, d['positions'], d['numbers'], d['cell'], (2, 2),
                          rcut=100.0)
    offs, nao = orbital_offsets(d['numbers'])
    flat = [(offs[s['i']], offs[s['j']], s['H'].shape[0], s['H'].shape[1],
             s['d'], s['H'], s['S']) for s in samples]
    icell = 2 * np.pi * np.linalg.inv(d['cell']).T
    ks = [kf @ icell for kf in d['kpts']]
    wk = np.ones(len(ks)) / len(ks)
    area = float(np.abs(np.linalg.det(d['cell'][:2, :2])))
    sig, ne, e_all = sigma_xx(flat, nao, ks, wk, area, omega,
                              mu=float(d['efermi']), eta=eta)
    return sig, e_all


def vac_separation(d):
    """Separation of the two S vacancies (minimum image) in a 5x5 cell."""
    at0 = supercell(5, 5)
    p0 = at0.positions[at0.numbers == 16]
    pS = d['positions'][d['numbers'] == 16]
    miss = [r for r in p0 if np.min(np.linalg.norm(pS - r, axis=1)) > 0.8]
    if len(miss) != 2:
        return np.nan
    cell = d['cell']
    best = 1e9
    for m1 in (-1, 0, 1):
        for m2 in (-1, 0, 1):
            best = min(best, np.linalg.norm(miss[1] + m1*cell[0] + m2*cell[1]
                                            - miss[0]))
    return float(best)


def main():
    # pristine gap of the 5x5 cell to set the sub-gap window
    omega = np.linspace(0.05, 3.0, 90)
    files = sorted(glob.glob(os.path.join(ROOT, 'sep', '*.npz')))
    if not files:
        print('no separation data'); return
    # reference gap from the widest-separation config (least perturbed)
    out = []
    gaps = []
    for fn in files:
        d = np.load(fn)
        sig, e_all = optics(d, omega)
        num = d['numbers']
        occ = int(round((14*(num == 42).sum() + 6*(num == 16).sum())/2))
        vbm = e_all[:, occ-1].max(); cbm = e_all[:, occ].min()
        gaps.append(cbm - vbm)
    gap0 = max(gaps)
    for fn in files:
        d = np.load(fn)
        sig, e_all = optics(d, omega)
        a_sub = float(np.trapezoid(sig[(omega > 0.15) & (omega < gap0 - 0.2)],
                                   omega[(omega > 0.15) & (omega < gap0 - 0.2)]))
        sep = vac_separation(d)
        out.append(dict(name=os.path.basename(fn)[:-4], sep=sep, a_sub=a_sub))
        print(f'{out[-1]["name"]}: sep={sep:.2f} A_sub={a_sub:.2f}', flush=True)
    json.dump(out, open(os.path.join(ROOT, 'separation.json'), 'w'), indent=1)
    print('separation analysis done')


if __name__ == '__main__':
    main()
