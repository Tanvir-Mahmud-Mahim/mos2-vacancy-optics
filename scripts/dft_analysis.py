"""Coupled optical and electronic fingerprint of sulfur vacancies (DFT).

For every vacancy configuration this reads the exact DFT H(k), S(k) and
computes, on the exact k-points (where H and S are exact and positive
definite):
  * the Kubo optical conductivity sigma_xx(omega);
  * the sub-gap absorption A_sub (integrated conductivity below the
    pristine gap);
  * the near-DC conductivity sigma(omega -> 0) as a transport proxy;
  * the in-gap density of states (states inside the pristine gap);
  * a vacancy-arrangement metric (mean nearest-neighbour vacancy
    separation), to test whether the optical fingerprint sees the
    defect arrangement and not only the count.
Real-space blocks are rebuilt at the full Born-von-Karman range so the
Hamiltonian and its k-derivative are exact.
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.blocks import realspace_matrices, pair_blocks, orbital_offsets
from mos2hamop.kubo import sigma_xx
from mos2hamop.structures import make_structure

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
NX = NY = 4


def flat_from_dft(d):
    mats = realspace_matrices(d['H_kMM'], d['S_kMM'], d['kpts'], (2, 2))
    mr = {k: (H.real, S.real) for k, (H, S) in mats.items()}
    samples = pair_blocks(mr, d['positions'], d['numbers'], d['cell'], (2, 2),
                          rcut=100.0)
    offs, nao = orbital_offsets(d['numbers'])
    flat = [(offs[s['i']], offs[s['j']], s['H'].shape[0], s['H'].shape[1],
             s['d'], s['H'], s['S']) for s in samples]
    return flat, nao


def optics(d, omega, eta=0.08):
    num = d['numbers']; cell = d['cell']
    flat, nao = flat_from_dft(d)
    icell = 2 * np.pi * np.linalg.inv(cell).T
    ks = [kf @ icell for kf in d['kpts']]
    wk = np.ones(len(ks)) / len(ks)
    area = float(np.abs(np.linalg.det(cell[:2, :2])))
    ef = float(d['efermi'])
    sig, ne, e_all = sigma_xx(flat, nao, ks, wk, area, omega, mu=ef, eta=eta)
    return sig, e_all, ef


def vacancy_positions(d):
    """Recover which S sites are missing by comparing to the pristine lattice."""
    num = d['numbers']
    at0, _ = make_structure(NX, NY, n_vac=0, rattle=0.0)
    p0 = at0.positions[at0.numbers == 16]
    # present S positions
    pS = d['positions'][num == 16]
    missing = []
    for r in p0:
        if np.min(np.linalg.norm(pS - r, axis=1)) > 0.8:
            missing.append(r)
    return np.array(missing)


def clustering_metric(d, cell):
    vpos = vacancy_positions(d)
    if len(vpos) < 2:
        return np.nan
    dists = []
    for i in range(len(vpos)):
        best = 1e9
        for j in range(len(vpos)):
            if i == j:
                continue
            for m1 in (-1, 0, 1):
                for m2 in (-1, 0, 1):
                    dd = np.linalg.norm(vpos[j] + m1*cell[0] + m2*cell[1] - vpos[i])
                    best = min(best, dd)
        dists.append(best)
    return float(np.mean(dists))


def main():
    omega = np.linspace(0.05, 3.0, 90)
    dp = np.load(os.path.join(ROOT, 'train', 'prist_000.npz'))
    sig_p, e_p, ef_p = optics(dp, omega)
    num_p = dp['numbers']
    occ_p = int(round((14*(num_p == 42).sum() + 6*(num_p == 16).sum())/2))
    vbm0 = e_p[:, occ_p-1].max(); cbm0 = e_p[:, occ_p].min(); gap0 = cbm0 - vbm0
    print(f'pristine gap {gap0:.3f} eV', flush=True)

    results = []
    specs = {}
    for fn in sorted(glob.glob(os.path.join(ROOT, 'train', '*.npz'))):
        name = os.path.basename(fn)[:-4]
        d = np.load(fn)
        num = d['numbers']
        if (num == 42).sum() != 16:
            continue
        nvac = int(32 - (num == 16).sum())
        sig, e_all, ef = optics(d, omega)
        hi = gap0 - 0.2
        a_sub = float(np.trapezoid(sig[(omega > 0.15) & (omega < hi)],
                                   omega[(omega > 0.15) & (omega < hi)]))
        sig_dc = float(sig[np.argmin(np.abs(omega - 0.1))])
        e_in = e_all[(e_all > vbm0 + 0.1) & (e_all < cbm0 - 0.1)]
        n_ingap = float(e_in.size / e_all.shape[0])
        clus = clustering_metric(d, d['cell']) if nvac >= 2 else np.nan
        results.append(dict(name=name, nvac=nvac, dens=nvac / 32 * 100,
                            a_sub=a_sub, sig_dc=sig_dc, n_ingap=n_ingap,
                            cluster=clus))
        specs.setdefault(nvac, []).append(sig)
        print(f'{name}: nv={nvac} A_sub={a_sub:.2f} sig_dc={sig_dc:.3f} '
              f'n_ingap={n_ingap:.2f} clus={clus}', flush=True)

    json.dump(dict(gap0=float(gap0), vbm0=float(vbm0), cbm0=float(cbm0),
                   results=results),
              open(os.path.join(ROOT, 'dft_analysis.json'), 'w'), indent=1)
    # mean spectrum per vacancy count (representative, averaged over configs)
    save = dict(omega=omega, gap0=gap0, vbm0=vbm0, cbm0=cbm0, sig_pristine=sig_p)
    for nv, sgs in specs.items():
        save[f'sig_{nv}'] = np.mean(sgs, axis=0)
    np.savez_compressed(os.path.join(ROOT, 'dft_spectra.npz'), **save)
    print('DFT analysis done')


if __name__ == '__main__':
    main()
