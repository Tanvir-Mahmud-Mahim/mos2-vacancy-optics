"""Figure 4: one Hamiltonian, coupled electronic and optical fingerprints.
(a) in-gap density of states grows with vacancy count.
(b) the near-DC (transport) conductivity switches on with vacancies.
(c) a dilute, isolated divacancy is optically dark at every separation.
(d) the quantum mechanism: bringing the pair to contact splits the
    mid-gap levels (hybridization of the defect wavefunctions).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')


def main():
    an = json.load(open(os.path.join(ROOT, 'dft_analysis.json')))
    res = an['results']

    fig, ax = plt.subplots(1, 4, figsize=(7.1, 2.1))

    # (a) in-gap DOS vs density
    a = ax[0]
    byv = {}
    for r in res:
        byv.setdefault(r['nvac'], []).append(r['n_ingap'])
    xs = sorted(byv); dens = [nv/32*100 for nv in xs]
    mean = [np.mean(byv[nv]) for nv in xs]; err = [np.std(byv[nv]) for nv in xs]
    a.errorbar(dens, mean, yerr=err, fmt='o-', color=F.C['purple'], ms=4,
               lw=1.1, capsize=2, mfc='white')
    a.set_xlabel('S-vacancy density (%)')
    a.set_ylabel('in-gap states per $k$-point')
    a.set_ylim(bottom=0)
    F.panel_label(a, '(a)')

    # (b) near-DC conductivity vs density
    b = ax[1]
    byd = {}
    for r in res:
        byd.setdefault(r['nvac'], []).append(r['sig_dc'])
    xs = sorted(byd); dens = [nv/32*100 for nv in xs]
    for nv, dn in zip(xs, dens):
        b.plot([dn]*len(byd[nv]), byd[nv], 'o', color=F.C['teal'], ms=3,
               alpha=0.5, mec='none')
    b.plot(dens, [np.mean(byd[nv]) for nv in xs], 's-', color=F.C['navy'],
           ms=4, lw=1.1, mfc='white', label='mean')
    b.set_xlabel('S-vacancy density (%)')
    b.set_ylabel(r'near-DC $\sigma(\omega\!\to\!0)\;(e^2/4\hbar)$')
    b.set_ylim(bottom=0)
    b.legend(fontsize=7.5, loc='upper left')
    F.panel_label(b, '(b)')

    # (c) a dilute, isolated divacancy is optically dark
    c = ax[2]
    import json as _json
    sepf = os.path.join(ROOT, 'separation.json')
    if os.path.exists(sepf):
        sd = _json.load(open(sepf))
        seps = np.array([p['sep'] for p in sd]); asb = np.array([p['a_sub'] for p in sd])
        o = np.argsort(seps)
        c.plot(seps[o], asb[o], 'o-', color=F.C['red'], ms=5, lw=1.2, mfc='white')
    c.axhline(0, color='0.7', lw=0.6)
    c.set_xlabel('vacancy separation (Å)')
    c.set_ylabel(r'sub-gap absorption $A_{\mathrm{sub}}$')
    c.set_title('dilute divacancy', fontsize=8, pad=2)
    c.text(0.5, 0.5, 'optically dark\nwhen isolated', transform=c.transAxes,
           ha='center', fontsize=7.5, color='0.35')
    F.panel_label(c, '(c)')

    # (d) hybridization splitting: width of the mid-gap manifold vs
    # separation of the same dilute pair (same numbers as
    # hybridization_split.py)
    d = ax[3]
    import glob as _glob
    vbm0, cbm0 = an['vbm0'], an['cbm0']
    sepmap = {p['name']: p['sep'] for p in sd}
    pts = []
    for fn in sorted(_glob.glob(os.path.join(ROOT, 'sep', 'sep5_*.npz'))):
        name = os.path.basename(fn)[:-4]
        ev = np.load(fn)['eigenvalues']
        g = np.concatenate([e[(e > vbm0 + 0.1) & (e < cbm0 - 0.1)]
                            for e in ev])
        pts.append((sepmap[name], float(g.max() - g.min())))
    pts.sort()
    d.plot([p[0] for p in pts], [p[1] for p in pts], 'D',
           color=F.C['purple'], ms=5, mec='none')
    d.set_xlabel('vacancy separation (Å)')
    d.set_ylabel('mid-gap manifold width (eV)')
    d.set_title('level splitting', fontsize=8, pad=2)
    F.panel_label(d, '(d)')
    fig.tight_layout(w_pad=1.3)
    out = os.path.join(ROOT, '..', 'figures', 'fig4_coupling.pdf')
    fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'))
    print('wrote', out)


if __name__ == '__main__':
    main()
