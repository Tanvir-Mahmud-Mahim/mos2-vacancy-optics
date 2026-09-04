"""Figure 4: one Hamiltonian, coupled electronic and optical fingerprints.
(a) in-gap density of states grows with vacancy count.
(b) the near-DC (transport) conductivity switches on with vacancies.
(c) the optical sub-gap absorption tracks the in-gap density of states:
    a contactless optical readout of the electronic degradation.
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

    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))

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
    fig.tight_layout(w_pad=1.3)
    out = os.path.join(ROOT, '..', 'figures', 'fig4_coupling.pdf')
    fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'))
    print('wrote', out)


if __name__ == '__main__':
    main()
