"""Figure 3: optical fingerprint of sulfur vacancies (DFT).
(a) sigma_xx(omega): a sub-gap absorption band grows with vacancy count.
(b) sub-gap absorption A_sub versus vacancy density (with per-config scatter).
(c) A_sub is set by arrangement too: it rises as the two vacancies of a
    fixed-count pair are brought together (separation series).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')


def main():
    sp = np.load(os.path.join(ROOT, 'dft_spectra.npz'))
    an = json.load(open(os.path.join(ROOT, 'dft_analysis.json')))
    res = an['results']
    gap0 = an['gap0']; vbm0 = an['vbm0']; cbm0 = an['cbm0']
    omega = sp['omega']

    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))

    # (a) spectra by vacancy count
    a = ax[0]
    nvs = sorted(int(k.split('_')[1]) for k in sp.files if k.startswith('sig_') and k.split('_')[1].isdigit())
    a.axvspan(0.15, gap0 - 0.2, color='0.88', lw=0, zorder=0)
    a.plot(omega, sp['sig_pristine'], color='0.35', lw=1.3, label='pristine')
    for i, nv in enumerate(nvs):
        if nv == 0:
            continue
        dens = nv / 32 * 100
        a.plot(omega, sp[f'sig_{nv}'], color=F.SEQ[i % len(F.SEQ)], lw=1.2,
               label=f'{nv} vac ({dens:.0f}%)')
    a.set_xlabel('photon energy (eV)')
    a.set_ylabel(r'$\sigma_{xx}\;(e^2/4\hbar)$')
    a.set_xlim(0.1, 2.6); a.set_ylim(bottom=0)
    a.text((0.15 + gap0 - 0.2) / 2, a.get_ylim()[1]*0.92, 'sub-gap',
           ha='center', fontsize=7, color='0.3')
    a.legend(fontsize=6.6, loc='upper right', handlelength=1.1)
    F.panel_label(a, '(a)')

    # (b) A_sub vs density
    b = ax[1]
    byv = {}
    for r in res:
        if r['nvac'] == 0 and r['a_sub'] > 5:
            continue  # skip strongly rattled pristine outliers
        byv.setdefault(r['nvac'], []).append(r['a_sub'])
    xs = sorted(byv)
    dens = [nv / 32 * 100 for nv in xs]
    for nv, dn in zip(xs, dens):
        ys = byv[nv]
        b.plot([dn]*len(ys), ys, 'o', color=F.C['blue'], ms=3.2, alpha=0.5,
               mec='none')
    mean = [np.mean(byv[nv]) for nv in xs]
    b.plot(dens, mean, 's-', color=F.C['navy'], ms=4, lw=1.1, mfc='white',
           label='mean')
    b.set_xlabel('S-vacancy density (%)')
    b.set_ylabel(r'sub-gap absorption $A_{\mathrm{sub}}$')
    b.legend(fontsize=7.5, loc='upper left')
    F.panel_label(b, '(b)')

    # (c) optical brightness decouples from the mid-gap state count
    c = ax[2]
    xi = np.array([r['n_ingap'] for r in res])
    ya = np.array([r['a_sub'] for r in res])
    nv = np.array([r['nvac'] for r in res])
    sc = c.scatter(xi, ya, c=nv, cmap='viridis', s=24, edgecolors='k',
                   linewidths=0.3)
    cb = fig.colorbar(sc, ax=c, pad=0.02, fraction=0.05)
    cb.set_label('vacancies', fontsize=7.5); cb.ax.tick_params(labelsize=7)
    # correlation over ALL configurations, identical to gen_numbers.py, so
    # the figure and the text always quote the same value
    rr = np.corrcoef(xi, ya)[0, 1]
    c.text(0.05, 0.92, f'$r={rr:.2f}$', transform=c.transAxes, fontsize=8)
    c.text(0.05, 0.80, 'same count,\ndark or bright', transform=c.transAxes,
           fontsize=7, color='0.3')
    c.set_xlabel('in-gap states per $k$-point')
    c.set_ylabel(r'sub-gap absorption $A_{\mathrm{sub}}$')
    F.panel_label(c, '(c)')
    fig.tight_layout(w_pad=1.2)
    out = os.path.join(ROOT, '..', 'figures', 'fig3_optics.pdf')
    fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'))
    print('wrote', out)


if __name__ == '__main__':
    main()
