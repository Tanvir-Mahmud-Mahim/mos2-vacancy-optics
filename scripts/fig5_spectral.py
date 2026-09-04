"""Figure 5: the learned Hamiltonian read out spectrally (Gamma 5x5).
(a) Kubo optical conductivity of the five held-out structures: exact DFT
    versus the fully learned H and S assembled from geometry alone (both
    read out in the common well-conditioned subspace).
(b) Sub-gap absorption of the learned operator against the exact DFT
    value: the brightness ordering is reproduced.
(c) Gap parity, learned versus exact DFT.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')


def main():
    sp = np.load(os.path.join(ROOT, 'spectral_validation.npz'))
    an = json.load(open(os.path.join(ROOT, 'spectral_validation.json')))
    res = an['results']
    gap0 = an['gap0']
    omega = sp['omega']

    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.5),
                           gridspec_kw={'width_ratios': [1.45, 1, 1]})

    # (a) spectra in the sub-gap window: exact DFT solid, learned dashed,
    # offset by structure
    a = ax[0]
    wlo, whi = 0.10, gap0 - 0.1
    a.axvspan(0.15, gap0 - 0.2, color='0.92', lw=0, zorder=0)
    order = np.argsort([r['nvac'] for r in res])
    mwin = (omega >= wlo) & (omega <= whi)
    step = 0.0
    for idx in order:
        name = res[idx]['name']
        step = max(step, float(sp[f'exact_{name}'][mwin].max()),
                   float(sp[f'ml_{name}'][mwin].max()))
    step *= 1.12
    yticks, ylabels = [], []
    off = 0.0
    for idx in order:
        r = res[idx]; name = r['name']
        a.plot(omega[mwin], sp[f'exact_{name}'][mwin] + off,
               color='0.25', lw=1.2)
        a.plot(omega[mwin], sp[f'ml_{name}'][mwin] + off,
               color=F.C['red'], lw=1.0, ls='--')
        yticks.append(off)
        ylabels.append(f"{r['nvac']} vac")
        off += step
    a.plot([], [], color='0.25', lw=1.2, label='DFT')
    a.plot([], [], color=F.C['red'], lw=1.0, ls='--', label='learned')
    a.set_xlabel('photon energy (eV)')
    a.set_ylabel(r'$\sigma_{xx}\;(e^2/4\hbar)$, offset')
    a.set_xlim(wlo, whi)
    a.set_ylim(-0.08 * step, off + 0.55 * step)
    a.set_yticks(yticks); a.set_yticklabels(ylabels, fontsize=7)
    a.legend(fontsize=7, loc='upper right', handlelength=1.4)
    a.text(0.03, 0.965, 'sub-gap window', transform=a.transAxes,
           fontsize=7, color='0.35', va='top')
    F.panel_label(a, '(a)')

    nv = np.array([r['nvac'] for r in res])
    colors = [F.SEQ[min(v * 2, len(F.SEQ) - 1)] for v in nv]

    # (b) A_sub: learned vs exact DFT
    b = ax[1]
    xd = np.array([r['a_sub_exact'] for r in res])
    xm = np.array([r['a_sub_ml'] for r in res])
    lim = 1.12 * max(xd.max(), xm.max())
    b.plot([0, lim], [0, lim], color='0.7', lw=0.8, zorder=0)
    for x, y, c in zip(xd, xm, colors):
        b.plot(x, y, 'o', color=c, ms=5, mec='none')
    hs = [plt.Line2D([], [], marker='o', ls='none', ms=4.5, mec='none',
                     color=F.SEQ[min(v * 2, len(F.SEQ) - 1)],
                     label=f'{v} vac')
          for v in sorted(set(nv))]
    b.legend(handles=hs, fontsize=6.6, loc='upper left',
             bbox_to_anchor=(0.02, 0.86), handlelength=0.8)
    b.set_xlabel(r'$A_{\mathrm{sub}}$ DFT (eV$\,e^2/4\hbar$)')
    b.set_ylabel(r'$A_{\mathrm{sub}}$ learned')
    b.set_xlim(0, lim); b.set_ylim(0, lim)
    b.text(0.05, 0.90, f"$r={an['pearson_exact']:.2f}$",
           transform=b.transAxes, fontsize=8)
    F.panel_label(b, '(b)')

    # (c) gap parity, learned vs exact DFT
    c = ax[2]
    gd = np.array([r['gap_exact'] for r in res])
    gm = np.array([r['gap_ml'] for r in res])
    lo = 0.88 * min(gd.min(), gm.min())
    hi = 1.06 * max(gd.max(), gm.max())
    c.plot([lo, hi], [lo, hi], color='0.7', lw=0.8, zorder=0)
    for x, y, col in zip(gd, gm, colors):
        c.plot(x, y, 's', color=col, ms=5, mec='none')
    mae = np.abs(gm - gd).mean()
    c.text(0.05, 0.90, f'MAE {mae*1e3:.0f} meV', transform=c.transAxes,
           fontsize=8)
    c.set_xlabel('gap DFT (eV)')
    c.set_ylabel('gap learned (eV)')
    c.set_xlim(lo, hi); c.set_ylim(lo, hi)
    F.panel_label(c, '(c)')

    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..', 'figures',
                       'fig5_spectral.pdf')
    fig.savefig(out)
    print('wrote', out)


if __name__ == '__main__':
    main()
