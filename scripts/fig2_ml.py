"""Figure 2: the learned Hamiltonian reproduces the DFT Hamiltonian.
(a) held-out matrix-element parity (ML vs DFT).
(b) block error as a function of interatomic distance.
(c) per-block-type held-out RMSE.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')


def main():
    p = np.load(os.path.join(ROOT, 'ml_parity.npz'))
    rep = json.load(open(os.path.join(ROOT, 'ml_report.json')))
    dft, ml, dist = p['dft'], p['ml'], p['dist']

    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.4))

    # (a) parity
    a = ax[0]
    sel = np.random.default_rng(0).choice(len(dft), min(8000, len(dft)),
                                          replace=False)
    a.plot(dft[sel], ml[sel], '.', ms=1.3, color=F.C['blue'], alpha=0.3,
           rasterized=True)
    lim = [min(dft.min(), ml.min()) - 0.3, max(dft.max(), ml.max()) + 0.3]
    a.plot(lim, lim, '-', color='0.4', lw=0.8)
    # count-weighted overall held-out RMSE (consistent with panel (c))
    tot = sum(v['n_test'] for v in rep.values())
    rms = np.sqrt(sum(v['n_test'] * v['rmse_meV']**2 for v in rep.values()) / tot)
    a.set_xlim(lim); a.set_ylim(lim); a.set_aspect('equal')
    a.set_xlabel(r'$H_{ij}$ DFT (eV)'); a.set_ylabel(r'$H_{ij}$ learned (eV)')
    a.text(0.05, 0.9, f'overall {rms:.0f} meV', transform=a.transAxes, fontsize=7.5)
    F.panel_label(a, '(a)')

    # (b) error vs distance
    b = ax[1]
    err = np.abs(dft - ml)
    bins = np.linspace(0, dist.max(), 22)
    cen = 0.5 * (bins[1:] + bins[:-1])
    mae = [err[(dist >= bins[i]) & (dist < bins[i+1])].mean()
           if ((dist >= bins[i]) & (dist < bins[i+1])).sum() else np.nan
           for i in range(len(bins)-1)]
    b.plot(cen, np.array(mae) * 1e3, 'o-', color=F.C['green'], ms=3.5, lw=1.1,
           mfc='white')
    b.set_xlabel('interatomic distance (Å)')
    b.set_ylabel('block MAE (meV)')
    b.set_ylim(bottom=0)
    F.panel_label(b, '(b)')

    # (c) per-type RMSE
    c = ax[2]
    names = {'onsite_42_42': 'Mo\nonsite', 'onsite_16_16': 'S\nonsite',
             'pair_42_42': 'Mo-Mo', 'pair_42_16': 'Mo-S', 'pair_16_16': 'S-S'}
    tags = [t for t in ['onsite_42_42', 'onsite_16_16', 'pair_42_42',
                        'pair_42_16', 'pair_16_16'] if t in rep]
    vals = [rep[t]['rmse_meV'] for t in tags]
    cols = [F.C['purple'], F.C['purple'], F.C['orange'], F.C['blue'],
            F.C['teal']]
    c.bar(range(len(tags)), vals, color=cols[:len(tags)], width=0.65,
          edgecolor='k', linewidth=0.4)
    c.set_xticks(range(len(tags)))
    c.set_xticklabels([names[t] for t in tags], fontsize=6.8)
    c.set_ylabel('held-out block RMSE (meV)')
    c.set_ylim(0, max(vals) * 1.16)          # headroom so value labels never clip
    for i, v in enumerate(vals):
        c.text(i, v + max(vals)*0.03, f'{v:.0f}', ha='center', fontsize=6.6)
    F.panel_label(c, '(c)')

    fig.tight_layout(w_pad=1.3)
    out = os.path.join(ROOT, '..', 'figures', 'fig2_ml.pdf')
    fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'))
    print('wrote', out)


if __name__ == '__main__':
    main()
