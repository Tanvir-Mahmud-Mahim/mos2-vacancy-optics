"""Graphical abstract: a defective monolayer MoS2 (left) maps through the
same Hamiltonian to a sub-gap optical read-out (right). The key message is
selectivity: at fixed vacancy count the sub-gap brightness ranges from dark
to bright. Annotations are kept out of the data and off the axis labels.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
import atomrender as AR
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def main():
    sp = np.load(os.path.join(os.path.dirname(__file__), '..', 'data',
                              'dft_spectra.npz'))
    omega = sp['omega']; gap0 = float(sp['gap0'])

    fig = plt.figure(figsize=(3.3, 1.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2], wspace=0.42)

    # ---- left: detailed 3D defective MoS2 (a divacancy) ----
    axl = fig.add_subplot(gs[0])
    # pick two nearby upper-S vacancies for a divacancy
    mo, s = AR.cluster(4, 4)
    cx, cy = mo[:, 0].mean(), mo[:, 1].mean()
    up = np.where(s[:, 2] > 0)[0]
    order = up[np.argsort((s[up, 0]-cx)**2 + (s[up, 1]-cy)**2)]
    vac = [int(order[0]), int(order[1])]
    info = AR.render(axl, nx=4, ny=4, vac_sites=vac, elev=26, azim=-54,
                     scale=0.92)
    from matplotlib.patches import Circle as _Circ
    # ring the second vacancy too (render only rings the first)
    v2 = info['s2'][vac[1]]
    axl.add_patch(_Circ((v2[0], v2[1]), AR.R_S * 0.92 * 1.7, fill=False,
                        ec=F.C['red'], lw=1.6, zorder=200))
    x0, x1 = info['xlim']; y0, y1 = info['ylim']
    axl.set_ylim(y0 - 2.0, y1 + 0.9)
    axl.set_title('defective MoS$_2$', fontsize=8, pad=3)
    # key-style label below the structure (no leader line, nothing on atoms)
    cx = 0.5 * (x0 + x1)
    axl.add_patch(_Circ((cx - 2.3, y0 - 1.15), 0.45, fill=False,
                        ec=F.C['red'], lw=1.5, zorder=210))
    axl.text(cx - 1.4, y0 - 1.15, 'S vacancies', color=F.C['red'],
             fontsize=7.4, ha='left', va='center', zorder=210)

    # ---- right: sub-gap absorption ----
    axr = fig.add_subplot(gs[1])
    axr.axvspan(0.15, gap0 - 0.2, color='0.92', lw=0)
    axr.plot(omega, sp['sig_pristine'], color='0.5', lw=1.1, label='pristine')
    nvk = [k for k in sp.files if k.startswith('sig_')
           and k.split('_')[1].isdigit()]
    nvs = sorted(int(k.split('_')[1]) for k in nvk)
    for i, nv in enumerate([n for n in nvs if n > 0][:2]):
        axr.plot(omega, sp[f'sig_{nv}'], color=F.SEQ[i * 2], lw=1.3,
                 label=f'{nv} vac.')
    axr.set_xlim(0.1, 2.4); axr.set_ylim(bottom=0)
    axr.set_xlabel('photon energy (eV)', fontsize=7.5, labelpad=1)
    axr.set_ylabel(r'$\sigma_{xx}$ (arb.)', fontsize=8, labelpad=2)
    axr.tick_params(labelsize=6.5)
    ytop = axr.get_ylim()[1]
    peak = sp['sig_2'][np.argmin(abs(omega - 1.2))]
    axr.annotate('sub-gap\nread-out', xy=(1.2, peak),
                 xytext=(1.35, ytop * 0.5), fontsize=7, color=F.C['red'],
                 ha='left', va='center',
                 arrowprops=dict(arrowstyle='->', color=F.C['red'], lw=0.8))
    axr.legend(fontsize=6, loc='upper right', handlelength=1.0, borderpad=0.2)

    # no connector arrow: the gutter is narrow and an arrow would sit on the
    # sigma_xx axis label; the two panels read left to right on their own

    out = os.path.join(os.path.dirname(__file__), '..', 'figures',
                       'fig0_abstract.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    print('wrote', out)


if __name__ == '__main__':
    main()
