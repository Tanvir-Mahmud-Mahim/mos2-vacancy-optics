"""Figure 1: concept. A detailed 3D monolayer MoS2 with a sulfur vacancy,
the single density-functional / learned Hamiltonian operator, and its two
read-outs (electronic structure and Kubo optics). All annotations are
placed in clear space with leader lines, so no label overlaps the data.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import figstyle as F
import atomrender as AR
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


def legend_sphere(ax, x, y, r, base):
    AR._shaded_sphere(ax, x, y, r, np.array(base), zorder=50)


def main():
    fig = plt.figure(figsize=(7.1, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.86, 1.12], wspace=0.06)

    # ---- (a) detailed 3D structure ----
    axa = fig.add_subplot(gs[0])
    info = AR.render(axa, nx=4, ny=4, elev=26, azim=-54, scale=1.0)
    x0, x1 = info['xlim']; y0, y1 = info['ylim']
    # headroom above the structure for a dedicated key (no label touches data)
    axa.set_ylim(y0, y1 + 5.0)
    axa.set_xlim(x0 - 0.3, x1 + 0.3)

    # ---- key (top-right): a real legend, so the frame always fits ----
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker='o', ls='none', ms=7.5, mfc='#6c5ce7',
               mec='k', mew=0.4),
        Line2D([], [], marker='o', ls='none', ms=5.5, mfc='#f4c542',
               mec='k', mew=0.3),
        Line2D([], [], marker='o', ls='none', ms=8, mfc='none',
               mec=F.C['red'], mew=1.6),
    ]
    leg = axa.legend(handles, ['Mo', 'S', 'S vacancy'], loc='upper right',
                     frameon=True, fontsize=8, borderpad=0.55,
                     handletextpad=0.45, labelspacing=0.55,
                     borderaxespad=0.15)
    fr = leg.get_frame()
    fr.set_facecolor('#f7f7f4'); fr.set_edgecolor('0.6'); fr.set_linewidth(0.7)
    leg.get_texts()[2].set_color(F.C['red'])
    leg.set_zorder(60)

    axa.text(0.0, 1.0, '(a)', transform=axa.transAxes, fontsize=10,
             fontweight='bold', va='top', ha='left')
    axa.text(0.5, -0.02, r'disordered monolayer MoS$_2$', ha='center',
             va='top', transform=axa.transAxes, fontsize=8.5)
    # imshow spheres above reset the aspect to 'auto'; restore equal so the
    # atoms render as true circles, not ellipses
    axa.set_aspect('equal')

    # ---- middle: the operator ----
    axm = fig.add_subplot(gs[1]); axm.axis('off')
    axm.set_xlim(0, 1); axm.set_ylim(0, 1)
    box = FancyBboxPatch((0.13, 0.42), 0.70, 0.24,
                         boxstyle='round,pad=0.02,rounding_size=0.04',
                         fc='#eaf1f7', ec=F.C['navy'], lw=1.3)
    axm.add_patch(box)
    axm.text(0.5, 0.545, 'density-functional /\nlearned Hamiltonian',
             ha='center', va='center', fontsize=8.4, color=F.C['navy'])
    axm.text(0.5, 0.30, r'$\{H_{ij},\,S_{ij}\}=\mathcal{N}(\mathrm{geometry})$',
             ha='center', va='center', fontsize=8.0)
    axm.annotate('', xy=(0.10, 0.54), xytext=(-0.18, 0.54),
                 arrowprops=dict(arrowstyle='-|>', color=F.C['gray'], lw=1.4))
    axm.text(0.5, 0.86, 'one model,\ntwo observables', ha='center', fontsize=8,
             color=F.C['gray'], style='italic')
    axm.annotate('', xy=(1.16, 0.80), xytext=(0.85, 0.57),
                 arrowprops=dict(arrowstyle='-|>', color=F.C['blue'], lw=1.4))
    axm.annotate('', xy=(1.16, 0.24), xytext=(0.85, 0.51),
                 arrowprops=dict(arrowstyle='-|>', color=F.C['orange'], lw=1.4))

    # ---- right: two schematic read-outs ----
    gsr = gs[2].subgridspec(2, 1, hspace=0.62)

    # (b) electronic structure
    axt = fig.add_subplot(gsr[0])
    E = np.linspace(-1.4, 1.4, 400)
    val = 1.35 * np.exp(-((E + 0.9) / 0.26) ** 2)
    con = 1.15 * np.exp(-((E - 0.9) / 0.26) ** 2)
    ingap = 0.42 * np.exp(-((E - 0.0) / 0.09) ** 2)
    axt.plot(E, val + con, color=F.C['blue'], lw=1.3)
    axt.plot(E, val + con + ingap, color=F.C['purple'], lw=1.2)
    axt.fill_between(E, 0, ingap, color=F.C['purple'], alpha=0.28)
    axt.set_ylim(0, 2.15)                       # headroom for the label
    axt.set_xlim(-1.5, 1.5)
    axt.annotate('mid-gap\nstates', xy=(0.0, ingap.max() + 0.02),
                 xytext=(-1.35, 1.62), fontsize=7.2, color=F.C['purple'],
                 ha='left', va='center',
                 arrowprops=dict(arrowstyle='->', color=F.C['purple'], lw=0.8))
    axt.set_title('electronic structure', fontsize=8.2, color=F.C['purple'],
                  pad=3)
    axt.set_xlabel('energy', fontsize=8, labelpad=1)
    axt.set_ylabel('DOS', fontsize=8, labelpad=1)
    axt.set_xticks([]); axt.set_yticks([])
    axt.text(-0.11, 1.06, '(b)', transform=axt.transAxes, fontsize=10,
             fontweight='bold')

    # (c) Kubo optics
    axo = fig.add_subplot(gsr[1])
    w = np.linspace(0.1, 3.0, 400)
    band = 1.5 * np.exp(-((w - 2.15) / 0.34) ** 2)
    sub = 0.46 * np.exp(-((w - 1.0) / 0.26) ** 2)
    axo.plot(w, band, color=F.C['orange'], lw=1.4)
    axo.plot(w, band + sub, color=F.C['red'], lw=1.2, ls='--')
    axo.fill_between(w, 0, sub, color=F.C['red'], alpha=0.18)
    axo.set_ylim(0, 2.05)
    axo.set_xlim(0.0, 3.1)
    axo.annotate('sub-gap\nabsorption', xy=(1.0, sub.max() + 0.02),
                 xytext=(0.05, 1.62), fontsize=7.2, color=F.C['red'],
                 ha='left', va='center',
                 arrowprops=dict(arrowstyle='->', color=F.C['red'], lw=0.8))
    axo.set_title('Kubo optics', fontsize=8.4, color=F.C['orange'], pad=3)
    axo.set_xlabel('photon energy', fontsize=8, labelpad=1)
    axo.set_ylabel(r'$\sigma(\omega)$', fontsize=8, labelpad=1)
    axo.set_xticks([]); axo.set_yticks([])
    axo.text(-0.11, 1.06, '(c)', transform=axo.transAxes, fontsize=10,
             fontweight='bold')

    out = os.path.join(os.path.dirname(__file__), '..', 'figures',
                       'fig1_concept.pdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    print('wrote', out)


if __name__ == '__main__':
    main()
