"""Supplementary validation figure: Kubo pristine edge + NEGF analytic chain."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import figstyle as F
import matplotlib.pyplot as plt
from mos2hamop.negf import transmission

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')


def main():
    sp = np.load(os.path.join(ROOT, 'dft_spectra.npz'))
    gap0 = float(sp['gap0'])
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.4))

    a = ax[0]
    a.axvspan(0, gap0, color='0.9', lw=0)
    a.plot(sp['omega'], sp['sig_pristine'], color=F.C['blue'], lw=1.5)
    a.axvline(gap0, color=F.C['red'], ls='--', lw=1)
    a.text(gap0 + 0.05, a.get_ylim()[1]*0.5 if a.get_ylim()[1] > 0 else 1,
           f'gap {gap0:.2f} eV', color=F.C['red'], fontsize=7.5, rotation=90,
           va='center')
    a.set_xlabel('photon energy (eV)')
    a.set_ylabel(r'$\sigma_{xx}\;(e^2/4\hbar)$')
    a.set_xlim(0, 3); a.set_ylim(bottom=0)
    a.set_title('pristine MoS$_2$: optical edge', fontsize=8.5)
    F.panel_label(a, '(a)')

    # NEGF analytic chain
    b = ax[1]
    t = -1.0
    H00 = np.array([[0.0]]); H01 = np.array([[t]])
    S00 = np.array([[1.0]]); S01 = np.array([[0.0]])
    N = 6
    E = np.linspace(-2.4, 2.4, 200)
    lH = [H00.copy() for _ in range(N)]; lS = [S00.copy() for _ in range(N)]
    cH = [H01.copy() for _ in range(N-1)]; cS = [S01.copy() for _ in range(N-1)]
    T = transmission(E, lH, lS, cH, cS, H00, H01, S00, S01, eta=1e-4)
    lHb = [H00.copy() for _ in range(N)]; lHb[3] = np.array([[1.2]])
    Tb = transmission(E, lHb, lS, cH, cS, H00, H01, S00, S01, eta=1e-4)
    b.plot(E, T, color=F.C['blue'], lw=1.5, label='clean chain')
    b.plot(E, Tb, color=F.C['orange'], lw=1.3, ls='--', label='with barrier')
    b.set_xlabel('energy (units of $|t|$)')
    b.set_ylabel(r'transmission $T(E)$')
    b.set_ylim(-0.05, 1.15)
    b.set_title('NEGF: analytic 1D chain', fontsize=8.5)
    b.legend(fontsize=7.5, loc='lower center')
    F.panel_label(b, '(b)')

    fig.tight_layout(w_pad=1.4)
    out = os.path.join(ROOT, '..', 'figures', 'figS_validation.pdf')
    fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'))
    print('wrote', out)


if __name__ == '__main__':
    main()
