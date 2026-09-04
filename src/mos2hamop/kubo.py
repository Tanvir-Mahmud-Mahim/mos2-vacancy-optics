"""Kubo-Greenwood optical conductivity from real-space LCAO matrices.

The velocity operator uses the standard tight-binding (Peierls-like)
position gauge: the position operator is taken diagonal at the atomic
sites, so dH/dk carries a factor of the inter-site displacement. The
intra-atomic dipole contribution is neglected, which is the common
approximation in atomistic tight-binding optics; it preserves the
absorption edge and the relative spectral weight changes studied here.
For the nonorthogonal basis the interband matrix element at k is
    v_nm = <n| dH/dk - (e_n + e_m)/2 dS/dk |m> / hbar.
sigma_xx(omega) is reported in units of sigma_mono = e^2 / (4 hbar),
the universal sheet conductivity scale, per MoS2 layer.
"""
import numpy as np
from scipy.linalg import eigh, LinAlgError

from .eigsolve import gen_eigh

HBAR = 6.582119569e-16  # eV s
KB = 8.617333e-5        # eV / K


def bloch_matrices(blocks_R, nao, kpt, cell):
    """Assemble H(k), S(k), dH/dk_x, dS/dk_x from real-space blocks.

    blocks_R: list of (oi, oj, ni, nj, d_cart, H_block, S_block) with the
    convention H(k)_{ij} += exp(i k . d) H_block (d the Cartesian
    displacement from atom i to atom j including the lattice vector).
    kpt: Cartesian k (1/A).
    """
    H = np.zeros((nao, nao), complex)
    S = np.zeros((nao, nao), complex)
    dHx = np.zeros((nao, nao), complex)
    dSx = np.zeros((nao, nao), complex)
    for oi, oj, ni, nj, d, Hb, Sb in blocks_R:
        ph = np.exp(1j * (kpt @ d))
        H[oi:oi+ni, oj:oj+nj] += ph * Hb
        S[oi:oi+ni, oj:oj+nj] += ph * Sb
        dHx[oi:oi+ni, oj:oj+nj] += 1j * d[0] * ph * Hb
        dSx[oi:oi+ni, oj:oj+nj] += 1j * d[0] * ph * Sb
    # Hermitize (assembly includes both directions; rounding asymmetry only)
    H = 0.5 * (H + H.conj().T)
    S = 0.5 * (S + S.conj().T)
    dHx = 0.5 * (dHx + dHx.conj().T)
    dSx = 0.5 * (dSx + dSx.conj().T)
    return H, S, dHx, dSx


def sigma_xx(blocks_R, nao, kpts_cart, wk, area, omega, mu, T=300.0,
             eta=0.05, ne_expected=None):
    """Optical sheet conductivity in units of e^2/(4 hbar).

    omega: array of photon energies (eV). mu: chemical potential (eV,
    same vacuum reference as H). eta: Gaussian broadening (eV).
    Returns (sigma(omega), n_carriers_per_cm2, gap_info dict).
    """
    sig = np.zeros_like(omega)
    n_e = 0.0
    e_all = []
    for kpt, w in zip(kpts_cart, wk):
        H, S, dHx, dSx = bloch_matrices(blocks_R, nao, kpt, None)
        try:
            e, c = eigh(H, S)
        except LinAlgError:
            # reconstructed/interpolated S can be mildly non-positive-definite;
            # fall back to canonical orthogonalization (drops the near-null
            # overcomplete directions) so the optics stays well defined
            e, c = gen_eigh(H, S, thresh=1e-4, eigvals_only=False)
        e_all.append(e)
        f = 1.0 / (1.0 + np.exp(np.clip((e - mu) / (KB * T), -60, 60)))
        # velocity matrix (eV*A units before dividing by hbar)
        M = c.conj().T @ (dHx - 0.5 * dSx * 0) @ c
        # subtract (e_n+e_m)/2 dS/dk term
        Sd = c.conj().T @ dSx @ c
        Emat = 0.5 * (e[:, None] + e[None, :])
        M = M - Emat * Sd
        dE = e[None, :] - e[:, None]      # E_m - E_n
        df = f[:, None] - f[None, :]      # f_n - f_m
        A2 = np.abs(M) ** 2
        for iw, hw in enumerate(omega):
            g = np.exp(-0.5 * ((dE - hw) / eta) ** 2) / (eta * np.sqrt(2*np.pi))
            val = (df * A2 * g / np.maximum(dE, 1e-4))[dE > 1e-3].sum()
            sig[iw] += w * val
        n_e += w * 2.0 * f.sum()
    # sigma_xx(w) = (pi e^2 / hbar) (1/A) sum ... ; in units of e^2/(4 hbar):
    # sigma / (e^2/4hbar) = 4 pi / A * sum_k w_k sum_nm (f_n - f_m)|M|^2
    #                        g(hw - dE) / dE   with M in eV*A, dE in eV
    sig *= 4.0 * np.pi / area
    return sig, n_e, np.array(e_all)
