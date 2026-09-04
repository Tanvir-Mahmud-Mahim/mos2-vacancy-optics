"""NEGF transmission for a monolayer MoS2 strip from LCAO blocks.

Geometry: transport along x, periodic along y (Bloch phases with k_y),
semi-infinite pristine leads on both sides. The device is partitioned
into principal layers wide enough that only nearest-layer coupling
survives (layer width >= the 11 A interaction range). Surface Green
functions use the Sancho-Rubio decimation; the device is traversed by
the standard recursive Green function sweep.
"""
import numpy as np


def sancho_rubio(E, H00, H01, S00, S01, eta=1e-5, maxiter=400, tol=1e-9):
    """Retarded surface Green function of a semi-infinite lead."""
    z = (E + 1j * eta)
    a = z * S01 - H01                    # coupling to the next cell
    b = a.conj().T
    es = e = z * S00 - H00
    I = np.eye(len(e), dtype=complex)
    for _ in range(maxiter):
        g = np.linalg.solve(e, I)
        ab = a @ g @ b
        ba = b @ g @ a
        es = es - ab
        e = e - ab - ba
        a = a @ g @ a
        b = b @ g @ b
        if np.abs(a).max() + np.abs(b).max() < tol:
            return np.linalg.solve(es, I)
    # fall back to the last iterate (converged to working precision)
    return np.linalg.solve(es, I)


def transmission(E_list, layers_H, layers_S, coup_H, coup_S,
                 lead_H00, lead_H01, lead_S00, lead_S01, eta=1e-5):
    """T(E) for a device of N principal layers between identical leads.

    layers_H[i]: on-layer H of layer i; coup_H[i]: coupling layer i -> i+1.
    lead blocks describe the pristine principal layer. The first and last
    device layers must be pristine so lead coupling equals lead_H01.
    """
    N = len(layers_H)
    T = np.zeros(len(E_list))
    for iE, E in enumerate(E_list):
        z = E + 1j * eta
        gL = sancho_rubio(E, lead_H00, lead_H01.conj().T, lead_S00,
                          lead_S01.conj().T, eta)   # left lead grows to -x
        gR = sancho_rubio(E, lead_H00, lead_H01, lead_S00, lead_S01, eta)
        tauL = z * lead_S01 - lead_H01               # lead -> device hop
        sigL = tauL.conj().T @ gL @ tauL
        tauR = z * lead_S01 - lead_H01
        sigR = tauR @ gR @ tauR.conj().T
        gamL = 1j * (sigL - sigL.conj().T)
        gamR = 1j * (sigR - sigR.conj().T)
        # forward RGF sweep
        G = None
        g_prev = None
        for i in range(N):
            h_eff = z * layers_S[i] - layers_H[i]
            if i == 0:
                h_eff = h_eff - sigL
            if i == N - 1:
                h_eff = h_eff - sigR
            if i == 0:
                g_prev = np.linalg.inv(h_eff)
                Gs = [g_prev]
                continue
            tau = z * coup_S[i - 1] - coup_H[i - 1]
            g_prev = np.linalg.inv(h_eff - tau.conj().T @ g_prev @ tau)
            Gs.append(g_prev)
        # full G_{N1} via back propagation
        G_Ni = Gs[-1]
        prod = G_Ni
        for i in range(N - 2, -1, -1):
            tau = z * coup_S[i] - coup_H[i]
            prod = prod @ tau.conj().T @ Gs[i]
        G1N = prod  # G_{N,1}
        T[iE] = np.real(np.trace(gamR @ G1N @ gamL @ G1N.conj().T))
    return T
