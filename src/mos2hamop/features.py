"""Pair-frame descriptors for Hamiltonian block regression.

Every sample is an ordered atom pair (i, j) with displacement d (Cartesian,
minimum image on the BvK lattice). The pair frame keeps the global z axis
(the monolayer normal) and rotates x onto the in-plane projection of d.
Neighbor atoms of both endpoints are encoded in cylindrical coordinates
of that frame with smooth radial Gaussians, angular Fourier components,
and a linear z profile, separately per neighbor species. On-site samples
(i == j, d == 0) use the global frame. Because the frame co-rotates with
the structure, the descriptor is exactly invariant under any rotation
about z, and the target block is rotated into the same frame with the
matrices of rotations.py, making the whole regression z-equivariant.
"""
import numpy as np

R_ENV = 6.0          # neighbor environment radius (< half the 4x4 cell)
N_RAD = 6            # radial Gaussians
M_ANG = 3            # angular Fourier order (1, cos/sin up to M)
RAD_CENTERS = np.linspace(0.8, R_ENV, N_RAD)
RAD_SIGMA = 0.9


def frame_angle(d):
    """In-plane rotation angle of the pair frame (0 for on-site pairs)."""
    if np.hypot(d[0], d[1]) < 1e-6:
        return 0.0
    return np.arctan2(d[1], d[0])


def _env_features(center, others_by_species, theta):
    """Encode neighbor cloud around one endpoint in the rotated frame."""
    c, s = np.cos(-theta), np.sin(-theta)
    feats = []
    for pts in others_by_species:  # list of (n, 3) arrays: Mo then S
        if len(pts) == 0:
            feats.append(np.zeros(N_RAD * (1 + 2 * M_ANG) * 2))
            continue
        rel = pts - center
        x = c * rel[:, 0] - s * rel[:, 1]
        y = s * rel[:, 0] + c * rel[:, 1]
        z = rel[:, 2]
        rho = np.hypot(x, y)
        r3 = np.sqrt(rho**2 + z**2)
        keep = r3 < R_ENV
        x, y, z, rho = x[keep], y[keep], z[keep], rho[keep]
        phi = np.arctan2(y, x)
        rad = np.exp(-0.5 * ((rho[:, None] - RAD_CENTERS[None, :])
                             / RAD_SIGMA) ** 2)      # (n, N_RAD)
        ang = [np.ones_like(phi)]
        for m in range(1, M_ANG + 1):
            ang += [np.cos(m * phi), np.sin(m * phi)]
        ang = np.stack(ang, axis=1)                   # (n, 1+2M)
        zprof = np.stack([np.ones_like(z), z / 2.0], axis=1)  # (n, 2)
        f = np.einsum('na,nb,nc->abc', rad, ang, zprof).ravel()
        feats.append(f)
    return np.concatenate(feats)


def pair_descriptor(sample, positions, numbers, cell, kgrid):
    """Full descriptor for one pair sample (dict from pair_blocks)."""
    d = sample['d']
    theta = frame_angle(d)
    c, s = np.cos(-theta), np.sin(-theta)
    dx = c * d[0] - s * d[1]
    dz = d[2]

    # neighbor positions on the BvK torus around endpoint i and endpoint j
    n1, n2 = kgrid
    imgs = []
    for m1 in range(-2, n1 + 2):
        for m2 in range(-2, n2 + 2):
            imgs.append(positions + m1 * cell[0] + m2 * cell[1])
    allpos = np.concatenate(imgs)
    allnum = np.tile(numbers, len(imgs))

    pi = positions[sample['i']]
    pj = positions[sample['i']] + d  # endpoint j in the same image as the pair
    fi = _env_features(pi, [allpos[allnum == 42], allpos[allnum == 16]], theta)
    fj = _env_features(pj, [allpos[allnum == 42], allpos[allnum == 16]], theta)
    scal = np.array([sample['dist'], dx, dz, dz * dz])
    return np.concatenate([scal, fi, fj])
