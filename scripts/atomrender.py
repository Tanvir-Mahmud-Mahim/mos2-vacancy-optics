"""Publication-quality 3D atom renderer (shaded spheres, depth sorted).

Draws a real monolayer MoS2 cluster in trigonal-prismatic coordination
(Mo triangular lattice at z=0; the two S sub-planes vertically aligned at
z = +/- t/2 over the hollow site of the Mo triangle). Atoms are projected
with a fixed view direction and drawn back-to-front as radially shaded
discs, so the result reads as a genuine 3D ball-and-stick model without
any external ray tracer. Geometry uses the DFT lattice constants, so it
is accurate, not schematic.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection

# cache of rendered sphere RGBA images, keyed by (colour, size)
_SPHERE_CACHE = {}


def _sphere_image(base, size=256, light=(-0.42, 0.52, 0.75), shininess=38):
    """A photometric sphere: Lambertian diffuse + Blinn-Phong specular on a
    unit disc, returned as an RGBA image (transparent outside the disc)."""
    key = (tuple(np.round(base, 4)), size)
    if key in _SPHERE_CACHE:
        return _SPHERE_CACHE[key]
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    r2 = xx ** 2 + yy ** 2
    inside = r2 <= 1.0
    z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    N = np.dstack([xx, yy, z])
    L = np.array(light, float); L /= np.linalg.norm(L)
    V = np.array([0.0, 0.0, 1.0])
    H = L + V; H /= np.linalg.norm(H)
    diff = np.clip((N * L).sum(-1), 0.0, 1.0)
    spec = np.clip((N * H).sum(-1), 0.0, 1.0) ** shininess
    amb = 0.34
    shade = amb + 0.72 * diff
    col = base[None, None, :] * shade[..., None] + 0.85 * spec[..., None]
    # gentle rim darkening for silhouette definition
    rim = np.clip((r2 - 0.72) / 0.28, 0.0, 1.0) * inside
    col = col * (1.0 - 0.35 * rim[..., None])
    col = np.clip(col, 0.0, 1.0)
    rgba = np.zeros((size, size, 4))
    rgba[..., :3] = col
    rgba[..., 3] = inside.astype(float)
    _SPHERE_CACHE[key] = rgba
    return rgba

A = 3.184            # in-plane lattice constant (Ang), PBE
T = 3.127            # S-S vertical separation (Ang)
R_MO = 0.62          # drawing radii (Ang, illustrative covalent-ish)
R_S = 0.42
COL_MO = np.array([0.42, 0.36, 0.85])     # indigo
COL_S = np.array([0.96, 0.77, 0.26])      # gold


def cluster(nx, ny):
    """Mo and S sites for an nx x ny patch; S carries +/- z (prismatic)."""
    a1 = np.array([A, 0, 0]); a2 = np.array([A / 2, A * np.sqrt(3) / 2, 0])
    mo, s = [], []
    for i in range(nx):
        for j in range(ny):
            o = i * a1 + j * a2
            mo.append(o.copy())
            st = o + (a1 + a2) / 3.0
            s.append(st + np.array([0, 0, T / 2]))
            s.append(st + np.array([0, 0, -T / 2]))
    return np.array(mo), np.array(s)


def view_matrix(elev, azim):
    e = np.radians(elev); a = np.radians(azim)
    Rz = np.array([[np.cos(a), -np.sin(a), 0],
                   [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)],
                   [0, np.sin(e), np.cos(e)]])
    return Rx @ Rz


def _shaded_sphere(ax, x, y, R, base, zorder):
    """Draw a photometric shaded sphere of radius R centred at (x, y)."""
    img = _sphere_image(np.asarray(base, float))
    ax.imshow(img, extent=[x - R, x + R, y - R, y + R], origin='lower',
              zorder=zorder, interpolation='bilinear', aspect='equal')


def render(ax, nx=4, ny=4, vac_sites=None, elev=20, azim=-58,
           label_vac=True, vac_color=(0.86, 0.12, 0.12), scale=1.0,
           bond_cut=2.6):
    """Render a cluster into a 2D axis. vac_sites: list of S indices removed.

    Returns dict with 2D projected Mo/S positions for external annotation.
    """
    mo, s = cluster(nx, ny)
    M = view_matrix(elev, azim)
    mo2 = mo @ M.T
    s2 = s @ M.T

    # remove vacancy S atoms (choose central upper S if not given)
    if vac_sites is None:
        cx, cy = mo[:, 0].mean(), mo[:, 1].mean()
        up = np.where(s[:, 2] > 0)[0]
        vac_sites = [up[np.argmin((s[up, 0]-cx)**2 + (s[up, 1]-cy)**2)]]
    keep = np.ones(len(s), bool)
    for v in vac_sites:
        keep[v] = False

    # bonds Mo-S among kept atoms
    segs, seg_depth = [], []
    for mi, m in enumerate(mo):
        for si in range(len(s)):
            if not keep[si]:
                continue
            if np.linalg.norm(m - s[si]) < bond_cut:
                p, q = mo2[mi], s2[si]
                segs.append([(p[0], p[1]), (q[0], q[1])])
                seg_depth.append(0.5 * (p[2] + q[2]))

    # painter's algorithm: everything sorted by projected depth (y-view axis)
    items = []
    for mi in range(len(mo)):
        items.append((mo2[mi, 2], 'mo', mi))
    for si in range(len(s)):
        if keep[si]:
            items.append((s2[si, 2], 's', si))
    items.sort(key=lambda t: t[0])
    order = {}
    for rank, (_, _, _) in enumerate(items):
        pass

    # draw bonds first, depth-ranked below atoms
    zb = np.argsort(np.argsort(seg_depth))
    if segs:
        lc = LineCollection(segs, colors='0.55', linewidths=1.1 * scale,
                            zorder=1)
        ax.add_collection(lc)

    base_z = 3
    for rank, (depth, kind, i) in enumerate(items):
        z = base_z + rank * 0.05
        if kind == 'mo':
            _shaded_sphere(ax, mo2[i, 0], mo2[i, 1], R_MO * scale, COL_MO, z)
        else:
            _shaded_sphere(ax, s2[i, 0], s2[i, 1], R_S * scale, COL_S, z)

    # vacancy ring at the projected site of the removed atom
    vac2 = s2[vac_sites[0]]
    ring = Circle((vac2[0], vac2[1]), R_S * scale * 1.7, fill=False,
                  ec=vac_color, lw=1.7 * scale, zorder=base_z + len(items) + 1)
    ax.add_patch(ring)

    ax.set_aspect('equal'); ax.axis('off')
    allx = np.concatenate([mo2[:, 0], s2[:, 0]])
    ally = np.concatenate([mo2[:, 1], s2[:, 1]])
    pad = 1.0
    ax.set_xlim(allx.min()-pad, allx.max()+pad)
    ax.set_ylim(ally.min()-pad, ally.max()+pad)
    return dict(mo2=mo2, s2=s2, vac2=vac2, keep=keep, xlim=ax.get_xlim(),
                ylim=ax.get_ylim())


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(3, 3))
    render(ax, 4, 4)
    fig.savefig('/tmp/atomtest.png', dpi=200, bbox_inches='tight')
    print('ok')
