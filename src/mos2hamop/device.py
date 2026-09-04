"""Two-probe NEGF device built from an assembled block list.

The device is a finite-width MoS2 ribbon, transport along x, open in y.
Atoms are sorted along x and grouped into principal layers of width w
(w >= the 11 A interaction range), so only nearest-layer coupling
survives. Leads are the pristine ribbon principal layer, taken from a
reference pristine assembly with identical width. All on-layer,
coupling, and lead blocks are cut from the same ML block list.
"""
import numpy as np


def orbital_ranges(numbers, nao_map):
    offs, n = [], 0
    for Z in numbers:
        offs.append((n, n + nao_map[Z]))
        n += nao_map[Z]
    return offs, n


def assemble_dense(flat_blocks, nao):
    """Dense H0, S0 (the (0,0) image) and per-image full matrices.

    flat_blocks: list of (oi, oj, ni, nj, d, H, S) with d the Cartesian
    displacement from orbital-i atom to orbital-j atom (image included).
    Returns callable giving H(x_shift) contributions grouped by the
    integer x-layer difference. Here we instead build the full real-space
    coupling directly in layer space (see build_layers).
    """
    raise NotImplementedError


def build_layers(positions, numbers, flat_blocks, nao_map, layer_width,
                 x0=None, layer_of=None):
    """Partition into principal layers and return per-layer H, S and
    nearest-layer couplings.

    Returns dict with 'layers' (list of (Hd, Sd)), 'coup' (list of
    (Hc, Sc) from layer i to i+1), 'ranges' (orbital slice per layer),
    'layer_of' (layer index per atom), and 'maxskip' (largest layer
    separation with nonzero coupling, for the nearest-layer check).

    layer_of: optional explicit integer layer index per atom (used when
    the partition must be commensurate with the lattice period rather
    than binned from raw x, so every principal layer has equal size).
    """
    if layer_of is None:
        x = positions[:, 0]
        if x0 is None:
            x0 = x.min()
        layer_of = np.floor((x - x0) / layer_width).astype(int)
    layer_of = np.asarray(layer_of, int)
    layer_of = layer_of - layer_of.min()
    nlayer = layer_of.max() + 1
    offs, ntot = orbital_ranges(numbers, nao_map)

    # orbital ranges per layer
    lay_atoms = [np.nonzero(layer_of == L)[0] for L in range(nlayer)]
    lay_orb = []
    cursor = 0
    orb_index = np.zeros(ntot, int)  # global orbital -> packed layer orbital
    packed_ranges = []
    for L in range(nlayer):
        start = cursor
        for a in lay_atoms[L]:
            o0, o1 = offs[a]
            for g in range(o0, o1):
                orb_index[g] = cursor
                cursor += 1
        packed_ranges.append((start, cursor))
    sizes = [packed_ranges[L][1] - packed_ranges[L][0] for L in range(nlayer)]

    Hd = [np.zeros((s, s), complex) for s in sizes]
    Sd = [np.zeros((s, s), complex) for s in sizes]
    Hc = [None] * (nlayer - 1)   # coupling L -> L+1
    Sc = [None] * (nlayer - 1)
    for L in range(nlayer - 1):
        Hc[L] = np.zeros((sizes[L], sizes[L + 1]), complex)
        Sc[L] = np.zeros((sizes[L], sizes[L + 1]), complex)
    atom_of_orbital = np.zeros(ntot, int)
    for a, (o0, o1) in enumerate(offs):
        atom_of_orbital[o0:o1] = a
    maxskip = 0
    for (oi, oj, ni, nj, d, Hb, Sb) in flat_blocks:
        ai, aj = atom_of_orbital[oi], atom_of_orbital[oj]
        Li, Lj = layer_of[ai], layer_of[aj]
        skip = abs(Li - Lj)
        if skip > 0 and (abs(Hb).max() > 1e-6):
            maxskip = max(maxskip, skip)
        rows = orb_index[oi:oi + ni] - packed_ranges[Li][0]
        cols = orb_index[oj:oj + nj] - packed_ranges[Lj][0]
        if Li == Lj:
            Hd[Li][np.ix_(rows, cols)] += Hb
            Sd[Li][np.ix_(rows, cols)] += Sb
        elif Lj == Li + 1:
            Hc[Li][np.ix_(rows, cols)] += Hb
            Sc[Li][np.ix_(rows, cols)] += Sb
        elif Lj == Li - 1:
            Hc[Lj][np.ix_(cols, rows)] += Hb.conj().T
            Sc[Lj][np.ix_(cols, rows)] += Sb.conj().T
        # skips > 1 are dropped (checked via maxskip)
    return dict(layers=list(zip(Hd, Sd)), coup=list(zip(Hc, Sc)),
                sizes=sizes, nlayer=nlayer, maxskip=maxskip,
                layer_of=layer_of)
