"""Structure generation for monolayer MoS2 with sulfur vacancies.

All cells are 1H monolayer MoS2 built with the ASE mx2 helper at the
PBE lattice constant a = 3.184 A and S-S thickness 3.127 A, with 7.5 A
of vacuum on each side. Vacancies remove sulfur atoms; rattles apply
Gaussian displacements with a fixed random seed so every structure in
the dataset is exactly reproducible.
"""
import numpy as np
from ase.build import mx2

A_LATT = 3.184      # PBE in-plane lattice constant (Angstrom)
THICKNESS = 3.127   # PBE S-S vertical distance (Angstrom)
VACUUM = 5.5        # vacuum on each side (Angstrom)


def supercell(nx, ny, strain=0.0):
    atoms = mx2(formula='MoS2', kind='2H', a=A_LATT * (1.0 + strain),
                thickness=THICKNESS, vacuum=VACUUM)
    atoms = atoms.repeat((nx, ny, 1))
    atoms.pbc = (True, True, False)
    return atoms


def sulfur_indices(atoms, layer='top'):
    zmid = atoms.positions[:, 2].mean()
    out = []
    for i, at in enumerate(atoms):
        if at.symbol != 'S':
            continue
        if layer == 'top' and at.position[2] > zmid:
            out.append(i)
        elif layer == 'bottom' and at.position[2] < zmid:
            out.append(i)
        elif layer == 'any':
            out.append(i)
    return out


def make_structure(nx, ny, n_vac=0, strain=0.0, rattle=0.0, seed=0,
                   vac_layer='top', vac_seed=None):
    """Build one dataset structure. Returns (atoms, meta)."""
    rng = np.random.default_rng(seed)
    atoms = supercell(nx, ny, strain=strain)
    removed = []
    if n_vac > 0:
        vrng = np.random.default_rng(seed if vac_seed is None else vac_seed)
        cand = sulfur_indices(atoms, vac_layer)
        pick = sorted(vrng.choice(len(cand), size=n_vac, replace=False))
        removed = [cand[i] for i in pick]
        for i in sorted(removed, reverse=True):
            del atoms[i]
    if rattle > 0:
        atoms.positions += rng.normal(0.0, rattle, atoms.positions.shape)
    meta = dict(nx=nx, ny=ny, n_vac=n_vac, strain=strain, rattle=rattle,
                seed=seed, vac_layer=vac_layer, removed=removed)
    return atoms, meta
