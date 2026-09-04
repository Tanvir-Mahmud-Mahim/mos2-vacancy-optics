"""Exact LCAO overlap S(R) for an arbitrary geometry, without SCF.

The overlap matrix is a strictly two-center geometric integral of the
basis functions; it does not depend on the self-consistent density, so
it is obtained from a single LCAO setup step (no SCF convergence). This
is the cheap part of a density-functional calculation that the learned
model does not need to replace: only the self-consistent Hamiltonian H
is learned, while S is taken exactly here, following the operator-
learning literature. The returned real-space blocks are matched to the
learned H blocks by atom pair and displacement.
"""
import os
import numpy as np


def exact_overlap_blocks(atoms, kgrid=(2, 2, 1), h=0.28):
    """Return {(i, j, d_key): S_block} for all pairs within the cutoff.

    d_key is tuple(round(d, 4)) of the Cartesian displacement, matching
    the convention used by assemble.predict_blocks.
    """
    from gpaw import GPAW, FermiDirac
    from gpaw.lcao.tools import get_lcao_hamiltonian
    from .blocks import realspace_matrices, pair_blocks, orbital_offsets

    calc = GPAW(mode='lcao', basis='dzp', xc='PBE', h=h,
                kpts={'size': kgrid, 'gamma': True},
                occupations=FermiDirac(0.01), symmetry='off',
                convergence={'energy': 1e30, 'density': 1e30,
                             'eigenstates': 1e30}, maxiter=1,
                txt=None)
    atoms = atoms.copy(); atoms.calc = calc
    try:
        atoms.get_potential_energy()
    except Exception:
        pass   # one non-converged step is enough; S is SCF-independent
    _, S_kMM = get_lcao_hamiltonian(calc)
    cell = atoms.cell.array
    mats = realspace_matrices(np.zeros_like(S_kMM), S_kMM,
                              calc.get_bz_k_points(), kgrid[:2])
    mr = {k: (H.real, S.real) for k, (H, S) in mats.items()}
    samples = pair_blocks(mr, atoms.positions, atoms.numbers, cell,
                          kgrid[:2], rcut=11.0)
    out = {}
    for s in samples:
        out[(s['i'], s['j'], tuple(np.round(s['d'], 4)))] = s['S']
    return out
