"""Run LCAO-PBE DFT on a structure and store H(k), S(k) labels.

The Hamiltonian is aligned to the vacuum level of each slab so that
matrix elements from different structures share one energy reference.
"""
import os
import numpy as np

os.environ.setdefault('GPAW_SETUP_PATH', os.path.expanduser('~/gpaw-data/setups'))


def run_structure(atoms, label, outdir, kpts=(2, 2, 1), txt=None):
    from gpaw import GPAW, FermiDirac
    from gpaw.lcao.tools import get_lcao_hamiltonian

    calc = GPAW(mode='lcao', basis='dzp', xc='PBE', kpts={'size': kpts, 'gamma': True}, h=0.24,
                occupations=FermiDirac(0.01), symmetry='off',
                txt=txt or os.path.join(outdir, label + '.txt'))
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    ef = calc.get_fermi_level()

    # vacuum level from the plane-averaged electrostatic potential at the
    # top of the cell (7.5 A above the slab)
    v = calc.get_electrostatic_potential().mean(axis=(0, 1))
    v_vac = 0.5 * (v[2] + v[-3])

    H_skMM, S_kMM = get_lcao_hamiltonian(calc)
    H_kMM = H_skMM[0] - v_vac * S_kMM  # align to vacuum = 0

    eig = np.array([calc.get_eigenvalues(kpt=k) for k in range(len(calc.get_ibz_k_points()))])
    bzk = calc.get_bz_k_points()

    np.savez_compressed(
        os.path.join(outdir, label + '.npz'),
        H_kMM=H_kMM.astype(np.complex128), S_kMM=S_kMM.astype(np.complex128),
        kpts=bzk, positions=atoms.positions, numbers=atoms.numbers,
        cell=atoms.cell.array, energy=energy, forces=forces,
        efermi=ef - v_vac, v_vac=v_vac, eigenvalues=eig - v_vac)
    return energy
