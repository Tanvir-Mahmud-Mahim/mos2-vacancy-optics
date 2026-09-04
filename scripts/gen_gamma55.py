"""Generate the Gamma-point 5x5 dataset for the spectral validation.

The 5x5 supercell sampled at the zone centre has a largest minimum-image
pair distance of about 9.3 A, inside the 11 A learning range, so every
Hamiltonian and overlap block of the Born-von-Karman torus is available
to a local model by construction; the range obstruction quantified in
range_analysis.py does not exist at this cell size and sampling. Usage:
    python gen_gamma55.py probe          # one pristine timing probe
    python gen_gamma55.py train I J      # manifest slice [I, J)
    python gen_gamma55.py test  I J
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.structures import make_structure
from mos2hamop.dftrun import run_structure


def manifest(which):
    man = []
    def add(tag, **kw):
        man.append((f'g55_{tag}_{len(man):03d}', kw))
    if which == 'train':
        add('prist', n_vac=0, rattle=0.0, seed=1)
        for s in (2, 3):
            add('prist', n_vac=0, rattle=0.02, seed=s)
        add('prist', n_vac=0, rattle=0.04, seed=4)
        for s in (10, 11, 12):
            add('vac1', n_vac=1, rattle=0.02, seed=s)
        add('vac1', n_vac=1, rattle=0.04, seed=13)
        add('vac1b', n_vac=1, rattle=0.02, seed=14, vac_layer='bottom')
        for s in (20, 21, 22):
            add('vac2', n_vac=2, rattle=0.02, seed=s)
        add('vac2', n_vac=2, rattle=0.04, seed=23)
        for s in (30, 31):
            add('vac3', n_vac=3, rattle=0.02, seed=s)
        add('vac3', n_vac=3, rattle=0.04, seed=32)
    else:
        add('prist', n_vac=0, rattle=0.03, seed=101)
        add('vac1', n_vac=1, rattle=0.03, seed=102)
        add('vac2', n_vac=2, rattle=0.03, seed=103)
        add('vac2', n_vac=2, rattle=0.02, seed=104)
        add('vac3', n_vac=3, rattle=0.03, seed=105)
    return man


def main():
    mode = sys.argv[1]
    if mode == 'probe':
        outdir = os.path.join(os.path.dirname(__file__), '..', 'data', 'probe55')
        os.makedirs(outdir, exist_ok=True)
        t0 = time.time()
        atoms, meta = make_structure(5, 5, n_vac=0, rattle=0.0, seed=1)
        e = run_structure(atoms, 'probe55_prist', outdir, kpts=(1, 1, 1))
        print(f'probe55: E={e:.3f} eV in {time.time()-t0:.0f} s', flush=True)
        return
    which = mode
    i, j = int(sys.argv[2]), int(sys.argv[3])
    outdir = os.path.join(os.path.dirname(__file__), '..', 'data',
                          'train55' if which == 'train' else 'test55')
    os.makedirs(outdir, exist_ok=True)
    for name, kw in manifest(which)[i:j]:
        done = os.path.join(outdir, name + '.npz')
        if os.path.exists(done):
            print('skip', name, flush=True); continue
        t0 = time.time()
        atoms, meta = make_structure(5, 5, **kw)
        e = run_structure(atoms, name, outdir, kpts=(1, 1, 1))
        print(f'{name}: E={e:.3f} eV, {time.time()-t0:.0f} s', flush=True)
    print('slice done', flush=True)


if __name__ == '__main__':
    main()
