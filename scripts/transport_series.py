"""NEGF transmission versus sulfur-vacancy density on a MoS2 ribbon.

A ribbon of width ny primitive cells (open in y, isolated by vacuum) and
length nx primitive cells (transport along x, periodic images along x
only) is assembled entirely from the learned operator. It is cut into
principal layers four cells wide so that only nearest-layer coupling
survives (checked). The two end regions are pristine and match the
semi-infinite pristine leads; sulfur vacancies at a target density are
placed in the interior. Transmission is averaged over disorder
realizations and reported around the conduction-band edge.
"""
import os, sys, time, json, pickle
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ase.build import mx2
from mos2hamop.structures import A_LATT, THICKNESS
from mos2hamop.mlmodel import BlockModel
from mos2hamop.reference import DistanceReference
from mos2hamop.assemble import predict_blocks, hermitize, flatten
from mos2hamop.device import build_layers
from mos2hamop.negf import transmission
from mos2hamop.blocks import NAO

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
LAYER_CELLS = 4                    # principal-layer width in primitive cells
GUARD_CELLS = 8                    # pristine cells kept at each ribbon end


def load_models():
    states = pickle.load(open(os.path.join(ROOT, 'models.pkl'), 'rb'))
    models = {}
    for key, (stH, stS) in states.items():
        mH = BlockModel(len(stH['x_mean']), stH['ni'], stH['nj']); mH.load(stH)
        mS = BlockModel(len(stS['x_mean']), stS['ni'], stS['nj']); mS.load(stS)
        models[key] = (mH, mS)
    refs = None
    rp = os.path.join(ROOT, 'refs.pkl')
    if os.path.exists(rp):
        rs = pickle.load(open(rp, 'rb'))
        refs = {k: (DistanceReference.load(a), DistanceReference.load(b))
                for k, (a, b) in rs.items()}
    return models, refs


def make_ribbon(nx, ny, n_vac=0, seed=0, rattle=0.02):
    """Return (atoms, cell_index) with cell_index the a1 lattice cell (0..nx-1)
    of every atom, used to build lattice-commensurate principal layers."""
    atoms = mx2(formula='MoS2', kind='2H', a=A_LATT, thickness=THICKNESS,
                vacuum=5.5)
    atoms = atoms.repeat((nx, ny, 1))
    atoms.pbc = (False, False, False)          # finite device along x and y
    # a1 cell index from the fractional coordinate along axis 0 (pre-extension)
    frac = atoms.get_scaled_positions(wrap=False)[:, 0]
    cell_index = np.clip(np.floor(frac * nx + 1e-6).astype(int), 0, nx - 1)
    cell = atoms.cell.array.copy()
    ylen = np.linalg.norm(cell[1])
    cell[1] = cell[1] / ylen * (ylen + 12.0)   # y vacuum isolates the ribbon
    xlen = np.linalg.norm(cell[0])
    cell[0] = cell[0] / xlen * (xlen + 20.0)   # x vacuum: finite device
    atoms.set_cell(cell)
    keep = np.ones(len(atoms), bool)
    if n_vac > 0:
        rng = np.random.default_rng(seed)
        cand = [i for i, at in enumerate(atoms) if at.symbol == 'S'
                and GUARD_CELLS <= cell_index[i] < nx - GUARD_CELLS]
        pick = rng.choice(cand, size=min(n_vac, len(cand)), replace=False)
        keep[pick] = False
    atoms = atoms[keep]; cell_index = cell_index[keep]
    if rattle > 0:
        rng2 = np.random.default_rng(seed + 999)
        atoms.positions += rng2.normal(0, rattle, atoms.positions.shape)
    return atoms, cell_index


def assemble_ribbon(atoms, models, refs):
    pos, num, cell = atoms.positions, atoms.numbers, atoms.cell.array
    images = [(0, 0)]  # finite device: no wrap along x
    bbi, nao = predict_blocks(pos, num, cell, images, models, (1, 1), refs)
    bbi = hermitize(bbi)
    return flatten(bbi), pos, num, cell


def layers_of(atoms, cell_index, models, refs):
    flat, pos, num, cell = assemble_ribbon(atoms, models, refs)
    layer_of = cell_index // LAYER_CELLS
    return build_layers(pos, num, flat, NAO, LAYER_CELLS * A_LATT,
                        layer_of=layer_of)


def run(nx=24, ny=4, seed_list=(0, 1, 2), n_vac_list=(0, 4, 8, 16, 24),
        emin=-0.4, emax=1.2, ne=90):
    models, refs = load_models()
    at0, ci0 = make_ribbon(nx, ny, n_vac=0, rattle=0.0)
    lay0 = layers_of(at0, ci0, models, refs)
    N = lay0['nlayer']; mid = N // 2
    lead_H00, lead_S00 = lay0['layers'][mid]
    lead_H01, lead_S01 = lay0['coup'][mid]
    print(f'nlayer={N} maxskip={lay0["maxskip"]} lead_size={lay0["sizes"][mid]}',
          flush=True)
    E = np.linspace(emin, emax, ne)
    out = {'E': E.tolist(), 'nx': nx, 'ny': ny, 'curves': {}}
    for nv in n_vac_list:
        Ts = []
        seeds = (0,) if nv == 0 else seed_list
        for seed in seeds:
            at, ci = make_ribbon(nx, ny, n_vac=nv, seed=seed, rattle=0.02)
            lay = layers_of(at, ci, models, refs)
            lH = [h for h, s in lay['layers']]; lS = [s for h, s in lay['layers']]
            cH = [h for h, s in lay['coup']]; cS = [s for h, s in lay['coup']]
            t0 = time.time()
            T = transmission(E, lH, lS, cH, cS, lead_H00, lead_H01,
                             lead_S00, lead_S01, eta=1e-4)
            Ts.append(T)
            print(f'nv={nv} seed={seed}: maxT={T.max():.2f} skip={lay["maxskip"]}'
                  f' {time.time()-t0:.0f}s', flush=True)
        out['curves'][str(nv)] = np.array(Ts).tolist()
    json.dump(out, open(os.path.join(ROOT, 'transport.json'), 'w'))
    print('transport done')


if __name__ == '__main__':
    run()
