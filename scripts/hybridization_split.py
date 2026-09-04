"""Hybridization splitting of the mid-gap manifold in the dilute
two-vacancy separation series (data/sep).

For each configuration of the controlled two-vacancy series, collect the
eigenvalues that fall inside the pristine gap (0.1 eV margins) over all
k-points and report the total width of that mid-gap manifold. Bringing
the two vacancies to contact widens the manifold: the defect
wavefunctions hybridize and their levels split, the quantum mechanism
behind the collective brightness discussed in the paper.
"""
import os, sys, json, glob
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
MARGIN = 0.1


def main():
    an = json.load(open(os.path.join(ROOT, 'dft_analysis.json')))
    vbm0, cbm0 = an['vbm0'], an['cbm0']
    sep = {r['name']: r['sep'] for r in
           json.load(open(os.path.join(ROOT, 'separation.json')))}
    rows = []
    for fn in sorted(glob.glob(os.path.join(ROOT, 'sep', 'sep5_*.npz'))):
        name = os.path.basename(fn)[:-4]
        ev = np.load(fn)['eigenvalues']
        gap_states = np.concatenate([
            e[(e > vbm0 + MARGIN) & (e < cbm0 - MARGIN)] for e in ev])
        width = float(gap_states.max() - gap_states.min())
        rows.append((sep[name], width, name))
        print(f'{name}: separation {sep[name]:.2f} A, '
              f'mid-gap manifold width {width*1e3:.0f} meV')
    rows.sort()
    near = np.mean([w for s, w, n in rows if s < 4])
    far = np.mean([w for s, w, n in rows if s > 4])
    print(f'mean width at contact (<4 A): {near*1e3:.0f} meV; '
          f'separated (>4 A): {far*1e3:.0f} meV; '
          f'widening {(near-far)*1e3:.0f} meV')


if __name__ == '__main__':
    main()
