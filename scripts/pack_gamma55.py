"""Pack the Gamma-5x5 spectral-validation benchmark for Zenodo.

The raw 5x5 Hamiltonian/overlap matrices are hundreds of megabytes and
fully regenerable by gen_gamma55.py with the settings pinned in
src/mos2hamop/dftrun.py, so this archive ships the verifiable core
instead: every structure's geometry and exact DFT eigenvalues, the
trained block models and distance references, and the complete
validation output (spectra included). With these files the learned
operators can be re-assembled from geometry alone (predict_blocks) and
checked against the stored DFT eigenvalues and spectra without any DFT.
"""
import os, io, json, zipfile
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT = os.path.join(ROOT, 'gamma55-spectral-benchmark.zip')

README = """Gamma-5x5 spectral-validation benchmark
=======================================

Companion archive for "Learning the quantum Hamiltonian of defective
monolayer MoS2 reveals collective vacancy brightness decoupled from
defect count" (see the paper's Data Availability for the code
repository).

Contents
--------
structures/g55_*.npz   geometry (positions, numbers, cell, kpts) and the
                       exact DFT eigenvalues of every training and test
                       structure of the zone-center-sampled 5x5 set
                       (raw H(k), S(k) are regenerable with
                       scripts/gen_gamma55.py; settings pinned in the
                       code repository)
models55.pkl           trained per-block-type models (state dicts)
refs55.pkl             distance-binned reference blocks
spectral_validation.json  per-structure gaps, sub-gap absorption and
                       eigenvalue errors (exact, projected-reference and
                       learned), as printed by spectral_validation.py
spectral_validation.npz   the corresponding optical spectra

Reproduction: scripts/spectral_validation.py in the code repository
retrains and revalidates everything from the raw data; with this
archive alone, predict_blocks in src/mos2hamop/assemble.py rebuilds the
learned H and S of any structure from its geometry for comparison with
the stored DFT eigenvalues and spectra.
"""


def main():
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('README.txt', README)
        for sub in ('train55', 'test55'):
            d = os.path.join(ROOT, sub)
            for fn in sorted(os.listdir(d)):
                if not fn.endswith('.npz'):
                    continue
                a = np.load(os.path.join(d, fn))
                buf = io.BytesIO()
                np.savez_compressed(
                    buf, positions=a['positions'], numbers=a['numbers'],
                    cell=a['cell'], kpts=a['kpts'],
                    eigenvalues=a['eigenvalues'], efermi=a['efermi'])
                z.writestr(f'structures/{sub}_{fn}', buf.getvalue())
        for fn in ('models55.pkl', 'refs55.pkl', 'spectral_validation.json',
                   'spectral_validation.npz'):
            z.write(os.path.join(ROOT, fn), fn)
    print('wrote', OUT, f'{os.path.getsize(OUT)/1e6:.1f} MB')


if __name__ == '__main__':
    main()
