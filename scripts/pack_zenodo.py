"""Package the benchmark dataset and trained models for Zenodo.

The archive keeps, for every DFT configuration, the reproducibility-critical
metadata (positions, numbers, cell, k-points, Fermi level, total energy,
forces, vacuum reference, and all Kohn-Sham eigenvalues) plus the analysis
products and trained models. The raw H(k), S(k) matrices (about 100 MB per
configuration, about 3 GB in total) are omitted; they are regenerated
bit-reproducibly by scripts/gen_dataset.py from the archived geometries.
"""
import io, os, sys, glob, json, zipfile, hashlib
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT = os.path.join(os.path.dirname(__file__), '..',
                   'mos2-vacancy-optics-benchmark.zip')
DROP = ('H_kMM', 'S_kMM')       # regenerable, dominates the size
PARITY_KEEP = 400_000           # archived parity points (fixed seed 0)


def slim_npz_bytes(path):
    """The npz at `path` with the DROP arrays removed, as bytes."""
    d = np.load(path)
    keep = {k: d[k] for k in d.files if k not in DROP}
    buf = io.BytesIO()
    np.savez_compressed(buf, **keep)
    return buf.getvalue()


def slim_parity_bytes(path, n_keep=PARITY_KEEP, seed=0):
    """A fixed-seed subsample of the held-out parity arrays, float32.

    The full element-wise parity set regenerates deterministically with
    scripts/ml_validation.py; the archive keeps a documented random
    subsample that is more than enough to reproduce the parity figure
    and the error-versus-distance analysis.
    """
    d = np.load(path)
    n = len(d['dft'])
    rng = np.random.default_rng(seed)
    sel = np.sort(rng.choice(n, min(n_keep, n), replace=False))
    buf = io.BytesIO()
    np.savez_compressed(buf,
                        dft=d['dft'][sel].astype(np.float32),
                        ml=d['ml'][sel].astype(np.float32),
                        dist=d['dist'][sel].astype(np.float32),
                        n_full=np.int64(n), seed=np.int64(seed))
    return buf.getvalue()


def main():
    manifest = []
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
        def add_bytes(rel, payload):
            z.writestr(rel, payload)
            manifest.append({'file': rel, 'bytes': len(payload),
                             'md5': hashlib.md5(payload).hexdigest()})

        for sub in ('train', 'sep'):
            for p in sorted(glob.glob(os.path.join(ROOT, sub, '*.npz'))):
                rel = os.path.join('data', sub, os.path.basename(p))
                add_bytes(rel, slim_npz_bytes(p))
        for f in ['dft_analysis.json', 'dft_spectra.npz', 'separation.json',
                  'ml_report.json', 'ablation.json', 'models.pkl',
                  'refs.pkl']:
            p = os.path.join(ROOT, f)
            if os.path.exists(p):
                add_bytes(os.path.join('data', f), open(p, 'rb').read())
        pp = os.path.join(ROOT, 'ml_parity.npz')
        if os.path.exists(pp):
            add_bytes('data/ml_parity.npz', slim_parity_bytes(pp))

        readme = (
            "Benchmark dataset for 'Optical brightness of sulfur vacancies "
            "decouples from their count in monolayer MoS2'.\n\n"
            "Contents:\n"
            "  data/train/*.npz   per-configuration DFT (GPAW/PBE/dzp) "
            "metadata of each sulfur-vacancy 4x4 supercell: positions, "
            "numbers, cell, k-points, Fermi level, total energy, forces, "
            "vacuum reference, and all Kohn-Sham eigenvalues. The raw "
            "H(k), S(k) matrices (~3 GB) are omitted; they are regenerated "
            "by scripts/gen_dataset.py from these geometries.\n"
            "  data/sep/*.npz     controlled two-vacancy separation series "
            "(5x5), same fields.\n"
            "  data/dft_analysis.json   per-configuration optical/electronic "
            "descriptors.\n"
            "  data/dft_spectra.npz     mean optical conductivity per vacancy "
            "count.\n"
            "  data/separation.json     sub-gap absorption vs vacancy "
            "separation.\n"
            "  data/models.pkl, refs.pkl   trained learned-Hamiltonian "
            "models (environment descriptor on onsite blocks, two-center "
            "displacement on pair blocks).\n"
            "  data/ml_report.json   per-block held-out validation errors.\n"
            "  data/ml_parity.npz    held-out matrix-element parity, a "
            "fixed-seed (seed 0) subsample of 400000 points in float32; the "
            "full parity set regenerates deterministically with "
            "scripts/ml_validation.py.\n"
            "  data/ablation.json   ablation and comparison against a "
            "conventional two-center tight-binding model.\n\n"
            "Code: see the GitHub repository cited in the paper.\n")
        add_bytes('README.txt', readme.encode())
        z.writestr('MANIFEST.json', json.dumps(manifest, indent=1))
    print(f'wrote {OUT} with {len(manifest)} files, '
          f'{os.path.getsize(OUT)/1e6:.1f} MB')


if __name__ == '__main__':
    main()
