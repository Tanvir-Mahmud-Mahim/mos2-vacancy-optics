"""Worker: run DFT for a slice of the manifest. Usage: gen_dataset.py START END [test]"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mos2hamop.structures import make_structure
from mos2hamop.dftrun import run_structure
from manifest import build_manifest, build_test_manifest

if sys.argv[1] == 'idx':
    which = sys.argv[2]
    idxs = [int(x) for x in sys.argv[3].split(',')]
    man_full = build_test_manifest() if which == 'test' else build_manifest()
    man = [man_full[i] for i in idxs]
    start, end = 0, len(man)
else:
    start, end = int(sys.argv[1]), int(sys.argv[2])
    which = sys.argv[3] if len(sys.argv) > 3 else 'train'
    man = build_test_manifest() if which == 'test' else build_manifest()
outdir = os.path.join(os.path.dirname(__file__), '..', 'data', which)
os.makedirs(outdir, exist_ok=True)

for name, kw in man[start:end]:
    done = os.path.join(outdir, name + '.npz')
    if os.path.exists(done):
        print('skip', name); continue
    t0 = time.time()
    atoms, meta = make_structure(4, 4, **kw)
    e = run_structure(atoms, name, outdir)
    print(f'{name}: E={e:.3f} eV, {time.time()-t0:.0f} s', flush=True)
print('worker done')

# (indices mode appended)
