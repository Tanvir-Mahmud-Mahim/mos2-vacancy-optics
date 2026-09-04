# mos2-vacancy-optics

Open pipeline for the paper **"Sulfur vacancies leave an
arrangement-sensitive optical fingerprint in monolayer MoS2."**

Sulfur vacancies are the dominant point defect in monolayer MoS2. This
code builds the density-functional Hamiltonian of vacancy-disordered
MoS2 supercells and, from the *same* Hamiltonian, computes both the Kubo
optical conductivity and the electronic structure, so that the optical
and electronic signatures of a defect configuration are strictly
consistent. It shows that the vacancies write a quantitative, sub-gap
optical absorption band whose strength grows with vacancy density and,
at fixed count, with vacancy clustering, making the sub-gap absorption a
contactless optical read-out of the electronic degradation. It also
trains a machine-learned Hamiltonian that reproduces the reference
Hamiltonian at the matrix-element level, the route to carrying the
read-out to device scale.

## What is here

```
src/mos2hamop/     the library
  structures.py    reproducible vacancy supercells (ASE)
  dftrun.py        GPAW LCAO run + H(k), S(k) extraction
  blocks.py        real-space Hamiltonian/overlap blocks
  rotations.py     GPAW real-harmonic rotations (pair-frame equivariance)
  features.py      rotation-invariant pair descriptor
  reference.py     distance-binned reference blocks
  mlmodel.py       per-block-type MLP Hamiltonian model
  assemble.py      assemble H (and S) for arbitrary geometries
  overlap.py       exact two-center overlap (no SCF)
  kubo.py          Kubo-Greenwood optical / DC conductivity
  negf.py          NEGF transmission (Sancho-Rubio + RGF)
  device.py        principal-layer partitioning
  eigsolve.py      canonical-orthogonalization generalized eigensolver
scripts/           dataset generation, analysis, figures
tests/             numerical validation (transforms, NEGF, Kubo)
```

## Reproducing the results

Requirements: Python 3.11, GPAW, ASE, NumPy, SciPy, PyTorch, Matplotlib
(see `requirements.txt`). GPAW ships the PAW datasets and dzp basis used
here.

```bash
# 1. generate the DFT Hamiltonian dataset (sulfur-vacancy configurations)
python scripts/gen_dataset.py 0 55 train

# 2. coupled optical + electronic analysis (Kubo + DOS + transport)
python scripts/dft_analysis.py

# 3. controlled two-vacancy separation series (arrangement dependence)
python scripts/gen_separation.py
python scripts/analyze_sep.py

# 4. learned Hamiltonian: samples, training, held-out validation
python scripts/build_samples.py train
python scripts/train_models.py
python scripts/ml_validation.py

# 5. figures
python scripts/fig1_concept.py scripts/fig2_ml.py \
       scripts/fig3_optics.py scripts/fig4_coupling.py
```

Numerical validations (real-space transform, rotation equivariance, NEGF
against an analytic chain, Kubo against the pristine absorption edge) are
in `tests/`.

## Data

The benchmark dataset (DFT Hamiltonians and overlaps of the vacancy
configurations, the per-configuration optical and electronic descriptors,
and the trained models) is archived on Zenodo; see the paper for the DOI.
The manuscript and Supporting Information are not part of this repository
or the data archive.

## License

Apache License 2.0. See `LICENSE`.
