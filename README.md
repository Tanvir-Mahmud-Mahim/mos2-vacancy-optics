# mos2-vacancy-optics

Open pipeline for the paper **"Optical brightness of sulfur vacancies
decouples from their count in monolayer MoS2."**

Sulfur vacancies are the dominant point defect in monolayer MoS2. This
code builds the density-functional Hamiltonian of vacancy-disordered
MoS2 supercells and, from the *same* Hamiltonian, computes both the Kubo
optical conductivity and the electronic structure, so that the optical
and electronic signatures of a defect configuration are strictly
consistent. It shows that the vacancies open a sub-gap optical absorption
band whose average strength grows with vacancy density, but whose
brightness at a fixed vacancy count varies by more than two orders of
magnitude between configurations, from optically dark to bright, while the
number of mid-gap states barely changes. The brightness is set by the
defect wavefunction character, not the state count, so optics is a
selective rather than a counting probe of the vacancies. A dilute isolated
divacancy is dark, so brightness requires an overlapping defect band. The
code also trains a machine-learned Hamiltonian that reproduces the
reference Hamiltonian at the matrix-element level, benchmarked against a
conventional two-center tight-binding model, as the route to carrying the
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
  mlmodel.py       per-block-type MLP Hamiltonian model (physics-selected
                   descriptor: environment for onsite, two-center for pairs)
  assemble.py      assemble H (and S) for arbitrary geometries
  overlap.py       exact two-center overlap (no SCF)
  kubo.py          Kubo-Greenwood optical / DC conductivity
  negf.py          NEGF transmission (Sancho-Rubio + RGF)
  device.py        principal-layer partitioning
  eigsolve.py      canonical-orthogonalization generalized eigensolver
scripts/           dataset generation, analysis, figures
  atomrender.py    3D atomistic renderer for the structure figures
  ml_ablation.py   ablation and comparison against two-center tight binding
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

# 4. learned Hamiltonian: samples, training, held-out validation, ablation
python scripts/build_samples.py train
python scripts/train_models.py
python scripts/ml_validation.py
python scripts/ml_ablation.py

# 5. figures and numbers
python scripts/fig1_concept.py
python scripts/fig2_ml.py
python scripts/fig_ablation.py
python scripts/fig3_optics.py
python scripts/fig4_coupling.py
python scripts/gen_numbers.py
python scripts/gen_ablation_table.py
```

Numerical validations (real-space transform, rotation equivariance, NEGF
against an analytic chain, Kubo against the pristine absorption edge) are
in `tests/`.

## Data

The benchmark dataset (DFT Hamiltonians and overlaps of the vacancy
configurations, the per-configuration optical and electronic descriptors,
and the trained models) is archived on Zenodo; see the paper for the DOI.

## License

Apache License 2.0. See `LICENSE`.
