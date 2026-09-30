# Learned Hamiltonian and Optics of Defective MoS2: Simulation Code

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Code for the paper **"Learning the quantum Hamiltonian of defective monolayer
MoS2 reveals collective vacancy brightness decoupled from defect count."**

**The paper's author list is not recorded in this repository**, and neither
is its publication status. The code itself was committed by Tanvir M. Mahim
(BRAC University), who is the only author in the git history.

- Repository: https://github.com/Tanvir-Mahmud-Mahim/mos2-vacancy-optics
- Archived data: the earlier README says the benchmark dataset is archived
  on Zenodo and that the DOI is given in the paper. The DOI is not recorded
  in this repository, so it is not repeated here.

---

## Contents

1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [What is in this repository](#2-what-is-in-this-repository)
3. [Installation](#3-installation)
4. [Quick start: three ways to use the code](#4-quick-start-three-ways-to-use-the-code)
5. [The scripts, step by step](#5-the-scripts-step-by-step)
6. [Which script makes which figure](#6-which-script-makes-which-figure)
7. [The Python modules](#7-the-python-modules)
8. [Where the numbers come from](#8-where-the-numbers-come-from)
9. [Built-in checks](#9-built-in-checks)
10. [Notes on the calculations](#10-notes-on-the-calculations)
11. [Version history](#11-version-history)
12. [How to cite](#12-how-to-cite)
13. [License and contact](#13-license-and-contact)

---

## 1. The idea in one minute

Monolayer MoS2 is a semiconductor sheet only three atoms thick: a layer of
molybdenum (Mo) atoms between two layers of sulfur (S) atoms. The most common
flaw in real sheets is a **sulfur vacancy**, a spot where one S atom is
missing. Vacancies add electronic states inside the band gap (the energy range
where the perfect crystal has no states). These are called **mid-gap states**.
They can also make the sheet absorb light at photon energies below the gap.

The code computes both effects from one quantum-mechanical description of the
electrons. This description is the **Hamiltonian** H, a matrix that gives the
electron energies. Because the atomic orbitals used here overlap, it comes
together with an **overlap matrix** S. The steps are:

- **Build model sheets with vacancies.** Each model is a repeating patch of the
  crystal (a **supercell**) of 4 x 4 or 5 x 5 unit cells, with 0 to 3 sulfur
  atoms removed and the atoms shaken slightly at random.
- **Compute H and S with density-functional theory (DFT).** DFT is the standard
  first-principles method for electrons in materials. Here it is run with
  the program GPAW, using atom-centred orbitals.
- **Read out two things from the same H and S.** The first is the electronic
  structure (energy levels and mid-gap states). The second is the optical
  conductivity, which says how strongly the sheet absorbs light at each photon
  energy. It is computed with the **Kubo formula**, the standard linear-response
  expression for conductivity. The main optical number is the **sub-gap
  absorption** A_sub: the optical conductivity added up over photon energies
  below the band gap.
- **Learn H and S with machine learning.** Small neural networks predict each
  block of H and S from the positions of nearby atoms. They are compared with
  a conventional two-centre tight-binding model (hopping values that depend only
  on the distance between two atoms). For the spectral test, a separate set of
  networks is trained on 16 5 x 5 cells computed at the zone centre only (a
  single k-point). Those networks then predict 5 further 5 x 5 structures
  that were kept out of their training.

According to the earlier README of this repository, the paper finds the
following:

- The vacancies open a band of sub-gap absorption. Its **average** strength
  grows with the vacancy density.
- At a fixed number of vacancies, the sub-gap brightness varies by more than
  two orders of magnitude between arrangements, from optically dark to
  bright. Meanwhile, the number of mid-gap states barely changes.
- The brightness is set by the character of the defect wavefunctions (the
  shape of the electron states around the vacancies), not by how many such
  states there are. Light absorption is therefore a **selective rather than
  a counting probe** of the vacancies.
- A dilute, isolated pair of vacancies is dark. Brightness requires the
  vacancy wavefunctions to **hybridize** (mix) into an overlapping defect
  band. This is the "collective" property of the whole set of vacancies
  named in the title.
- On the 5 held-out 5 x 5 cells, the fully learned H and S predict the band
  gaps to 73 meV on average and reproduce the brightness ordering from the
  atomic geometry alone.

These are the paper's results as summarised in the earlier README. This
repository does not contain the data files needed to check them without
re-running the calculations.

---

## 2. What is in this repository

```
mos2-vacancy-optics/
|-- README.md            this guide
|-- CHANGELOG.md         what changed, newest first
|-- CITATION.cff         citation details (drives the "Cite this repository" button)
|-- LICENSE              Apache-2.0 license
|-- requirements.txt     Python packages to install
|-- run_pipeline.sh      runs the main steps in order (Linux and macOS; see Section 4)
|-- src/mos2hamop/       the library (the Python package mos2hamop)
|   |-- structures.py    builds the MoS2 supercells with sulfur vacancies (uses ASE)
|   |-- dftrun.py        runs one GPAW DFT calculation and saves H(k), S(k)
|   |-- blocks.py        turns H(k), S(k) into real-space blocks per atom pair
|   |-- rotations.py     rotates orbital blocks about the vertical axis (uses GPAW)
|   |-- features.py      describes each atom pair by its neighbourhood (340 numbers)
|   |-- reference.py     average block versus distance (the starting guess for learning)
|   |-- mlmodel.py       one small neural network per block type (uses PyTorch)
|   |-- assemble.py      predicts and assembles H and S for any structure
|   |-- overlap.py       exact overlap S from GPAW without a full DFT run
|   |-- kubo.py          Kubo optical conductivity from H and S
|   |-- eigsolve.py      stable solver when S is nearly singular
|   |-- negf.py          electron transmission through a device (Green's functions)
|   `-- device.py        cuts a ribbon into slices for negf.py
|-- scripts/             data generation, analysis, tables and figures (Section 5)
|-- tests/               three numerical check scripts (Section 9)
`-- figures/             the eight figure PDFs as committed by the author
```

The scripts write their results to a folder `data/` inside the repository.
This includes the LaTeX number and table files written by `gen_numbers.py`,
`gen_tables.py` and `gen_ablation_table.py` (`data/numbers.tex`,
`data/tab_ml.tex`, `data/tab_configs.tex`, `data/tab_ablation.tex`), which
sit next to the analysis files they are made from. `data/` is not stored on
GitHub (it is listed in `.gitignore`); it is created by the scripts that
produce the analysis files. No script writes outside the repository.

---

## 3. Installation

The earlier README asks for **Python 3.11**. The lightweight parts listed in
Section 4, Way A were run here with Python 3.11.15.

```
pip install -r requirements.txt
```

This installs `ase` (atomic structures, 3.23 or newer), `gpaw` (the DFT
program, 25.1 or newer), `numpy` (2.0 or newer), `scipy` (1.13 or newer),
`torch` (PyTorch, for the neural networks; 2.3 or newer, 2.4.1 or newer on
Windows), `matplotlib` (3.8.4 or newer), and `spglib` (2.0 or newer). The
repository's own code does not import `spglib` directly.

Things to know before installing:

- **Why these minimum versions.** `dft_analysis.py`, `analyze_sep.py` and
  `spectral_validation.py` call `numpy.trapezoid`, which first appeared in
  NumPy 2.0, so NumPy 2.0 is the minimum. The other minimums are the first
  releases that work with NumPy 2: SciPy 1.13 and Matplotlib 3.8.4 (older
  SciPy releases and Matplotlib 3.7.3 to 3.8.3 declare `numpy<2` or a
  similar limit; Matplotlib 3.7.0 to 3.7.2 declare none but fail to import
  with NumPy 2), PyTorch 2.3 (2.2.2
  fails here with "Numpy is not available"; on Windows the PyTorch wheels
  work with NumPy 2 only from 2.4.1), ASE 3.23 (ASE 3.22.1 fails in
  `ase.build.mx2`, which `structures.py` uses, because it calls
  `numpy.product`, removed in NumPy 2), and GPAW 25.1 (GPAW 24.6.0 requires
  `numpy<2`; its release notes say "GPAW almost works with numpy-2, but not
  quite"; 25.1.0 is the next release and drops that limit).
- **Checked with the minimum versions** (30 September 2026, Python 3.11,
  numpy 2.0.0, scipy 1.13.0, matplotlib 3.8.4, torch 2.3.0, ase 3.23.0):
  the three commands of Way A below ran (`max |T-1| in band: 9.07e-06` and
  `55 train structures, 6 test structures`, as with current versions, and
  `fig1_concept.pdf` was written), `structures.make_structure`
  built a 4 x 4 cell with two vacancies through `ase.build.mx2`, and
  `gen_numbers.py`, `gen_tables.py` and `gen_ablation_table.py` ran on
  made-up test input. GPAW could not be checked: building it needs libxc
  and BLAS, which are not installed on the machine used, so nothing that
  imports GPAW was run.
- **GPAW is compiled when pip installs it.** You need a working C/C++
  compiler. On the machine used to check this guide, the build failed
  because the C++ standard headers were missing. Follow the GPAW installation
  guide for your system if pip fails.
- **GPAW needs its atomic data files** (PAW setups and the `dzp` orbital
  basis). `dftrun.py` looks for them in the folder named by the environment
  variable `GPAW_SETUP_PATH`. If that variable is not set, it uses
  `~/gpaw-data/setups`.
- **Most learning scripts import GPAW even though they run no DFT.** This is
  because `rotations.py` uses GPAW's own rotation matrices. The table below
  shows what each group of scripts needs.

| Scripts | Needs |
|---|---|
| `gen_dataset.py`, `gen_separation.py`, `gen_gamma55.py`, `mos2hamop/overlap.py` | GPAW with its data files, ASE (runs DFT) |
| `build_samples.py`, `train_models.py`, `ml_validation.py`, `ml_ablation.py`, `spectral_validation.py`, `eval_test.py`, `range_analysis.py`, `transport_series.py`, `tests/test_blocks.py` | GPAW (imported only), PyTorch, NumPy; some also ASE |
| `dft_analysis.py`, `analyze_sep.py` | NumPy, SciPy, ASE |
| all `fig*.py`, `gen_numbers.py`, `gen_tables.py`, `gen_ablation_table.py`, `hybridization_split.py`, `pack_*.py`, `manifest.py`, `tests/test_negf_chain.py` | NumPy (and Matplotlib for figures) |

**Fonts (optional).** `scripts/figstyle.py` uses Times New Roman when
`times.ttf`, `timesbd.ttf` and `timesi.ttf` are in a folder named `fonts/` in
the repository folder (this folder is not on GitHub). Otherwise Matplotlib
prints many `findfont: Font family 'Times New Roman' not found` warnings and
falls back to DejaVu Sans. The figure is still written.

**Threads.** `run_pipeline.sh` sets `OMP_NUM_THREADS=2` unless you have
already set it.

---

## 4. Quick start: three ways to use the code

Run all commands from the repository folder.

### Way A: check that the lightweight parts work (a few seconds)

These need only NumPy and Matplotlib, and no DFT data:

```
python tests/test_negf_chain.py
python scripts/manifest.py
python scripts/fig1_concept.py
```

- `test_negf_chain.py` prints the transmission of a perfect one-dimensional
  chain. It should be 1 at every listed energy. Measured here:
  `max |T-1| in band: 9.07e-06`, 0.4 s.
- `manifest.py` prints `55 train structures, 6 test structures` and the first
  few entries. Measured here: under 0.1 s.
- `fig1_concept.py` redraws the concept figure, which is a drawing and
  needs no data. Measured here: 6 s. **It overwrites
  `figures/fig1_concept.pdf`** and also writes `figures/fig1_concept.png`.

All three were measured on a shared two-core machine.

### Way B: redraw figures from archived results

This needs the analysis outputs in `data/`. `scripts/pack_zenodo.py` writes
an archive `mos2-vacancy-optics-benchmark.zip` whose files sit under `data/`:
`dft_analysis.json`, `dft_spectra.npz`, `separation.json`,
`ml_report.json`, `ablation.json`, `models.pkl`, `refs.pkl`,
`ml_parity.npz`, and the `train/` and `sep/` structure files, which contain
the eigenvalues but not H and S. Unzipping that archive in the repository
folder therefore fills `data/`. `ml_parity.npz` is a 400,000-point sample
drawn with a fixed random seed, not the full set. This guide could not check
the copy deposited on Zenodo, because its DOI is not recorded here.

With those files in `data/`, these commands need no DFT and no GPAW:

```
python scripts/fig0_abstract.py
python scripts/fig2_ml.py
python scripts/fig3_optics.py
python scripts/fig4_coupling.py
python scripts/figS_validation.py
python scripts/fig_ablation.py
python scripts/hybridization_split.py
python scripts/gen_numbers.py
python scripts/gen_tables.py
python scripts/gen_ablation_table.py
```

Figure 5 (`fig5_spectral.py`) needs `data/spectral_validation.json` and
`data/spectral_validation.npz`. `scripts/pack_gamma55.py` puts these files in
a separate archive (`gamma55-spectral-benchmark.zip`), at the top level of
the zip. Copy them into `data/` first.

### Way C: recompute everything from scratch (long)

The DFT steps dominate the cost. They were not re-timed for this guide.
`pack_zenodo.py` notes that the raw H(k), S(k) files are about 100 MB per
4 x 4 configuration and about 3 GB in total.

**Option 1: `run_pipeline.sh`** (Linux and macOS):

```
./run_pipeline.sh
```

The script stops at the first error (`set -e`). `run_pipeline.sh` leaves out
the ablation and the 5 x 5 spectral read-out, so it does not make
`fig_ablation.pdf` or `fig5_spectral.pdf`. For those, use Option 2.

**Option 2: the full sequence by hand.** This is the order the files depend
on each other:

```
# DFT data (long)
python scripts/gen_dataset.py 0 55 train      # 55 training structures, 4x4 cells
python scripts/gen_dataset.py 0 6 test        # 6 test structures (only for eval_test.py, range_analysis.py)
python scripts/gen_separation.py              # two-vacancy separation series, 5x5 cells
python scripts/gen_gamma55.py train 0 16      # 16 zone-centre 5x5 training structures
python scripts/gen_gamma55.py test 0 5        # 5 zone-centre 5x5 test structures

# optics and electronic structure straight from DFT
python scripts/dft_analysis.py
python scripts/analyze_sep.py
python scripts/hybridization_split.py

# learned Hamiltonian
python scripts/build_samples.py train
python scripts/train_models.py
python scripts/ml_validation.py
python scripts/ml_ablation.py
python scripts/build_samples.py train55
python scripts/spectral_validation.py

# numbers, tables, figures
python scripts/gen_numbers.py
python scripts/gen_tables.py
python scripts/gen_ablation_table.py
python scripts/fig0_abstract.py
python scripts/fig1_concept.py
python scripts/fig2_ml.py
python scripts/fig3_optics.py
python scripts/fig4_coupling.py
python scripts/fig5_spectral.py
python scripts/figS_validation.py
python scripts/fig_ablation.py
```

The DFT workers skip any structure whose output file already exists. You can
therefore split a slice across several runs, for example
`gen_dataset.py 0 20 train` and `gen_dataset.py 20 55 train`.
`gen_dataset.py idx train 3,7,12` runs chosen manifest entries only.

---

## 5. The scripts, step by step

"Not re-timed" means the step needs DFT data or GPAW, which could not be
run for this guide. The repository itself records no run times.

| Command | What it does | Time | Results |
|---|---|---|---|
| `python scripts/gen_dataset.py START END [train\|test]` | Builds entries START to END-1 of the 4 x 4 manifest (`manifest.py`) and runs DFT on each | long (not re-timed) | `data/train/*.npz` or `data/test/*.npz` |
| `python scripts/gen_separation.py [ranks]` | Two top-layer vacancies in a 5 x 5 cell, from touching to far apart (default partner ranks 0, 2, 5, 9, 14), and runs DFT | long (not re-timed) | `data/sep/sep5_*.npz` |
| `python scripts/gen_gamma55.py train\|test I J` | 5 x 5 structures computed at the zone centre only (a single k-point) | long (not re-timed) | `data/train55/`, `data/test55/` |
| `python scripts/gen_gamma55.py probe` | One pristine 5 x 5 run, used to measure how long a run takes | long (not re-timed) | `data/probe55/` |
| `python scripts/dft_analysis.py` | For every 4 x 4 training structure: Kubo optical conductivity, sub-gap absorption A_sub, near-zero-frequency conductivity, and number of in-gap states | not re-timed | `data/dft_analysis.json`, `data/dft_spectra.npz` |
| `python scripts/analyze_sep.py` | A_sub versus vacancy separation for the separation series | not re-timed | `data/separation.json` |
| `python scripts/hybridization_split.py` | Width of the band of mid-gap levels versus separation (printed only) | not re-timed | screen |
| `python scripts/build_samples.py train\|test\|train55\|test55` | Cuts H and S into per-atom-pair blocks, rotates them into the pair frame, and computes the pair descriptors | not re-timed | `data/samples_<name>/` |
| `python scripts/train_models.py` | Trains the five block-type networks for H and S on all training structures | not re-timed | `data/models.pkl`, `data/refs.pkl`, `data/train_report.pkl` |
| `python scripts/ml_validation.py` | Retrains on 85% of the structures and measures the error on the other 15% | not re-timed | `data/ml_report.json`, `data/ml_parity.npz` |
| `python scripts/ml_ablation.py` | Five model variants on the same 85/15 split, including the two-centre tight-binding baseline | not re-timed | `data/ablation.json` |
| `python scripts/spectral_validation.py` | Trains on the 5 x 5 zone-centre set (unless `models55.pkl` exists) and predicts gaps, eigenvalues and A_sub of the 5 held-out structures | not re-timed | `data/models55.pkl`, `data/refs55.pkl`, `data/spectral_validation.json`, `.npz` |
| `python scripts/gen_numbers.py` | Writes key numbers from the analysis files as LaTeX macros | not re-timed | `data/numbers.tex` |
| `python scripts/gen_tables.py` | Writes the held-out error table and the per-configuration table | not re-timed | `data/tab_ml.tex`, `data/tab_configs.tex` |
| `python scripts/gen_ablation_table.py` | Writes the ablation table | not re-timed | `data/tab_ablation.tex` |
| `python scripts/fig*.py` | Draws the figures (Section 6) | fig1: 6 s; others not re-timed | `figures/` |
| `python scripts/eval_test.py` | Optional: errors of the trained 4 x 4 models on the 6 test structures | not re-timed | `data/eval_test.json`, `data/eval_parity.npz`, `data/eval_eigs.npz` |
| `python scripts/range_analysis.py` | Optional: shows why the 4 x 4 cells with the 2 x 2 k-point grid cannot be read out spectrally by an 11 A local model (printed only) | not re-timed | screen |
| `python scripts/transport_series.py` | Optional: transmission through a ribbon (24 x 4 cells) versus vacancy count, built from the learned model | not re-timed | `data/transport.json` |
| `python scripts/pack_zenodo.py` | Packs the 4 x 4 benchmark without the raw H, S | not re-timed | `mos2-vacancy-optics-benchmark.zip` |
| `python scripts/pack_gamma55.py` | Packs the 5 x 5 spectral benchmark | not re-timed | `data/gamma55-spectral-benchmark.zip` |
| `python scripts/manifest.py` | Prints the size of the 4 x 4 training and test manifests | under 0.1 s | screen |

`build_samples.py test55` appears in the earlier README, but no script reads
its output (`data/samples_test55/`). `spectral_validation.py` reads the raw
`data/test55/` files directly. `eval_test.py`, `range_analysis.py` and
`transport_series.py` are not called by `run_pipeline.sh`, and no figure
uses their output.

Random choices (which S atoms are removed, how atoms are shaken, the
train/test split, network initialisation) all use fixed seeds written in the
scripts.

---

## 6. Which script makes which figure

The figure labels are taken from each script's own description. The
committed PDFs in `figures/` are the author's versions.

| File | Label in the script | Content | Data needed | Drawn by |
|---|---|---|---|---|
| `fig0_abstract.pdf` | Graphical abstract | Vacancy structure and sub-gap spectra | `dft_spectra.npz` | `fig0_abstract.py` |
| `fig1_concept.pdf` | Figure 1 | Concept: structure, one Hamiltonian, two read-outs (drawn, not computed) | none | `fig1_concept.py` |
| `fig2_ml.pdf` | Figure 2 | Learned versus DFT matrix elements; error versus distance; error per block type | `ml_parity.npz`, `ml_report.json` | `fig2_ml.py` |
| `fig3_optics.pdf` | Figure 3 | Spectra by vacancy count; A_sub versus density; A_sub versus in-gap states | `dft_spectra.npz`, `dft_analysis.json` | `fig3_optics.py` |
| `fig4_coupling.pdf` | Figure 4 | In-gap states and near-zero-frequency conductivity versus density; isolated pair is dark; level splitting versus separation | `dft_analysis.json`, `separation.json`, `data/sep/*.npz` | `fig4_coupling.py` |
| `fig5_spectral.pdf` | Figure 5 | Learned versus DFT spectra, A_sub and gaps on the 5 x 5 test cells | `spectral_validation.json`, `.npz` | `fig5_spectral.py` |
| `figS_validation.pdf` | Supplementary validation figure | Pristine absorption edge; transmission of a one-dimensional chain | `dft_spectra.npz` | `figS_validation.py` |
| `fig_ablation.pdf` | (no number given) | Ablation and comparison with two-centre tight binding | `ablation.json` | `fig_ablation.py` |

Every figure script also writes a `.png`, except `fig5_spectral.py`, which
writes only the PDF.

---

## 7. The Python modules

| File | What it contains |
|---|---|
| `structures.py` | `supercell`, `sulfur_indices`, `make_structure`: MoS2 supercells from ASE's `mx2` builder; vacancies and random shaking with fixed seeds |
| `dftrun.py` | `run_structure`: GPAW calculation, energies shifted so the vacuum level is zero, and saving of H(k), S(k), eigenvalues, forces and geometry |
| `blocks.py` | Orbital counts (Mo 29, S 13); inverse Fourier transform H(k) to H(R); cutting per-pair blocks at the nearest periodic copy |
| `rotations.py` | Rotation of orbital blocks about the vertical axis, using GPAW's `gpaw.rotation` |
| `features.py` | Pair descriptor: 4 distance numbers plus a smoothed neighbour cloud around each end of the pair, in the pair's own frame (340 numbers in total) |
| `reference.py` | `DistanceReference`: mean block per distance bin (one bin for on-site blocks); the networks learn only the difference from it |
| `mlmodel.py` | Five block types; `BlockMLP` (two hidden layers of 320 units, SiLU activation); `BlockModel` training with Adam and early stopping; on-site blocks use the full descriptor, pair blocks only the 4 distance numbers |
| `assemble.py` | Lists pairs within 11 A, predicts blocks, rotates them back, and averages each block with its reverse so H and S stay symmetric |
| `overlap.py` | Exact S from a single non-converged GPAW step (S does not depend on the self-consistent density) |
| `kubo.py` | `bloch_matrices`, `sigma_xx`: Kubo-Greenwood conductivity with Gaussian broadening, in units of e^2/(4 hbar) |
| `eigsolve.py` | `gen_eigh`: solves H c = E S c after dropping directions where S has eigenvalues below a threshold (canonical orthogonalization) |
| `negf.py` | Surface Green's function of a semi-infinite lead (Sancho-Rubio method) and transmission through a chain of slices (recursive Green's function) |
| `device.py` | `build_layers`: splits a ribbon into principal layers (slices) and collects the couplings between neighbouring slices |

---

## 8. Where the numbers come from

All values below are as written in the code. The code gives no literature
source for them; the lattice values are labelled "PBE" in the comments.

**Structure** (`structures.py`, `atomrender.py`):

| Quantity | Value |
|---|---|
| In-plane lattice constant | 3.184 A ("PBE in-plane lattice constant") |
| Vertical S-S distance | 3.127 A ("PBE S-S vertical distance") |
| Vacuum on each side of the sheet | 5.5 A (see Section 10) |
| 4 x 4 supercell | 16 Mo + 32 S = 48 atoms before vacancies (checked here) |
| 5 x 5 supercell | 75 atoms before vacancies (checked here) |
| Random shaking ("rattle") | Gaussian, standard deviation 0 to 0.06 A, set per structure in the manifests |
| Strain (4 x 4 set) | -1% or +1% (test: +0.8%) |

**DFT settings** (`dftrun.py`): GPAW, atom-centred orbitals (`mode='lcao'`)
with the `dzp` basis, PBE exchange-correlation, grid spacing 0.24 A,
Fermi-Dirac smearing 0.01 eV, symmetry off. The k-point grid is centred on
the zone centre: 2 x 2 x 1 by default (4 x 4 set and separation series) and
1 x 1 x 1 for the 5 x 5 zone-centre set.

**Datasets** (`manifest.py`, `gen_gamma55.py`, `gen_separation.py`):

| Set | Cell | Structures | Content |
|---|---|---|---|
| `train` | 4 x 4 | 55 | 11 pristine, 16 with 1 vacancy (4 in the bottom layer), 14 with 2, 6 with 3, 8 strained (4 pristine, 4 with 1 vacancy) |
| `test` | 4 x 4 | 6 | 1 pristine, 1 with 1 vacancy, 2 with 2, 1 with 3, 1 strained with 1 vacancy |
| `train55` | 5 x 5, zone centre | 16 | 4 pristine, 5 with 1 vacancy, 4 with 2, 3 with 3 |
| `test55` | 5 x 5, zone centre | 5 | 1 pristine, 1 with 1 vacancy, 2 with 2, 1 with 3 |
| `sep` | 5 x 5 | 5 | 2 top-layer vacancies at partner ranks 0, 2, 5, 9, 14; shaken by 0.02 A (seed 7) |

The counts of 55, 6, 16 and 5 were checked by running the manifest code.

**Optics and electronic structure** (`dft_analysis.py`, `analyze_sep.py`,
`spectral_validation.py`):

| Quantity | Value |
|---|---|
| Photon energies | 0.05 to 3.0 eV, 90 points |
| Broadening | 0.08 eV (Gaussian) |
| Electron temperature in the Kubo formula | 300 K (default of `sigma_xx`) |
| A_sub window | 0.15 eV to (pristine gap - 0.2 eV) |
| Near-zero-frequency conductivity | value at the grid point closest to 0.1 eV |
| In-gap states | eigenvalues between (pristine valence top + 0.1 eV) and (pristine conduction bottom - 0.1 eV), per k-point |
| Vacancy density | vacancies / 32 S sites x 100% (4 x 4 cells) |
| Electron count | 14 per Mo and 6 per S (valence electrons) |

**Learned Hamiltonian** (`features.py`, `mlmodel.py`, `train_models.py`,
`ml_validation.py`, `ml_ablation.py`, `spectral_validation.py`):

| Quantity | Value |
|---|---|
| Pair range | 11 A |
| Neighbour radius in the descriptor | 6 A, 6 radial Gaussians (width 0.9 A), angular order 3 |
| Descriptor length | 340 (checked here) |
| Distance-reference bins | 24 |
| Network | 2 hidden layers of 320 units, SiLU; Adam, learning rate 1e-3, batch 512, 10% of blocks held back for early stopping |
| Epochs (maximum) | H: 350, S: 250 (`train_models.py`, `spectral_validation.py`); 300 (`ml_validation.py`); 220 (`ml_ablation.py`) |
| Held-out split | 85% / 15% by structure, random seed 0 |
| Overlap threshold | 1e-4 by default; 0.1 for the learned-versus-DFT comparison in `spectral_validation.py` |

---

## 9. Built-in checks

The three scripts in `tests/` print numbers for you to read. None of them
stops with an error when a value is off, so there is no automatic pass/fail.

- **`tests/test_negf_chain.py`** (no data needed). It computes the
  transmission of a clean one-dimensional chain, which should be exactly 1
  inside the band. Measured here: largest deviation 9.07e-06. It also prints
  the transmission of a two-channel chain with one raised slice: 1.4706 at
  zero energy, where 2 would be the value without the barrier.
- **`tests/test_blocks.py`** (needs `data/train/prist_000.npz` and GPAW). It
  checks that H(R) transforms back to the stored H(k), that H(R) is real,
  and that H is symmetric between pairs (i, j) and (j, i). It also checks
  that equivalent Mo-S and Mo-Mo pairs give the same block once rotated into
  their own frame.
- **`tests/test_kubo_pristine.py`** (needs `data/kubo_prim.npz`). It computes
  the optical conductivity of the perfect crystal on a dense 18 x 18 k-point
  grid and prints the smallest direct gap and the conductivity below and
  above the edge. No script in this repository creates `data/kubo_prim.npz`.
  The file is opened by a relative path, so run the test from the
  repository folder.

Other checks inside the scripts:

- `build_samples.py` prints and saves the largest imaginary part of H(R),
  which should be close to zero because the orbitals are real.
- `transport_series.py` prints `maxskip`, the largest slice-to-slice jump with
  a nonzero coupling. The method assumes that only neighbouring slices
  couple, which means `maxskip` should be 1.
- `spectral_validation.py` records the gap and A_sub three ways: the exact
  DFT result, DFT in the reduced subspace, and the learned model. The effect
  of the subspace reduction itself is therefore visible.
- `range_analysis.py` shows how the eigenvalues change when the exact DFT
  blocks are cut off at 11 A.

---

## 10. Notes on the calculations

- **Units.** Energies are in eV, lengths in angstrom (A), k in 1/A. The
  optical conductivity is per MoS2 layer, in units of e^2/(4 hbar). A_sub is
  the conductivity integrated over photon energy (trapezoid rule), so its unit
  is eV x e^2/(4 hbar).
- **Energy reference.** `dftrun.py` shifts every H so that the vacuum level
  is zero. It takes the vacuum level from the plane-averaged electrostatic
  potential near the cell edge. As a result, all structures share one energy
  zero.
- **Chemical potential.** `dft_analysis.py` and `analyze_sep.py` use the DFT
  Fermi level. `spectral_validation.py` places it in the middle of the gap,
  using the electron count, in the same way for DFT and for the learned
  model.
- **Optics approximation.** The velocity operator uses only the distances
  between atoms. Dipoles within a single atom are neglected, as the
  `kubo.py` description states. Only the x-direction conductivity
  (sigma_xx) is computed.
- **Real-space blocks.** A calculation on a 2 x 2 k-point grid defines H(R) on
  a torus two supercells wide. Each pair is taken at its nearest periodic
  copy. With 4 x 4 cells the largest such pair distance is 14.79 A. With
  5 x 5 cells at the zone centre only it is 9.32 A, inside the 11 A learning
  range. Both values were checked here and match the scripts' own statements
  (14.8 A and about 9.3 A).
- **Nearly singular overlap.** The `dzp` basis gives an overlap matrix with
  eigenvalues close to zero. Small errors along those directions are
  amplified, so `eigsolve.py` drops them. `spectral_validation.py` uses a
  larger threshold (0.1) for both DFT and the learned model, so that both are
  compared in the same subspace.
- **Proposed model.** On-site blocks use the full neighbourhood descriptor.
  Pair blocks use only the 4 distance numbers. In `ml_ablation.py` these are
  variants E and D; the "proposed" rows of `fig_ablation.py` and
  `gen_ablation_table.py` combine E for on-site blocks with D for pair
  blocks.
- **Vacuum value.** The code uses 5.5 A of vacuum on each side (checked
  here: the 4 x 4 cell is 14.13 A tall). The text descriptions in
  `structures.py` and `dftrun.py` say 7.5 A, which does not match the code.
- **Values fixed inside `gen_numbers.py`.** `subgapPeak` is always written
  as 1.3. If `ml_report.json` or `ablation.json` is missing, the script
  writes built-in fallback numbers instead of stopping. Make sure both files
  exist before trusting `data/numbers.tex`.
- **License file.** `LICENSE` is the standard Apache 2.0 text. The
  copyright line in its appendix (line 190) is still the template
  `Copyright [yyyy] [name of copyright owner]`, so the file names no
  copyright holder or year.
- **Titles in older files.** `run_pipeline.sh` and the text written by
  `pack_zenodo.py` still carry earlier working titles of the paper. The
  title above is the one in the latest README commit and in `pack_gamma55.py`.

---

## 11. Version history

The repository has no tags or version numbers. All code, tests and figures
were committed on 4 September 2026 (18 commits). A documentation update and
a set of fixes (LaTeX files now written to `data/` instead of `paper/`;
corrected minimum versions in `requirements.txt`), both on 30 September
2026, followed. Details are in [CHANGELOG.md](CHANGELOG.md).

---

## 12. How to cite

Please cite the paper and this code. GitHub shows a **"Cite this
repository"** button in the right-hand column, which reads `CITATION.cff`.

> "Learning the quantum Hamiltonian of defective monolayer MoS2 reveals
> collective vacancy brightness decoupled from defect count."
> (Author list, venue and DOI are not recorded in this repository.)
>
> Code (committed by T. M. Mahim; the paper's author list is not recorded
> here): mos2-vacancy-optics,
> https://github.com/Tanvir-Mahmud-Mahim/mos2-vacancy-optics

---

## 13. License and contact

Code: Apache License 2.0 (see `LICENSE`).

Questions and bug reports: please open an issue on this repository, or
contact Tanvir M. Mahim, BRAC University (tanvir.mahim@bracu.ac.bd).
