# Learned Hamiltonian and Optics of Defective MoS2: Simulation Code

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This repository holds the code for the paper **"Learning the quantum Hamiltonian of defective monolayer
MoS2 reveals collective vacancy brightness decoupled from defect count."**

**The paper's author list is not recorded in this repository**, and neither
is its publication status. I, Tanvir M. Mahim (BRAC University), committed
the code, and I am the only author in the git history.

- Repository: https://github.com/Tanvir-Mahmud-Mahim/mos2-vacancy-optics
- Archived data: the earlier README says the benchmark dataset is archived
  on Zenodo and that the DOI is given in the paper. The DOI is not recorded
  in this repository, so I do not repeat it here.

---

## Contents

1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [What is in this repository](#2-what-is-in-this-repository)
3. [Installation](#3-installation)
4. [Quick start: three ways to use the code](#4-quick-start-three-ways-to-use-the-code)
5. [The scripts, step by step](DETAILS.md#5-the-scripts-step-by-step)
6. [Which script makes which figure](DETAILS.md#6-which-script-makes-which-figure)
7. [The Python modules](DETAILS.md#7-the-python-modules)
8. [Where the numbers come from](DETAILS.md#8-where-the-numbers-come-from)
9. [Built-in checks](DETAILS.md#9-built-in-checks)
10. [Notes on the calculations](DETAILS.md#10-notes-on-the-calculations)
11. [Version history](DETAILS.md#11-version-history)
12. [How to cite](#12-how-to-cite)
13. [License and contact](#13-license-and-contact)

Extra notes for Sections 1 to 4 are in
[DETAILS.md](DETAILS.md#extra-notes-for-sections-1-to-4).

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
electron energies. The atomic orbitals used here overlap, so H comes together
with an **overlap matrix** S. Here are the steps:

- Build model sheets of 4 x 4 or 5 x 5 unit cells, with 0 to 3 sulfur atoms
  removed.
- Compute H and S with density-functional theory (DFT), using the program GPAW.
- From the same H and S, read out the mid-gap states and the **sub-gap
  absorption** A_sub: the optical conductivity added up over photon energies
  below the band gap.
- Learn H and S with small neural networks, from the positions of nearby atoms.

The earlier README of this repository describes the paper's findings as
follows:

- At a fixed number of vacancies, the sub-gap brightness varies by more than
  two orders of magnitude between arrangements, from optically dark to
  bright. Meanwhile, the number of mid-gap states barely changes.
- The brightness is set by the character of the defect wavefunctions (the
  shape of the electron states around the vacancies), not by how many such
  states there are. So light absorption is a **selective rather than
  a counting probe** of the vacancies.
- On the 5 held-out 5 x 5 cells, the fully learned H and S predict the band
  gaps to 73 meV on average. They also reproduce the brightness ordering
  from the atomic geometry alone.

These are the paper's results as the earlier README summarized them. This
repository does not contain the data files you would need to check them,
so you would have to re-run the calculations. The full list of steps and
findings is in [DETAILS.md](DETAILS.md#more-on-section-1-the-steps-and-the-findings-in-full).

---

## 2. What is in this repository

- **src/mos2hamop/**: the library (the Python package mos2hamop).
- **scripts/**: data generation, analysis, tables and figures (Section 5).
- **tests/**: three numerical check scripts (Section 9).
- **figures/**: the eight figure PDFs as I committed them.
- **run_pipeline.sh**: runs the main steps in order (Linux and macOS; see Section 4).
- **requirements.txt**: Python packages to install.

The scripts write their results to a folder `data/` inside the repository.
`data/` is not stored on GitHub (it is listed in `.gitignore`). No script
writes outside the repository. The full annotated file tree is in
[DETAILS.md](DETAILS.md#more-on-section-2-the-full-file-tree-and-the-data-folder).

---

## 3. Installation

The earlier README asks for **Python 3.11**. I ran the lightweight parts
listed in Section 4, Way A with Python 3.11.15.

```
pip install -r requirements.txt
```

This installs:

- `ase` (atomic structures, 3.23 or newer)
- `gpaw` (the DFT program, 25.1 or newer)
- `numpy` (2.0 or newer)
- `scipy` (1.13 or newer)
- `torch` (PyTorch, for the neural networks; 2.3 or newer, 2.4.1 or newer on
  Windows)
- `matplotlib` (3.8.4 or newer)
- `spglib` (2.0 or newer). The repository's own code does not import
  `spglib` directly.

A few things you should know before you install:

- **GPAW is compiled when pip installs it.** You need a working C/C++
  compiler. On the machine I used to check this guide, the build failed
  because the C++ standard headers were missing. If pip fails, follow the
  GPAW installation guide for your system.
- **GPAW needs its atomic data files** (PAW setups and the `dzp` orbital
  basis). `dftrun.py` looks for them in the folder named by the environment
  variable `GPAW_SETUP_PATH`. If you have not set that variable, it uses
  `~/gpaw-data/setups`.

More notes are in
[DETAILS.md](DETAILS.md#more-on-section-3-versions-checks-fonts-and-threads):
why these minimum versions, my checks with them, what each group of scripts
needs, fonts, and threads.

---

## 4. Quick start: three ways to use the code

Run all commands from the repository folder.

### Way A: check that the lightweight parts work (a few seconds)

You need only NumPy and Matplotlib for these, and no DFT data:

```
python tests/test_negf_chain.py
python scripts/manifest.py
python scripts/fig1_concept.py
```

- `test_negf_chain.py` prints the transmission of a perfect one-dimensional
  chain. It should be 1 at every listed energy.
- `manifest.py` prints `55 train structures, 6 test structures` and the first
  few entries.
- `fig1_concept.py` redraws the concept figure. **It overwrites
  `figures/fig1_concept.pdf`** and also writes `figures/fig1_concept.png`.

My outputs and timings for these three are in
[DETAILS.md](DETAILS.md#more-on-section-4-outputs-timings-and-archive-contents).

### Way B: redraw figures from archived results

For this you need the analysis outputs in `data/`. `scripts/pack_zenodo.py`
writes an archive `mos2-vacancy-optics-benchmark.zip`. If you unzip that
archive in the repository folder, it fills `data/`. The list of files in the
archive is in [DETAILS.md](DETAILS.md#more-on-section-4-outputs-timings-and-archive-contents).

Once those files are in `data/`, you can run these commands without DFT and
without GPAW:

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

The DFT steps take most of the time.
`pack_zenodo.py` notes that the raw H(k), S(k) files are about 100 MB per
4 x 4 configuration and about 3 GB in total.

**Option 1: `run_pipeline.sh`** (Linux and macOS):

```
./run_pipeline.sh
```

The script stops at the first error (`set -e`). `run_pipeline.sh` leaves out
the ablation and the 5 x 5 spectral read-out, so it does not make
`fig_ablation.pdf` or `fig5_spectral.pdf`. For those, use Option 2.

**Option 2: the full sequence by hand.** The commands below follow the order
in which the files depend on each other:

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

The DFT workers skip any structure whose output file already exists. So you
can split a slice across several runs, for example
`gen_dataset.py 0 20 train` and `gen_dataset.py 20 55 train`.
`gen_dataset.py idx train 3,7,12` runs only the manifest entries you choose.

---

## More details (Sections 5 to 11)

The full notes are in [DETAILS.md](DETAILS.md). There you find:

- [5. The scripts, step by step](DETAILS.md#5-the-scripts-step-by-step):
  a table of every script: what it does, its time and its results.
- [6. Which script makes which figure](DETAILS.md#6-which-script-makes-which-figure):
  each figure PDF, its content, the data it needs and the script that draws it.
- [7. The Python modules](DETAILS.md#7-the-python-modules):
  what each file of the library contains.
- [8. Where the numbers come from](DETAILS.md#8-where-the-numbers-come-from):
  the structure, DFT, dataset, optics and learning settings as written in the code.
- [9. Built-in checks](DETAILS.md#9-built-in-checks):
  the three test scripts and the other checks inside the scripts.
- [10. Notes on the calculations](DETAILS.md#10-notes-on-the-calculations):
  units, energy reference, approximations, and notes on a few values and files.
- [11. Version history](DETAILS.md#11-version-history):
  when the code was committed and what changed since.

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

If you have questions or find a bug, please open an issue on this
repository, or contact me, Tanvir M. Mahim, BRAC University
(tanvir.mahim@bracu.ac.bd).
