# Changelog

All notable changes to this code are listed here, newest first. The
repository has no tags or releases, so entries are dated rather than
numbered.

## Documentation update (30 September 2026)

Documentation only; no code, data or figure was changed.

- README rewritten as a step-by-step guide: plain-language summary, file
  tree, installation notes (NumPy 2.0 needed in practice, which scripts need
  GPAW or PyTorch, where GPAW looks for its data files), three ways to use
  the code, a table of every script with its outputs, which script makes
  which figure, parameter values as written in the code, what the test
  scripts check, and known caveats.
- Documented that `paper/` must be created before `gen_numbers.py`,
  `gen_tables.py`, `gen_ablation_table.py` or `run_pipeline.sh` is run.
- Added this CHANGELOG and a `CITATION.cff`.

## Initial code (4 September 2026, untagged)

Taken from the git history (18 commits, all on 4 September 2026).

- `src/mos2hamop/`: structure builder, GPAW DFT driver, real-space blocks,
  z-rotations, pair descriptor, distance reference, block networks,
  assembly, exact overlap, Kubo optics, NEGF transmission, device slicing,
  and the canonical-orthogonalization eigensolver.
- `tests/`: real-space transform and rotation check, NEGF chain check, and
  Kubo check on pristine MoS2.
- README, Apache-2.0 license, `requirements.txt`, `run_pipeline.sh`.
- Analysis, training, ablation, table, packaging and figure scripts, and the
  figure PDFs.
- Later the same day: on-site blocks use the full descriptor and pair
  blocks only the distance numbers ("physics-selected descriptor"); a more
  robust eigensolver in the Kubo code; a fixed-seed parity subsample in the
  Zenodo packaging; `range_analysis.py` and a revised `eval_test.py`; one
  correlation value used throughout Figure 3.
- Spectral read-out on 5 x 5 zone-centre cells: `gen_gamma55.py`,
  `spectral_validation.py`, `hybridization_split.py`, `pack_gamma55.py`,
  `fig5_spectral.py`; minimum-image pair enumeration and the projected Kubo
  solve in the library; README updated to the current paper title.
- Figure 4 gained the level-splitting panel; the Figure 5 legend moved
  outside the axes.
