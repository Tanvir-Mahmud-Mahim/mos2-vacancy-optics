#!/bin/bash
# End-to-end pipeline for
# "Sulfur vacancies leave an arrangement-sensitive optical fingerprint
#  in monolayer MoS2".
# Requires GPAW, ASE, NumPy, SciPy, PyTorch, Matplotlib (see requirements.txt).
set -e
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$(dirname "$0")"

echo "[1/6] DFT Hamiltonian dataset (sulfur-vacancy configurations)"
python3 scripts/gen_dataset.py 0 55 train

echo "[2/6] Controlled two-vacancy separation series"
python3 scripts/gen_separation.py

echo "[3/6] Coupled optical + electronic analysis"
python3 scripts/analyze_sep.py
python3 scripts/dft_analysis.py

echo "[4/6] Learned Hamiltonian: samples, training, held-out validation"
python3 scripts/build_samples.py train
python3 scripts/train_models.py
python3 scripts/ml_validation.py

echo "[5/6] Figures and tables"
python3 scripts/gen_numbers.py
python3 scripts/gen_tables.py
for f in fig0_abstract fig1_concept fig2_ml fig3_optics fig4_coupling figS_validation; do
    python3 scripts/$f.py
done

echo "[6/6] Package benchmark for archiving"
python3 scripts/pack_zenodo.py

echo "Done."
