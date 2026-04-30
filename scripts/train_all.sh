#!/usr/bin/env bash
# Submit all six Noise2Map experiments to SLURM.
# Run from the repo root: bash scripts/train_all.sh

set -e
mkdir -p slurm_outputs

for cfg in whu_cd whu_sem xview2_cd xview2_sem spacenet7_cd spacenet7_sem; do
    sbatch scripts/train_${cfg}.sbatch 2>/dev/null || \
    echo "No sbatch for ${cfg} — running directly:"
    python train.py --config configs/${cfg}.yaml &
done

wait
echo "All jobs submitted."
