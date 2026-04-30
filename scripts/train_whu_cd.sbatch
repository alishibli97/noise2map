#!/bin/bash
#SBATCH --job-name=n2m_whu_cd
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=64G

module load Python/3.11
source activate noise2map   # adjust to your conda/venv name

cd $SLURM_SUBMIT_DIR

python train.py --config configs/whu_cd.yaml \
    logging.write_to_wandb=true \
    training.batch_size=8
