#!/usr/bin/env bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 40G
#SBATCH --time 0-2:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBAtCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu
#SBATCH --gres=gpu:1

echo "Activating virtual environment"
##module load Anaconda3
#UPICKER_VENV_PATH=/struct/mahamid/Irene/envs/.conda/3d-cnn
conda activate $UPICKER_VENV_PATH

python3 /struct/mahamid/Irene/3d-cnn/runners/quantifications/runner.py