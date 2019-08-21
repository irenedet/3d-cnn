#!/usr/bin/env bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 128G
#SBATCH --time 0-02:00
#SBATCH -o slurm_outputs/run_mini_test_slurm.%N.%j.out
#SBAtCH -e slurm_outputs/run_mini_test_slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


module load Anaconda3
echo 'starting virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/

echo "running mini test"
python3 /g/scb2/zaugg/trueba/3d-cnn/runners/testing_runners/mini_test.py
echo "... done."