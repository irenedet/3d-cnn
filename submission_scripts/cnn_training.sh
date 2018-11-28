#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 5-00:00
#SBATCH -o ../jobs_outputs/slurm.%N.%j.out
#SBAtCH -e ../jobs_outputs/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu
#SBATCH -C gpu=1080Ti
#SBATCH --gres=gpu:1

module load GCC

.//g/scb2/zaugg/trueba/3d-cnn/submission_scripts/test_script.py
