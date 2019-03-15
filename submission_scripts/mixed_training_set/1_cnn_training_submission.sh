#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 8G
#SBATCH --time 0-10:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu
#SBATCH -C gpu=1080Ti
#SBATCH --gres=gpu:1

module load GCC
module load Anaconda3

source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/generate_mixed_training_set/1_cnn_training.py
source deactivate
