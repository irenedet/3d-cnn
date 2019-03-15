#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --time 0-10:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

export path_to_raw="/scratch/trueba/cnn/004/4bin/cnn/rawtomogram/180426_004_4bin.hdf"
export path_to_labeled="/scratch/trueba/cnn/004/4bin/cnn/centralregion_004.hdf"

export output_dir="/scratch/trueba/3d-cnn/training_data/TEST/"
export label_name="ribosomes"
export box_size=128

export output_zdim=221
export output_ydim=928
export output_xdim=928
# For data augmentation:
export number_iter=5
export split=40  # Only augment training data


echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

bash /g/scb2/zaugg/trueba/3d-cnn/training_pipeline/runner.sh