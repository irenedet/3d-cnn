#!/usr/bin/env bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 32G
#SBATCH --time 0-05:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

export path_to_raw="/struct/mahamid/Irene/yeast/ribosomes/180426_005/005_bin4.hdf"
export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180426_005"

#export segmentation_names="ribo,fas,memb"
export box_side=128
export overlap=12

module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

export PYTHONPATH=/g/scb2/zaugg/trueba/3d-cnn

export outh5=$output_dir"/tomo_partition.h5"
echo 'starting python script'
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/dice_multi-class/particle_picking_pipeline/1_partition_tomogram.py -raw $path_to_raw -outh5 $outh5 -output $output_dir -box $box_side -overlap $overlap
echo 'done'