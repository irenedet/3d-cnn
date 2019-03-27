#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --time 0-2:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de




## Tomogram to segment, z dimension, and shift of the current tomo w.r.t original:
#export training_file="/scratch/trueba/3d-cnn/training_data/TEST/mixed_training/004/train_and_test_partitions/partition_training.h5"
#export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/gaussian_aug/confs_4_5_/peaks_in_training_partition"
#export input_xdim=927
#export input_ydim=927
#export input_zdim=221
#
## to be set as zero (unless the tomogram is shifted w.r. to original):
#export z_shift=380

# Tomogram to segment, z dimension, and shift of the current tomo w.r.t original:
#export training_file="/scratch/trueba/3d-cnn/training_data/TEST/mixed_training/005/train_and_test_partitions/partition_training.h5"
#export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_4_5_bis_/peaks_in_training_partition"
#export input_xdim=927
#export input_ydim=927
#export input_zdim=251
#
# # to be set as zero (unless the tomogram is shifted w.r. to original):
#export z_shift=370

# Tomogram to segment, z dimension, and shift of the current tomo w.r.t original:
export training_file="/scratch/trueba/3d-cnn/training_data/TEST/mixed_training/006/train_and_test_partitions/partition_training.h5"
export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/mixed_trainset/no_shuffle/G1_confs_4_5_/peaks_in_training_partition"
export input_xdim=927
export input_ydim=927
export input_zdim=321

# to be set as zero (unless the tomogram is shifted w.r. to original):
export z_shift=330

# Trained UNet for the segmentation, category to be segmented, and UNet architecture features:
export path_to_model="/g/scb2/zaugg/trueba/3d-cnn/mixed_models/0_UNET_mixed_G_sigma1_D_5_IF_4_.pkl"
export model_nickname="mix_G1_4_5"
export depth=5
export init_feat=4
export box_side=128

# Parameters relevant for the peak computations:
export minimum_peak_distance=12

module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

bash /g/scb2/zaugg/trueba/3d-cnn/pipelines/find_detected_and_undetected/particle_picking_from_trainingset.sh -output $output_dir -training_file $training_file -model $path_to_model -model_nickname $model_nickname -init_feat $init_feat -depth $depth -box $box_side -xdim $input_xdim -ydim $input_ydim -zdim $input_zdim -min_peak_distance $minimum_peak_distance -z_shift $z_shift

cp /g/scb2/zaugg/trueba/3d-cnn/submission_scripts/find_detected_and_undetected_pipeline/particle_picking_from_trainingset_runner.sh $output_dir"/submission_script.txt"