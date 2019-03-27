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
#export path_to_raw='/scratch/trueba/3d-cnn/clean/180426_004/subtomo380-600.hdf'
#export input_xdim=927
#export input_ydim=927
#export input_zdim=221
#
## to be set as zero (unless the tomogram is shifted w.r. to original):
#export z_shift=380

# Tomogram to segment, z dimension, and shift of the current tomo w.r.t original:
export path_to_raw='/scratch/trueba/3d-cnn/clean/180426_005/subtomo370-620.hdf'
export input_xdim=927
export input_ydim=927
export input_zdim=251

# to be set as zero (unless the tomogram is shifted w.r. to original):
export z_shift=370

# Tomogram to segment, z dimension, and shift of the current tomo w.r.t original:
#export path_to_raw='/scratch/trueba/3d-cnn/clean/180426_006/subtomo330-650.hdf'
#export input_xdim=927
#export input_ydim=927
#export input_zdim=321
#
## to be set as zero (unless the tomogram is shifted w.r. to original):
#export z_shift=330

# Trained UNet for the segmentation, category to be segmented, and UNet architecture features:
export path_to_model="/g/scb2/zaugg/trueba/3d-cnn/mixed_models/1_UNET_shuffle_mixed_G_sigma1-5_D_5_IF_4_.pkl"
export label_name="ribosomes"
export depth=5
export init_feat=4
export box_side=128

# Output directory, where results will be stored:
export output_dir='/scratch/trueba/3d-cnn/training_data/TEST/mixed_training/005/train_and_test_partitions/shuffle/G1-5_confs_4_5_'

# Parameters relevant for the peak computations:
export minimum_peak_distance=12

module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

bash ./particle_picking_pipeline/runner.sh -raw $path_to_raw -output $output_dir -model $path_to_model -label $label_name -init_feat $init_feat -depth $depth -box $box_side -xdim $input_xdim -ydim $input_ydim -zdim $input_zdim -min_peak_distance $minimum_peak_distance -z_shift $z_shift
