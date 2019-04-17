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

## Tomogram to segment, z dimension, and shift of the current tomo w.r.t original:
#export path_to_raw='/scratch/trueba/3d-cnn/clean/180426_005/subtomo370-620.hdf'
#export input_xdim=927
#export input_ydim=927
#export input_zdim=251
#
## to be set as zero (unless the tomogram is shifted w.r. to original):
#export z_shift=370

# Tomogram to segment, z dimension, and shift of the current tomo w.r.t original:
#export path_to_raw='/scratch/trueba/3d-cnn/clean/180426_006/subtomo330-650.hdf'
#export input_xdim=927
#export input_ydim=927
#export input_zdim=321
#
## to be set as zero (unless the tomogram is shifted w.r. to original):
#export z_shift=330

# Tomogram to segment, z dimension, and shift of the current tomo w.r.t original:
export path_to_raw='/scratch/trueba/shrec/0/reconstruction_model_0.hdf'
export input_xdim=512
export input_ydim=512
export input_zdim=512

# to be set as zero (unless the tomogram is shifted w.r. to original):
export z_shift=0

# Trained UNet for the segmentation, category to be segmented, and UNet architecture features:
export path_to_model="/g/scb2/zaugg/trueba/3d-cnn/shrec_models/multi-class/Unet_all_diff_sph_adjusted_radius_all_particles_D_3_IF_8.pkl"
export label_name="all_particles"
export depth=3
export init_feat=8
export out_classes=13
export box_side=64

# Output directory, where results will be stored:
export output_dir='/scratch/trueba/shrec/0_sph_masks/cnn_evaluations/all_diff_D3_IF8/'

# Parameters relevant for the peak computations:
export class_number=12 # from 0 to n = out_classes - 1
export minimum_peak_distance=12

module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

bash /g/scb2/zaugg/trueba/3d-cnn/pipelines/multi-class/particle_picking_pipeline/runner.sh -raw $path_to_raw -output $output_dir -model $path_to_model -label $label_name -init_feat $init_feat -depth $depth -out_classes $out_classes -box $box_side -xdim $input_xdim -ydim $input_ydim -zdim $input_zdim -class_number $class_number -min_peak_distance $minimum_peak_distance -z_shift $z_shift
