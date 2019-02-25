#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 0-2:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

# Tomogram to segment, z dimension, and shift of the current tomo w.r.t original:
export input_xdim=927
export input_ydim=927
export input_zdim=321

# to be set as zero (unless the tomogram is shifted w.r. to original):
export z_shift=330

# Trained UNet for the segmentation, category to be segmented, and UNet architecture features:
#export path_to_model="/g/scb2/zaugg/trueba/3d-cnn/models/0_UNet_new_128_side_depth_5_ini_feat_16_.pkl"
export label_name="ribosomes"
export box_side=128

# Output directory, where results will be stored:
export output_dir='/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_16_5_bis_/'

# Parameters relevant for the peak computations:
export minimum_peak_distance=12
export box_overlap=12
export output_h5_file_path=$output_dir'4bin_subtomograms_.h5'

module load Anaconda3

source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
python3 ./particle_picking_pipeline/3_get_peaks_motive_list.py -output $output_dir -label $label_name -subtomo $output_h5_file_path -box $box_side -xdim $input_xdim -ydim $input_ydim -zdim $input_zdim -min_peak_distance $minimum_peak_distance -z_shift $z_shift -overlap $box_overlap
#source deactivate
