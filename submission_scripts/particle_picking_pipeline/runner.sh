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




# Tomogram to segment, z dimension, and shift of the current tomo w.r.t original:
# (the tomogram dimensions are assumed to be (output_zdim, 927, 927)):
export path_to_raw='/scratch/trueba/3d-cnn/clean/180426_006/subtomo330-650.hdf'
export output_zdim=321
export z_shift=330

#export path_to_raw='/scratch/trueba/3d-cnn/clean/180426_005/subtomo370-620.hdf'
#export output_zdim=251
#export z_shift=370


# Trained UNet for the segmentation, category to be segmented, and UNet architecture features:
export path_to_model="/g/scb2/zaugg/trueba/3d-cnn/models/0_UNet_new_128_side_depth_5_ini_feat_4_.pkl"
export label_name="ribosomes"
export depth=5
export init_feat=4
export box_side=128

# Output directory, where results will be stored:
export output_dir='/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_4_5_bis_/'
# Output file name for the segmentation:
export output_h5_file_path=$output_dir'4bin_subtomograms_.h5'

# Parameters relevant for the peak computations:
export overlap=12
export min_peak_distance=12

#export daa="1 2 3"

module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

bash ./particle_picking_pipeline/runner.sh -raw $path_to_raw -output $output_dir -model $path_to_model -label $label_name -out_h5 $output_h5_file_path -init_feat $init_feat -depth $depth -box $box_side -overlap $overlap -zdim $output_zdim -min_peak_distance $min_peak_distance -z_shift $z_shift
