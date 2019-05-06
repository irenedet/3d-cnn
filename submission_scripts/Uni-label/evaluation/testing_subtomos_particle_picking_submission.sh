#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 32G
#SBATCH --time 0-10:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
echo "... done"

export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn

export test_partition="/struct/mahamid/Irene/yeast/ribosomes/180426_021/G_sigma1/train_and_test_partitions/test_partition.h5"
export input_xdim=959
export input_ydim=927
export input_zdim=1000

export z_shift=0

# Trained UNet for the segmentation, category to be segmented, and UNet architecture features:
export path_to_model="/g/scb2/zaugg/trueba/3d-cnn/mixed_models/0_UNET_9TOMOS_D_5_IF_4.pkl"
export label_name="ribosomes"
export depth=5
export init_feat=4
export box_side=128
export new_loader='True' #True if the cnn loader has new format (after 04/2019)
# Output directory, where results will be stored:
export output_dir='/scratch/trueba/3d-cnn/cnn_evaluation/uni-class-9TOMOS/confs_D5_IF4_021'

# Parameters relevant for the peak computations:
export minimum_peak_distance=16

bash /g/scb2/zaugg/trueba/3d-cnn/particle_picking_pipeline/runner_in_partirtion_set.sh -test_partition $test_partition -output $output_dir -model $path_to_model -label $label_name -init_feat $init_feat -depth $depth -box $box_side -xdim $input_xdim -ydim $input_ydim -zdim $input_zdim -min_peak_distance $minimum_peak_distance -z_shift $z_shift
