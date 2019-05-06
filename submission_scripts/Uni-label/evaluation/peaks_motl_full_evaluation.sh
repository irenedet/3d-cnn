#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 32G
#SBATCH --time 0-01:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
echo "... done"

export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn
export QT_QPA_PLATFORM='offscreen'


export test_partition="/struct/mahamid/Irene/yeast/ribosomes/180713_027/G_sigma1/train_and_test_partitions/test_partition.h5"

# dimensions of lamella file
export input_xdim=960
export input_ydim=928
export input_zdim=500
export z_shift=0
# Output directory, where results will be stored:
export output_dir='/scratch/trueba/3d-cnn/cnn_evaluation/uni-class-9TOMOS/confs_D5_IF4_G1/180713_027'

# For lamella masking
export hdf_lamella_file="/scratch/trueba/3d-cnn/clean/180713_027/027_lamellamask.hdf"
export border_xy=20
export lamella_extension=40

# Parameters relevant for the peak calling algorithm:
export minimum_peak_distance=16

# For PR analysis
#export path_to_motl_clean="/scratch/trueba/3d-cnn/clean/180426_005/motl_clean_4b.em"
export path_to_motl_clean="/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/TM/motl_clean_4b.em"
export same_peak_distance=10
export x_shift=0 # 16 for 005

# Trained UNet for the segmentation, category to be segmented, and UNet architecture features:
export path_to_model="/g/scb2/zaugg/trueba/3d-cnn/mixed_models/0_UNET_9TOMOS_DATA_AUGD_5_IF_4.pkl"
export label_name="shuff_G1_D5_IF4"
export depth=5
export init_feat=4
export box_side=128
export new_loader='True' #True if the cnn loader has new format (after 04/2019)


# 1. Segmenting, peak calling and motl writing
echo "Calling particle picking pipeline"
bash /g/scb2/zaugg/trueba/3d-cnn/particle_picking_pipeline/runner_in_partirtion_set.sh -test_partition $test_partition -output $output_dir -model $path_to_model -label $label_name -init_feat $init_feat -depth $depth -box $box_side -xdim $input_xdim -ydim $input_ydim -zdim $input_zdim -min_peak_distance $minimum_peak_distance -z_shift $z_shift
echo "... done with particle picking pipeline."


# 2.0 Create hdf file of lamella mask (optional):
#export em_lamella_file="/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/TM/027_lamellamask.em"
#echo "converting em lamella file into hdf..."
#module load EMAN2
#lamella_dir="$(dirname $hdf_lamella_file)"
#mkdir $lamella_dir
#e2proc3d.py $em_lamella_file $hdf_lamella_file
#echo "...done converting em to hdf."


# 2. Mask coordinate points with lamella mask
export path_to_csv_motl=$(ls $output_dir/motl*)

echo "Now filtering points in lamella mask"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/performance_nnet_quantification/filter_with_lamella_mask.py -csv_motl $path_to_csv_motl -lamella_file $hdf_lamella_file -output_dir $output_dir -border_xy $border_xy -lamella_extension $lamella_extension -x_dim $input_xdim -y_dim $input_ydim -z_dim $input_zdim -z_shift $z_shift
echo "...done filtering points in lamella mask."


# 3. Precision-Recall analysis
export lamella_output_dir=$output_dir"/in_lamella"
export path_to_csv_motl=$(ls $lamella_output_dir/motl*)


echo "Starting to generate precision recall plots"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/performance_nnet_quantification/precision_recall_plots.py -motl $path_to_csv_motl -clean $path_to_motl_clean -output $lamella_output_dir -test_file $test_partition -radius $same_peak_distance -shape_x $input_xdim -shape_y $input_ydim -shape_z $input_zdim -x_shift $x_shift -z_shift $z_shift -box $box_side
echo "...done with precision recall plots."

# ... Finally:
echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $output_dir"/SCRIPT.txt"

