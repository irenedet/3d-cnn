#!/usr/bin/env bash

#file-specifics:
FILES="/g/scb2/zaugg/trueba/3d-cnn/submission_data/9_VPP_TOMOS/180426_004.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/9_VPP_TOMOS/180426_005.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/9_VPP_TOMOS/180426_021.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/9_VPP_TOMOS/180426_024.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/9_VPP_TOMOS/180711_003.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/9_VPP_TOMOS/180711_004.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/9_VPP_TOMOS/180711_005.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/9_VPP_TOMOS/180711_018.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/9_VPP_TOMOS/180713_027.sh"

#Non file-specific parameters
path_to_model="/g/scb2/zaugg/trueba/3d-cnn/mixed_models/1_UNET_8TOMOS_DATA_AUG_D_2_IF_8.pkl"
label_name="1_shuff_G1_D2_IF8"
depth=2
init_feat=8
box_side=128
new_loader='True'
minimum_peak_distance=16
border_xy=20
lamella_extension=40
same_peak_distance=10
output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/uni-class-9TOMOS/1_shuff_G1_D2_IF8"

mkdir $output_dir

for param_file in $FILES
do
	echo "Reading file $param_file"
	sbatch /g/scb2/zaugg/trueba/3d-cnn/submission_scripts/Uni-label/evaluation/multiple_cnn_evaluations/parameters_file_read.sh -output_dir $output_dir -parameters_file $param_file -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -box_side $box_side -new_loader $new_loader -minimum_peak_distance $minimum_peak_distance -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_distance
done

# ... Finally:
echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $output_dir"/SCRIPT.txt"