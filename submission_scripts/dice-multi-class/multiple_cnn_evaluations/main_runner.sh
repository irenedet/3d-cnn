#!/usr/bin/env bash

#file-specifics:
FILES="/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180426_004.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180426_005.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180426_021.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_003.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_004.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_005.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_018.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180713_027.sh"
#FILES="/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180426_024.sh"


path_to_model="/g/scb2/zaugg/trueba/3d-cnn/models/dice_multi_label/retrained/RRr_180711_all_except_180711_003_ribo_fas_memb_D_2_IF_8.pkl"
label_name="RRr_180711_all_except_180711_003_"
depth=2
init_feat=8
output_classes=3
class_number=1 # 0=ribo, 1=fas, 2=memb
box_side=128
new_loader='True'
minimum_peak_distance=16
border_xy=20
lamella_extension=40
same_peak_distance=16
output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/"$label_name

mkdir $output_dir

for param_file in $FILES
do
	echo "Reading file $param_file"
	sbatch /g/scb2/zaugg/trueba/3d-cnn/submission_scripts/dice-multi-class/multiple_cnn_evaluations/parameters_file_read.sh -output_dir $output_dir -parameters_file $param_file -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -minimum_peak_distance $minimum_peak_distance -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_distance
done

# ... Finally:
#echo "Save a copy of this script for future reference"
#SCRIPT=`realpath $0`
#cp $SCRIPT $output_dir"/SCRIPT.txt"
