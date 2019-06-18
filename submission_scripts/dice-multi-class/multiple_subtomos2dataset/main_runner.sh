#!/usr/bin/env bash


FILES="/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/ED_TOMOS/181119_030.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/ED_TOMOS/181126_002.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/ED_TOMOS/181126_012.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/ED_TOMOS/181126_025.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/ED_TOMOS/190301_005.sh"


# Global parameters for the segmentation to be reconstructed
numb_classes=3
label_name="Retrain_retrained_D4_IF8_NA_except_180711_003"
segmentation_names="ribo,fas,memb"
output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/"$label_name

# Partitioning parameters:
box_side=128
overlap_thickness=12

for param_file in $FILES
do
	echo "Reading file $param_file"
	sbatch ./submission_scripts/dice-multi-class/multiple_subtomos2dataset/parameters_file_read.sh -parameters_file $param_file -output_dir $output_dir -box_side $box_side -overlap $overlap_thickness -output_classes $numb_classes -label_name $label_name -segmentation_names $segmentation_names
	echo "job submitted"
done

# ... Finally:
#echo "Save a copy of this script for future reference"
#SCRIPT=`realpath $0`
#cp $SCRIPT $output_dir"/SCRIPT.txt"
