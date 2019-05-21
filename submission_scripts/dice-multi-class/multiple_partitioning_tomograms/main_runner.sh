#!/usr/bin/env bash

FILES="/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180426_004.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180426_005.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_003.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_004.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_005.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_018.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180713_027.sh"
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180426_021.sh
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180426_024.sh"




export global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA"
export box_side=128
export overlap=12

mkdir $global_output_dir

for param_file in $FILES
do
	echo "Reading file $param_file"
	sbatch /g/scb2/zaugg/trueba/3d-cnn/submission_scripts/dice-multi-class/multiple_partitioning_tomograms/partition_tomo_intersecting_lamella.sh -parameters_file $param_file -output_dir $global_output_dir -box_side $box_side -overlap $overlap
done

# ... Finally:
echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $global_output_dir"/SCRIPT.txt"

