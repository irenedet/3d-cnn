#!/usr/bin/env bash

#file-specifics:
#FILES="/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180426_004.sh"
FILES="/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180426_005.sh
"
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180426_024.sh
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180426_021.sh
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_003.sh
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_004.sh
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_005.sh
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_018.sh"
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180713_027.sh"


export output_dir="/scratch/trueba/3d-cnn/training_data/dice-multi-class"
export partition_dir_name="G_sigma1_non_sph"
export label_name="ribo,fas,memb"
export split=0.8  # percentage of training + validation set
export box_side=128

# For data augmentation:
export number_iter=6
export overlap=12

#mkdir $output_dir
#currently working only for 3 labels (mask0, mask1, mask2)
for parameters_file in $FILES
do
	echo "Reading file $parameters_file"
	source $parameters_file
	export eman2_filtered_file=$eman2_filtered_file
	export mask_0=$class_0_mask
	export mask_1=$class_1_mask
	export mask_2=$class_2_mask
	sbatch /g/scb2/zaugg/trueba/3d-cnn/submission_scripts/mixed_training_set/multiple_partitions_multi_label/parameters_file_read.sh -parameters_file $parameters_file -output_dir $output_dir -raw $eman2_filtered_file -mask_0 $mask_0 -mask_1 $mask_1 -mask_2 $mask_2 -partition_dir_name $partition_dir_name -label_name $label_name -split $split -box_side $box_side -number_iter $number_iter -overlap $overlap
done

# ... Finally:
#echo "Save a copy of this script for future reference"
#SCRIPT=`realpath $0`
#cp $SCRIPT $output_dir"/SCRIPT.txt"
