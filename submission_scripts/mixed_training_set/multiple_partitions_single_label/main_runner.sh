#!/usr/bin/env bash


FILES="/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/InVitro/01.sh"
#FILES="/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/InVitro/02.sh
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/InVitro/03.sh
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/InVitro/04.sh
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/InVitro/05.sh
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/InVitro/06.sh
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/InVitro/10.sh
#/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/InVitro/11.sh
#"

export output_dir="/struct/mahamid/Irene/NPC"
export partition_dir_name="G1_Rot_E1_1_bis"
export label_name="npc"
export split=0.8  # percentage of training + validation set
export box_side=128

# For data augmentation:
export number_iter=6
export overlap=12

for parameters_file in $FILES
do
	echo "Reading file $parameters_file"
	source $parameters_file
	export tomo_name=$tomo_name
	export eman2_filtered_file=$eman2_filtered_file
	export mask_0=$class_npc_mask
	sbatch /g/scb2/zaugg/trueba/3d-cnn/submission_scripts/mixed_training_set/multiple_partitions_single_label/parameters_file_read.sh -parameters_file $parameters_file -output_dir $output_dir -raw $eman2_filtered_file -mask_0 $mask_0 -partition_dir_name $partition_dir_name -label_name $label_name -split $split -box_side $box_side -number_iter $number_iter -overlap $overlap -tomo_name $tomo_name
done

# ... Finally:
#echo "Save a copy of this script for future reference"
#SCRIPT=`realpath $0`
#cp $SCRIPT $output_dir"/SCRIPT.txt"
