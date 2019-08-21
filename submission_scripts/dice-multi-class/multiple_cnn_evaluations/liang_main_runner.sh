!/usr/bin/env bash

#file-specifics:
FILES="/g/scb2/zaugg/trueba/3d-cnn/submission_data/SPOMBE/ED_DEFOCUS/190301_005.sh"

# CNN parameters:
path_to_model="/g/scb2/zaugg/trueba/3d-cnn/models/lang_unets/NO_DA_ribo_D_2_IF_8.pkl"
depth=2
init_feat=8
output_classes=1
box_side=128
new_loader='True'

# Evaluation and particle picking parameters:
label_name="NO_DA_ribo_D_2_IF_8"
class_number=0 # 0=ribo, 1=fas, 2=memb
minimum_peak_distance=30
border_xy=20
lamella_extension=0
same_peak_distance=60
score_threshold=-1
global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/"$label_name


mkdir $output_dir

for param_file in $FILES
do
	echo "Reading file $param_file"
	sbatch ./submission_scripts/dice-multi-class/multiple_cnn_evaluations/parameters_file_read.sh -output_dir $global_output_dir -parameters_file $param_file -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -minimum_peak_distance $minimum_peak_distance -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_distance -threshold $score_threshold
done

# ... Finally:
#echo "Save a copy of this script for future reference"
#SCRIPT=`realpath $0`
#cp $SCRIPT $output_dir"/SCRIPT.txt"

