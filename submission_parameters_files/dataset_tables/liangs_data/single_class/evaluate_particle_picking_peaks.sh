#!/usr/bin/env bash

#file-specifics:
# RIBO FAS HEALTHY VPP:
#TOMOS="172"
#TOMOS="173
#174
#175
#176
#177
#178
#179
#180
#190
#191
#192
#193
#194
#195
#196
#198
#199
#200
#201"
#TOMOS="
#203
#204
#205
#206
#207
#208
#240
#241
#242
#243
#244
#245
TOMOS="246"
#247
#248
#249
#250
#251"

# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/liang_data/liang_data.csv"
export class_number=0 # 0=ribo, 1=fas, 2=memb
export semantic_classes="ribo"

if [ $class_number == 0 ]; then
    echo "class_number is 0"
    export statistics_file="/struct/mahamid/Irene/liang_data/statistics.csv"
else
    echo "class_number non-supported for now"
fi


# CNN parameters:
path_to_model="/g/scb2/zaugg/trueba/3d-cnn/models/lang_unets/NO_DA_ribo_D_2_IF_8.pkl"
label_name="NO_DA_ribo_D_2_IF_8"

BN=True
depth=2
init_feat=8
output_classes=1
box_side=128
new_loader='True'

# Evaluation and particle picking parameters:
particle_picking_radius=30 #for particle picking
border_xy=10
lamella_extension=0
same_peak_radius_pr_analysis=20
score_threshold=-1
global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/"$label_name"_pr_radius_"$same_peak_radius_pr_analysis

mkdir -p $global_output_dir

for tomo_name in $TOMOS
do
	echo "Submitting job for tomo $tomo_name"
	sbatch  ./submission_scripts/dataset_tables/evaluations_from_dataset_table/parameters_file_read.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -minimum_peak_distance $particle_picking_radius -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_radius_pr_analysis -threshold $score_threshold -BN $BN -semantic_classes $semantic_classes
done

# ... Finally:
echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $global_output_dir"/SCRIPT.txt"
