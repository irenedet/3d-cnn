#!/usr/bin/env bash

#TOMOS="190301/005"
TOMOS="181119/002 181119/030 181126/002 181126/012 181126/025"

# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
export class_number=0
export semantic_classes="fas"

statistics_file="/struct/mahamid/Irene/yeast/fas_yeast_statistics.csv"

# CNN parameters:
path_to_model="models/fractions_004_005_021_ED/shuffle_false_frac_0_fas__D_2_IF_8.pkl"
model_nickname="004_005_021_ED_shuffle_false_frac_0_fas__D_2_IF_8"

BN=false
frac=0
depth=2
init_feat=8
output_classes=1
box_side=128
new_loader='True'

# Evaluation and particle picking parameters:
particle_picking_radius=12 #for particle picking
border_xy=10
lamella_extension=0
same_peak_radius_pr_analysis=10
score_threshold=-1
global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/yeast_dataset/"$model_nickname"/peak_calling/pr_radius_"$same_peak_radius_pr_analysis

mkdir -p $global_output_dir

for tomo_name in $TOMOS
do
	echo "Submitting job for tomo $tomo_name"
	sbatch ./submission_scripts/dataset_tables/cnn_evaluation_runners/peak_calling/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $model_nickname -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -minimum_peak_distance $particle_picking_radius -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_radius_pr_analysis -threshold $score_threshold -BN $BN -semantic_classes $semantic_classes
done



