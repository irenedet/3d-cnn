#!/usr/bin/env bash

TOMOS="190301/005"

# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
export class_number=1 # 0=ribo, 1=fas, 2=memb
export semantic_classes="ribo,fas,memb"

if [ $class_number == 0 ]; then
    echo "class_number is 0"
    export statistics_file="/struct/mahamid/Irene/yeast/ribos_yeast_statistics_corrected_motls.csv"
elif [ $class_number == 1 ]; then
    echo "class_number is 1"
    export statistics_file="/struct/mahamid/Irene/yeast/fas_yeast_statistics.csv"
else
    echo "class_number non-supported for now"
fi

# CNN parameters:
path_to_model="models/fractions_004_005/Basis_004_005_shuffle_false_frac_0_ribo_fas_memb__D_1_IF_8.pkl"
model_nickname="Basis_004_005_shuffle_false_frac_0_ribo_fas_memb__D_1_IF_8"

BN=false
depth=1
init_feat=8
output_classes=3
box_side=128
new_loader='True'

# Evaluation and particle picking parameters:
particle_picking_radius=12 #for particle picking
border_xy=10
lamella_extension=0
same_peak_radius_pr_analysis=8
score_threshold=-1
global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/yeast_dataset/"$model_nickname"/peak_calling/pr_radius_"$same_peak_radius_pr_analysis

mkdir -p $global_output_dir

for tomo_name in $TOMOS
do
	echo "Submitting job for tomo $tomo_name"
	sbatch ./submission_scripts/dataset_tables/cnn_evaluation_runners/peak_calling/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $model_nickname -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -minimum_peak_distance $particle_picking_radius -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_radius_pr_analysis -threshold $score_threshold -BN $BN -semantic_classes $semantic_classes
done



