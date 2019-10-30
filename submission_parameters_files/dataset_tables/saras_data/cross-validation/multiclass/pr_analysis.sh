#!/usr/bin/env bash

fractions="0 1 2 3 4"
TOMOS="180426_004 180426_005 180426_021 181119_002 181119_030 181126_002 181126_012"
fractions_name="cv_fractions"
# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/cross-validation/multiclass/CV_data.csv"
export class_number=0 #
export semantic_classes="ribo,fas,memb"
export statistics_file="/struct/mahamid/Irene/cross-validation/multiclass/peak_statistics_class_"$class_number".csv"



BN=false
depth=2
init_feat=8
output_classes=3
box_side=128
new_loader='True'
shuffle=false

# Evaluation and particle picking parameters:
particle_picking_radius=10 #for particle picking
border_xy=10
lamella_extension=20
same_peak_radius_pr_analysis=8
score_threshold=-1

for fraction in $fractions
do
    for tomo_name in $TOMOS
    do
        # CNN parameters:
        model_nickname="shuffle_"$shuffle"_frac_"$fraction"_fas__D_"$depth"_IF_"$init_feat
        path_to_model="models/"$fractions_name"/"$model_nickname".pkl"
        label_name=$fractions_name"_"$model_nickname
        global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/"$fractions_name"/"$fraction"/"$tomo_name"/"$model_nickname"/pr_radius_"$same_peak_radius_pr_analysis"/peak_calling"
        mkdir -p $global_output_dir

        export tomo_name=$tomo_name"_"$fraction"_"$fraction
        echo "Submitting job for tomo $tomo_name"
        sbatch ./submission_scripts/dataset_tables/cnn_evaluation_runners/peak_calling/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -minimum_peak_distance $particle_picking_radius -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_radius_pr_analysis -threshold $score_threshold -BN $BN -semantic_classes $semantic_classes
    done
done


