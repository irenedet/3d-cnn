#!/usr/bin/env bash

#TOMOS="180426/021"

#TOMOS="180426/004"
TOMOS="180426/005
180426/021
180426/024"
#TOMOS="180426/005
#180426/021"
fractions_name="FAS_non_sph_masks"
# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_64.csv"
DA=none

export class_number=0 #
export particle_picking_radius=12 #for particle picking
export same_peak_radius_pr_analysis=12
export semantic_classes="fas"
export statistics_file="/struct/mahamid/Irene/cnn_evaluation/FAS/full_tomo_pr_cluster_analysis_64pix.csv"



BN=false
depth=3
init_feat=32
output_classes=1
box_side=64
new_loader='True'
shuffle=true
retrain=false
encoder_dropout=0
decoder_dropout=0.2

# Evaluation and particle picking parameters:
border_xy=10
lamella_extension=0
score_threshold=0
min_cluster_size=0
max_cluster_size=6000000
for tomo_name in $TOMOS
do
    # CNN parameters:
    model_nickname="64pix_lr_scheduler_10_R_false_encoder_dropout_0_decoder_dropout_0.2_BN_false_DA_none_shuffle_true_frac_None_fas__D_"$depth"_IF_"$init_feat
    path_to_model="/struct/mahamid/Irene/cross-validation/multiclass/models/"$fractions_name"/"$model_nickname".pkl"
    label_name=$fractions_name"_"$model_nickname
    global_output_dir="/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/"$fractions_name"/"$model_nickname"/full_tomo_pr_analysis/pr_radius_"$same_peak_radius_pr_analysis
    mkdir -p $global_output_dir
    echo "Submitting peak calling job for tomo $tomo_name"
    sbatch $UPICKER_PATH/submission_scripts/dataset_tables/cnn_evaluation_runners/cluster_centroids/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -border_xy $border_xy -lamella_extension $lamella_extension -threshold $score_threshold -BN $BN -semantic_classes $semantic_classes -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout -min_cluster_size $min_cluster_size -max_cluster_size $max_cluster_size -minimum_peak_distance $particle_picking_radius -same_peak_distance $same_peak_radius_pr_analysis
#    bash submission_scripts/dataset_tables/cnn_evaluation_runners/peak_calling/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -minimum_peak_distance $particle_picking_radius -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_radius_pr_analysis -threshold $score_threshold -BN $BN -semantic_classes $semantic_classes -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout
done
