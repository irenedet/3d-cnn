#!/usr/bin/env bash

#fractions="0"
fractions="0 1 2 3 4"
TOMOS="180426_004 180426_005 180426_021 181119_002 181119_030 181126_002 181126_012"
fractions_name="cv_fas_fractions"
# Tomograms data
#export path_to_dataset_table="/struct/mahamid/Irene/cross-validation/multiclass/DA_G1.5_E2_R180_DArounds4/DA_CV_data.csv"
#DA="G1.5_E2_R180_DArounds4"

export path_to_dataset_table="/struct/mahamid/Irene/cross-validation/multiclass/DA_FAS_WT_test/DA_FAS_WT_data.csv"
DA=none

export class_number=0 #
export particle_picking_radius=12 #for particle picking
export same_peak_radius_pr_analysis=10
export semantic_classes="fas"
export statistics_file="/struct/mahamid/Irene/cross-validation/multiclass/DA_FAS_WT_test/peak_statistics.csv"



BN=false
depth=1
init_feat=16
output_classes=1
box_side=128
new_loader='True'
shuffle=true
retrain=false
encoder_dropout=0
decoder_dropout=0

# Evaluation and particle picking parameters:
border_xy=10
lamella_extension=0
score_threshold=-1

for fraction in $fractions
do
    for tomo_name in $TOMOS
    do
        # CNN parameters:
        model_nickname="Verion2_R_false_encoder_dropout_0_decoder_dropout_0_BN_false_DA_none_shuffle_true_frac_"$fraction"_fas__D_1_IF_16"
        #R_false_encoder_dropout_0_decoder_dropout_0_BN_true_DA_none_shuffle_true_frac_2_ribo_fas_memb__D_2_IF_8
#        model_nickname="R_"$retrain"_encoder_dropout_"$encoder_dropout"_decoder_dropout_"$decoder_dropout"_BN_"$BN"_DA_"$DA"_shuffle_"$shuffle"_frac_"$fraction"_ribo_fas_memb__D_"$depth"_IF_"$init_feat
        path_to_model="/struct/mahamid/Irene/cross-validation/multiclass/models/cross-validation/"$fractions_name"/"$model_nickname".pkl"
        label_name=$fractions_name"_"$model_nickname
        global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/"$fractions_name"/"$fraction"/"$model_nickname"/pr_radius_"$same_peak_radius_pr_analysis"/peak_calling"
        mkdir -p $global_output_dir

        export tomo_name=$tomo_name"_"$fraction"_"$fraction
        echo "Submitting job for tomo $tomo_name"
        sbatch submission_scripts/dataset_tables/cnn_evaluation_runners/peak_calling/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -minimum_peak_distance $particle_picking_radius -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_radius_pr_analysis -threshold $score_threshold -BN $BN -semantic_classes $semantic_classes -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout
    done
done
