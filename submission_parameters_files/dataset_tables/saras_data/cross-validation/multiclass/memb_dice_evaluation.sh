#!/usr/bin/env bash


fractions="0 1 2 3 4"
TOMOS="180426_004 180426_005 180426_021 181119_002 181119_030 181126_002 181126_012"
fractions_name="cv_fractions"
# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/cross-validation/multiclass/CV_data.csv"
DA=none
#DA="G1.5_E2_R180_DArounds4"

export class_number=2
export semantic_classes="ribo,fas,memb"
export statistics_file="/struct/mahamid/Irene/cross-validation/multiclass/dice_analysis_"$class_number".csv"

BN=false
depth=16
init_feat=8
output_classes=3
box_side=128
new_loader='True'
shuffle=true
retrain=false
encoder_dropout=0
decoder_dropout=0


test_partition_from_table=true

for fraction in $fractions
do
    for tomo_name in $TOMOS
  do
  echo $tomo_name
    #R_false_encoder_dropout_0.2_decoder_dropout_0.2_BN_false_DA_none_shuffle_true_frac_0_ribo_fas_memb__D_2_IF_8
    model_nickname="R_false_encoder_dropout_0.1_decoder_dropout_0.1_BN_false_DA_none_shuffle_true_frac_"$fraction"_ribo_fas_memb__D_2_IF_8"
    path_to_model="/struct/mahamid/Irene/cross-validation/multiclass/models/cross-validation/"$fractions_name"/"$model_nickname".pkl"
    label_name=$fractions_name"_"$model_nickname
    global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/"$fractions_name"/"$fraction"/"$model_nickname"/dice_analysis/"
    mkdir -p $global_output_dir
    export tomo_name=$tomo_name"_"$fraction"_"$fraction
    echo "Submitting job for tomo $tomo_name"
    sbatch $UPICKER_PATH/submission_scripts/dataset_tables/cnn_evaluation_runners/continuous_dice/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -BN $BN -semantic_classes $semantic_classes -test_partition_from_table $test_partition_from_table -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout
  done
done


#/struct/mahamid/Irene/cross-validation/multiclass/models/cross-validation/cv_fractions/DA_none_shuffle_true_frac_1_ribo_fas_memb__D_1_IF_8.pkl