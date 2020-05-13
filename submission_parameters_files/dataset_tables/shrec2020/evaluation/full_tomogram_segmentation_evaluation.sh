#!/usr/bin/env bash


TOMOS=""
fractions_name=""
# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_64.csv"
DA=none
#DA="G1.5_E2_R180_DArounds4"
export class_numbers="0"
export semantic_classes="fas"
export statistics_file="/struct/mahamid/Irene/cnn_evaluation/FAS/full_tomo_dice_analysis_64pix.csv"

BN=false
depth=2
init_feat=16
output_classes=1
box_side=64
new_loader='True'
shuffle=true
retrain=false
encoder_dropout=0
decoder_dropout=0
test_partition_from_table=false

for class_number in $class_numbers
  do
    for tomo_name in $TOMOS
    do
      model_nickname="64pix_lr_scheduler_10_R_false_encoder_dropout_0_decoder_dropout_0_BN_false_DA_none_shuffle_true_frac_None_fas__D_2_IF_16"
      path_to_model="/struct/mahamid/Irene/cross-validation/multiclass/models/"$fractions_name"/"$model_nickname".pkl"
      label_name=$fractions_name"_"$model_nickname
      global_output_dir="/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/"$fractions_name"/"$model_nickname"/full_tomo_dice_analysis/"
      mkdir -p $global_output_dir
      echo "Submitting job for tomo $tomo_name"
      bash $UPICKER_PATH/submission_scripts/dataset_tables/cnn_evaluation_runners/continuous_dice/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -BN $BN -semantic_classes $semantic_classes -test_partition_from_table $test_partition_from_table -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout
    done
done