#!/usr/bin/env bash

fractions="4"
#TOMOS="180426/024 180426/025"
TOMOS="180426/004"
fractions_name="cv_fractions"
# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
#DA=none
DA="G1.5_E2_R180_DArounds4"
export class_numbers="2"
export semantic_classes="ribo,fas,memb"
export statistics_file="/struct/mahamid/Irene/cross-validation/multiclass/full_tomo_dice_analysis_"$class_number".csv"

BN=false
depth=2
init_feat=8
output_classes=3
box_side=128
new_loader='True'
shuffle=true
retrain=false
encoder_dropout=0
decoder_dropout=0
test_partition_from_table=false

for class_number in $class_numbers
  do
  for fraction in $fractions
  do
    for tomo_name in $TOMOS
    do
      echo $tomo_name
        model_nickname="DA_G1.5_E2_R180_DArounds4_shuffle_true_frac_"$fraction"_ribo_fas_memb__D_2_IF_8"
        path_to_model="/struct/mahamid/Irene/cross-validation/multiclass/models/cross-validation/cv_fractions/"$model_nickname".pkl"
        label_name=$fractions_name"_"$model_nickname
        global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/"$fractions_name"/"$model_nickname"/full_tomo_dice_analysis/"
        mkdir -p $global_output_dir
        echo "Submitting job for tomo $tomo_name"
        sbatch $UPICKER_PATH/submission_scripts/dataset_tables/cnn_evaluation_runners/continuous_dice/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -BN $BN -semantic_classes $semantic_classes -test_partition_from_table $test_partition_from_table -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout
    done
  done
done