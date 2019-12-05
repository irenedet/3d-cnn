#!/usr/bin/env bash

fractions="0 1 2 3 4"
TOMOS="180426_024 181126_025"
#TOMOS="181126/025"
fractions_name="cv_fractions"
# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/cross-validation/multiclass/CV_data.csv"
DA=none
#DA="G1.5_E2_R180_DArounds4"
export class_numbers="0"
#export class_numbers="1 2"
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

for fraction in $fractions
do
  for tomo_name in $TOMOS
  do
    for class_number in $class_numbers
      do
      echo $tomo_name
#        DA_none_shuffle_true_frac_3_ribo_fas_memb__D_2_IF_8
        model_nickname="DA_"$DA"_shuffle_"$shuffle"_frac_"$fraction"_ribo_fas_memb__D_"$depth"_IF_"$init_feat
#        model_nickname="R_"$retrain"_encoder_dropout_"$encoder_dropout"_decoder_dropout_"$decoder_dropout"_BN_"$BN"_DA_"$DA"_shuffle_"$shuffle"_frac_"$fraction"_ribo_fas_memb__D_"$depth"_IF_"$init_feat
        path_to_model="/struct/mahamid/Irene/cross-validation/multiclass/models/cross-validation/"$fractions_name"/"$model_nickname".pkl"
        label_name=$fractions_name"_"$model_nickname
        global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/"$fractions_name"/"$model_nickname"/full_tomo_dice_analysis/"
        mkdir -p $global_output_dir
        export tomo_name=$tomo_name"_"$fraction
        echo "Submitting job for tomo" $tomo_name "for class" $class_number
        sbatch $UPICKER_PATH/submission_scripts/dataset_tables/cnn_evaluation_runners/continuous_dice/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -BN $BN -semantic_classes $semantic_classes -test_partition_from_table $test_partition_from_table -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout
    done
  done
done