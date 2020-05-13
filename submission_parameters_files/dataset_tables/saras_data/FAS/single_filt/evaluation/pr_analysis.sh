#!/usr/bin/env bash

export group="mahamid"
export nodes="1"
export ntasks="1"
export mem="70G"
export time="0-02:00"
export log_file="segment_partition.slurm.%N.%j.out"
export mail_type="END,FAIL"
export mail_address="irene.de.teresa@embl.de"
export partition="gpu"
export card="None"
#export gres="None"
export gres="gpu:4,-n1,-c4"


tomo_training_list1="180426/026 180426/027 180426/028 180426/030 180426/034 180426/037 180426/043 180426/045"
test_list1="180426/029" # 180426/041"
tomo_training_list2="180426/026,180426/027,180426/028,180426/029,180426/030,180426/034,180426/041,180426/045"
test_list2="180426/037 180426/043"
tomo_training_list3="180426/026,180426/027,180426/028,180426/029,180426/034,180426/037,180426/041,180426/043"
test_list3="180426/030 180426/045"

fraction="None"

#sets="1 2 3"
sets="2 3"
fractions_name="FAS_non_sph_masks"

# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_4bin_fas_single_filter.csv"

export class_number=0
export particle_picking_radius=12
export same_peak_radius_pr_analysis=12
export semantic_classes="fas"
export statistics_file="/struct/mahamid/Irene/cnn_evaluation/FAS/singlefilt_full_tomo_pr_cluster_analysis_64pix_grid_partition.csv"

DA="G5_E0_R0_DArounds4"
#DA="DA_2xf"
BN=false
depth=2
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

for set in $sets
do
      if   [ $set == 1 ]; then
          export TOMOS=$test_list1
      elif [ $set == 2 ]; then
          export TOMOS=$test_list2
      elif [ $set == 3 ]; then
          export TOMOS=$test_list3
      fi

    for tomo_name in $TOMOS
    do
      echo "Submitting job for tomo "$tomo_name" and set "$set

    # CNN parameters:
    #singlefilt_64pix_lr_scheduler_10_R_false_encoder_dropout_0_decoder_dropout_0.2_BN_false_DA_DA_2xf_shuffle_true_frac_3_fas__D_2_IF_32.pkl
    model_nickname="singlefilt_64pix_lr_scheduler_10_R_false_encoder_dropout_"$encoder_dropout"_decoder_dropout_"$decoder_dropout"_BN_"$BN"_DA_"$DA"_shuffle_"$shuffle"_frac_"$set"_fas__D_"$depth"_IF_"$init_feat
    path_to_model="/struct/mahamid/Irene/yeast/models/4bin/"$fractions_name"/"$model_nickname".pkl"
    label_name=$fractions_name"_"$model_nickname
    global_output_dir="/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/"$fractions_name"/"$model_nickname"/full_tomo_pr_analysis_full_partition/pr_radius_"$same_peak_radius_pr_analysis
    mkdir -p $global_output_dir


#    submission_template=$UPICKER_PATH/submission_scripts/dataset_tables/cnn_evaluation_runners/segmentation.sh
#    file_basename=$(basename -- $submission_template)
#    mkdir -p /tmp/$USER/.3dcnn_cache
#    my_submission=/tmp/$USER/.3dcnn_cache/$file_basename
#    cp $submission_template $my_submission
#    python3 $UPICKER_PATH/src/python/submission_files/slurm.py -file $my_submission -group $group -nodes $nodes -ntasks $ntasks -mem $mem -job_time $time -log_file $log_file -mail_type $mail_type -mail_address $mail_address -partition $partition -card $card -gres $gres
#    sbatch  $my_submission  -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -BN $BN -semantic_classes $semantic_classes -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout
    ##

    submission_template=$UPICKER_PATH/submission_scripts/dataset_tables/cnn_evaluation_runners/cluster_centroids/runner.sh
    file_basename=$(basename -- $submission_template)
    mkdir -p /tmp/$USER/.3dcnn_cache
    my_submission=/tmp/$USER/.3dcnn_cache/$file_basename
    cp $submission_template $my_submission
    python3 $UPICKER_PATH/src/python/submission_files/slurm.py -file $my_submission -group $group -nodes $nodes -ntasks $ntasks -mem $mem -job_time $time -log_file $log_file -mail_type $mail_type -mail_address $mail_address -partition $partition -card $card -gres $gres
    bash  $my_submission -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -border_xy $border_xy -lamella_extension $lamella_extension -threshold $score_threshold -BN $BN -semantic_classes $semantic_classes -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout -min_cluster_size $min_cluster_size -max_cluster_size $max_cluster_size -minimum_peak_distance $particle_picking_radius -same_peak_distance $same_peak_radius_pr_analysis
  done
done

