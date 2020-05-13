#! /bin/bash

export group="mahamid"
export nodes="1"
export ntasks="1"
export mem="100G"
export time="0-25:05"
export log_file="training.slurm.%N.%j.out"
export error_file="training.slurm.%N.%j.err"
export mail_type="END,FAIL"
export mail_address="irene.de.teresa@embl.de"
export partition="gpu"
export card="None"
#export gres="None"
export gres="gpu:4,-n1,-c4"

sets="1"
#sets="1 2 3"

tomo_training_list1="0,2,3,4,6,7,8"
test_list1="1,5"
tomo_training_list2="0,1,2,4,5,6,8"
test_list2="3,7"
tomo_training_list3="0,1,3,4,5,6,7"
test_list3="2,8"
fraction="None"

export path_to_dataset_table="/struct/mahamid/Irene/shrec2020/data/table.csv"
export segmentation_names='all'
export split=0.8

# Data for the new model
export fractions_name="particle_all"
export log_dir="/struct/mahamid/Irene/shrec2020/dataset_tables/logging/log_"$fractions_name
export model_path="/struct/mahamid/Irene/shrec2020/dataset_tables/models/"$fractions_name
export n_epochs=50
export depth=2
export initial_features=16
export output_classes=1
export shuffle=true
export DA="none"
export BN=False
export encoder_dropout=0
export decoder_dropout=0.2
export batch_size=4
export training_partition='train_partition'

# Data for old models for resuming training:
export models_notebook="/struct/mahamid/Irene/shrec2020/dataset_tables/models/"$fractions_name"_models.csv"


for set in $sets
do
  if   [ $set == 1 ]; then
    export tomo_training_list=$tomo_training_list1
  elif [ $set == 2 ]; then
    export tomo_training_list=$tomo_training_list2
  elif [ $set == 3 ]; then
    export tomo_training_list=$tomo_training_list3
  fi
  echo "Submitting job for set" $set
  export retrain="false"
  export path_to_old_model="none"
  export model_initial_name="64pix_encoder_dropout_"$encoder_dropout"_decoder_dropout_"$decoder_dropout"_BN_"$BN"_DA_"$DA"_shuffle_"$shuffle"_frac_"$set"_"
  mkdir -p $model_path
  mkdir -p log_dir
  submission_template=$UPICKER_PATH/submission_scripts/dataset_tables/training/training_runner.sh
  file_basename=$(basename -- $submission_template)
  mkdir -p /tmp/$USER/.3dcnn_cache
  my_submission=/tmp/$USER/.3dcnn_cache/$file_basename
  cp $submission_template $my_submission

  echo $(ls $my_submission)
  python3 $UPICKER_PATH/src/python/submission_files/slurm.py -file $my_submission -group $group -nodes $nodes -ntasks $ntasks -mem $mem -job_time $time -log_file $log_file -mail_type $mail_type -mail_address $mail_address -partition $partition -card $card -gres $gres -error_file $error_file
  sbatch $my_submission -fraction $fraction -path_to_dataset_table $path_to_dataset_table -tomo_training_list $tomo_training_list -split $split -output_classes $output_classes -log_dir $log_dir -model_initial_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -models_notebook $models_notebook -BN $BN -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout -batch_size $batch_size
done
