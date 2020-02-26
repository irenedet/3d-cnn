#! /bin/bash


tomo_training_list="180426/026,180426/027,180426/028,180426/029,180426/030,180426/034,180426/037,180426/041,180426/043,180426/045"
fraction="None"

export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_64.csv"
#export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
export segmentation_names='fas'
export split=0.8

# Data for the new model
export fractions_name="FAS_non_sph_masks"
export log_dir="/struct/mahamid/Irene/yeast/FAS_non_sph_masks/log_"$fractions_name
export model_path="/struct/mahamid/Irene/cross-validation/multiclass/models/"$fractions_name
export n_epochs=150
export depth=3
export initial_features=32
export output_classes=1
export shuffle=true
export DA="none"
export BN=false
export encoder_dropout=0
export decoder_dropout=0.2
export batch_size=4

# Data for old models for resuming training:
export models_notebook="/struct/mahamid/Irene/cross-validation/multiclass/FAS_no_eman_64pix/models_cv_fas_fractions.csv"

export retrain="false"
export path_to_old_model="none"
export model_initial_name="64pix_lr_scheduler_10_R_"$retrain"_encoder_dropout_"$encoder_dropout"_decoder_dropout_"$decoder_dropout"_BN_"$BN"_DA_"$DA"_shuffle_"$shuffle"_frac_"$fraction"_"
sbatch $UPICKER_PATH/submission_scripts/dataset_tables/training/training_runner.sh -fraction $fraction -path_to_dataset_table $path_to_dataset_table -tomo_training_list $tomo_training_list -split $split -output_classes $output_classes -log_dir $log_dir -model_initial_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -models_notebook $models_notebook -BN $BN -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout -batch_size $batch_size
