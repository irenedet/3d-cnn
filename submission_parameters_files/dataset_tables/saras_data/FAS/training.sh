#! /bin/bash

#FRACTIONS="0"
FRACTIONS="0 1 2 3 4"

tomo_training_list0="180426_004_0_1,180426_004_0_2,180426_004_0_3,180426_004_0_4,180426_005_0_1,180426_005_0_2,180426_005_0_3,180426_005_0_4,180426_021_0_1,180426_021_0_2,180426_021_0_3,180426_021_0_4,181119_002_0_1,181119_002_0_2,181119_002_0_3,181119_002_0_4,181119_030_0_1,181119_030_0_2,181119_030_0_3,181119_030_0_4,181126_002_0_1,181126_002_0_2,181126_002_0_3,181126_002_0_4,181126_012_0_1,181126_012_0_2,181126_012_0_3,181126_012_0_4"
tomo_training_list1="180426_004_1_0,180426_004_1_2,180426_004_1_3,180426_004_1_4,180426_005_1_0,180426_005_1_2,180426_005_1_3,180426_005_1_4,180426_021_1_0,180426_021_1_2,180426_021_1_3,180426_021_1_4,181119_002_1_0,181119_002_1_2,181119_002_1_3,181119_002_1_4,181119_030_1_0,181119_030_1_2,181119_030_1_3,181119_030_1_4,181126_002_1_0,181126_002_1_2,181126_002_1_3,181126_002_1_4,181126_012_1_0,181126_012_1_2,181126_012_1_3,181126_012_1_4"
tomo_training_list2="180426_004_2_1,180426_004_2_0,180426_004_2_3,180426_004_2_4,180426_005_2_1,180426_005_2_0,180426_005_2_3,180426_005_2_4,180426_021_2_1,180426_021_2_0,180426_021_2_3,180426_021_2_4,181119_002_2_1,181119_002_2_0,181119_002_2_3,181119_002_2_4,181119_030_2_1,181119_030_2_0,181119_030_2_3,181119_030_2_4,181126_002_2_1,181126_002_2_0,181126_002_2_3,181126_002_2_4,181126_012_2_1,181126_012_2_0,181126_012_2_3,181126_012_2_4"
tomo_training_list3="180426_004_3_1,180426_004_3_2,180426_004_3_0,180426_004_3_4,180426_005_3_1,180426_005_3_2,180426_005_3_0,180426_005_3_4,180426_021_3_1,180426_021_3_2,180426_021_3_0,180426_021_3_4,181119_002_3_1,181119_002_3_0,181119_002_3_2,181119_002_3_4,181119_030_3_1,181119_030_3_0,181119_030_3_2,181119_030_3_4,181126_002_3_1,181126_002_3_0,181126_002_3_2,181126_002_3_4,181126_012_3_1,181126_012_3_0,181126_012_3_2,181126_012_3_4"
tomo_training_list4="180426_004_4_1,180426_004_4_2,180426_004_4_3,180426_004_4_0,180426_005_4_1,180426_005_4_2,180426_005_4_3,180426_005_4_0,180426_021_4_1,180426_021_4_2,180426_021_4_3,180426_021_4_0,181119_002_4_1,181119_002_4_0,181119_002_4_3,181119_002_4_2,181119_030_4_1,181119_030_4_0,181119_030_4_3,181119_030_4_2,181126_002_4_1,181126_002_4_0,181126_002_4_3,181126_002_4_2,181126_012_4_1,181126_012_4_0,181126_012_4_3,181126_012_4_2"


export path_to_dataset_table="/struct/mahamid/Irene/cross-validation/multiclass/FAS_no_eman_no_aggregations_256pix/CV_data_no_eman_no_aggregations.csv"
export segmentation_names='fas'
export split=0.8

# Data for the new model
export fractions_name="cv_FAS_no_eman_no_aggreg_256pix"
export log_dir="/struct/mahamid/Irene/cross-validation/multiclass/FAS_no_eman_256pix/log_"$fractions_name
export model_path="/struct/mahamid/Irene/cross-validation/multiclass/models/cross-validation/"$fractions_name
export n_epochs=300
export depth=2
export initial_features=16
export output_classes=1
export shuffle=true
export DA="none"
export BN=false
export encoder_dropout=0
export decoder_dropout=0
export batch_size=1

# Data for old models for resuming training:
export models_notebook="/struct/mahamid/Irene/cross-validation/multiclass/FAS_no_eman_256pix/models_cv_fas_fractions.csv"

for fraction in $FRACTIONS
do
    echo "Submitting job for fraction "$fraction
    if [ $fraction == 0 ]; then
        tomo_training_list=$tomo_training_list0
    elif [ $fraction == 1 ]; then
        tomo_training_list=$tomo_training_list1
    elif [ $fraction == 2 ]; then
        tomo_training_list=$tomo_training_list2
    elif [ $fraction == 3 ]; then
        tomo_training_list=$tomo_training_list3
    elif [ $fraction == 4 ]; then
        tomo_training_list=$tomo_training_list4
    else
        echo "Inexistent fraction!"
    fi

    echo 'Job for fraction' $fraction
    export retrain="false"
    export path_to_old_model="none"
    export model_initial_name="R_"$retrain"_encoder_dropout_"$encoder_dropout"_decoder_dropout_"$decoder_dropout"_BN_"$BN"_DA_"$DA"_shuffle_"$shuffle"_frac_"$fraction"_"
    bash $UPICKER_PATH/submission_scripts/dataset_tables/training/training_runner.sh -fraction $fraction -path_to_dataset_table $path_to_dataset_table -tomo_training_list $tomo_training_list -split $split -output_classes $output_classes -log_dir $log_dir -model_initial_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -models_notebook $models_notebook -BN $BN -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout -batch_size $batch_size
done
