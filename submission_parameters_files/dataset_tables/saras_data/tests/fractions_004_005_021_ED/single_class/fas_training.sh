#! /bin/bash

FRACTIONS="2 3 4"


tomo_training_list0="004_0_1,004_0_2,004_0_3,004_0_4,005_0_1,005_0_2,005_0_3,005_0_4,021_0_1,021_0_2,021_0_3,021_0_4,181119_002_0_1,181119_002_0_2,181119_002_0_3,181119_002_0_4,181119_030_0_1,181119_030_0_2,181119_030_0_3,181119_030_0_4,181126_002_0_1,181126_002_0_2,181126_002_0_3,181126_002_0_4,181126_012_0_1,181126_012_0_2,181126_012_0_3,181126_012_0_4,181126_025_0_1,181126_025_0_2,181126_025_0_3,181126_025_0_4"
tomo_training_list1="004_1_0,004_1_2,004_1_3,004_1_4,005_1_0,005_1_2,005_1_3,005_1_4,021_1_0,021_1_2,021_1_3,021_1_4,181119_002_1_0,181119_002_1_2,181119_002_1_3,181119_002_1_4,181119_030_1_0,181119_030_1_2,181119_030_1_3,181119_030_1_4,181126_002_1_0,181126_002_1_2,181126_002_1_3,181126_002_1_4,181126_012_1_0,181126_012_1_2,181126_012_1_3,181126_012_1_4,181126_025_1_0,181126_025_1_2,181126_025_1_3,181126_025_1_4"
tomo_training_list2="004_2_1,004_2_0,004_2_3,004_2_4,005_2_1,005_2_0,005_2_3,005_2_4,021_2_1,021_2_0,021_2_3,021_2_4,181119_002_2_1,181119_002_2_0,181119_002_2_3,181119_002_2_4,181119_030_2_1,181119_030_2_0,181119_030_2_3,181119_030_2_4,181126_002_2_1,181126_002_2_0,181126_002_2_3,181126_002_2_4,181126_012_2_1,181126_012_2_0,181126_012_2_3,181126_012_2_4,181126_025_2_1,181126_025_2_0,181126_025_2_3,181126_025_2_4"
tomo_training_list3="004_3_1,004_3_2,004_3_0,004_3_4,005_3_1,005_3_2,005_3_0,005_3_4,021_3_1,021_3_2,021_3_0,021_3_4,181119_002_3_1,181119_002_3_0,181119_002_3_2,181119_002_3_4,181119_030_3_1,181119_030_3_0,181119_030_3_2,181119_030_3_4,181126_002_3_1,181126_002_3_0,181126_002_3_2,181126_002_3_4,181126_012_3_1,181126_012_3_0,181126_012_3_2,181126_012_3_4,181126_025_3_1,181126_025_3_0,181126_025_3_2,181126_025_3_4"
tomo_training_list4="004_4_1,004_4_2,004_4_3,004_4_0,005_4_1,005_4_2,005_4_3,005_4_0,021_4_1,021_4_2,021_4_3,021_4_0,181119_002_4_1,181119_002_4_0,181119_002_4_3,181119_002_4_2,181119_030_4_1,181119_030_4_0,181119_030_4_3,181119_030_4_2,181126_002_4_1,181126_002_4_0,181126_002_4_3,181126_002_4_2,181126_012_4_1,181126_012_4_0,181126_012_4_3,181126_012_4_2,181126_025_4_1,181126_025_4_0,181126_025_4_3,181126_025_4_2"


export path_to_dataset_table="/struct/mahamid/Irene/fractions/fas_fractions_data.csv"
export segmentation_names='fas'
export split=0.8

# Data for the new model
export fractions_name="fas_fractions_004_005_021_ED"
export log_dir="/g/scb2/zaugg/trueba/3d-cnn/log_"$fractions_name
export model_path="./models/"$fractions_name
export n_epochs=300
export depth=2
export initial_features=8
export output_classes=1
export shuffle=false

# Data for old models for resuming training:
export retrain="False"
export path_to_old_model="None"
export models_notebook="/struct/mahamid/Irene/fractions/models_004_005_021_ED_fas.csv"

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
    export model_initial_name="shuffle_"$shuffle"_frac_"$fraction"_"
    sbatch ./submission_scripts/dataset_tables/training/training_runner.sh -fraction $fraction -path_to_dataset_table $path_to_dataset_table -tomo_training_list $tomo_training_list -split $split -output_classes $output_classes -log_dir $log_dir -model_initial_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -models_notebook $models_notebook
done
