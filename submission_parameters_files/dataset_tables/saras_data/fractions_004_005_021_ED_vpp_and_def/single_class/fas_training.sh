#! /bin/bash

FRACTIONS="0 1 2 3 4"


tomo_training_list0="004_0_1,004_0_2,004_0_3,004_0_4,005_0_1,005_0_2,005_0_3,005_0_4,021_0_1,021_0_2,021_0_3,021_0_4,181119_002_0_1,181119_002_0_2,181119_002_0_3,181119_002_0_4,181119_030_0_1,181119_030_0_2,181119_030_0_3,181119_030_0_4,181126_002_0_1,181126_002_0_2,181126_002_0_3,181126_002_0_4,181126_012_0_1,181126_012_0_2,181126_012_0_3,181126_012_0_4,181126_025_0_1,181126_025_0_2,181126_025_0_3,181126_025_0_4,190301_003_0_1,190301_003_0_2,190301_003_0_3,190301_003_0_4,190301_005_0_1,190301_005_0_2,190301_005_0_3,190301_005_0_4,190301_009_0_1,190301_009_0_2,190301_009_0_3,190301_009_0_4,190301_016_0_1,190301_016_0_2,190301_016_0_3,190301_016_0_4,190301_022_0_1,190301_022_0_2,190301_022_0_3,190301_022_0_4,190301_028_0_1,190301_028_0_2,190301_028_0_3,190301_028_0_4,190301_031_0_1,190301_031_0_2,190301_031_0_3,190301_031_0_4,190301_032_0_1,190301_032_0_2,190301_032_0_3,190301_032_0_4,190301_033_0_1,190301_033_0_2,190301_033_0_3,190301_033_0_4,190301_035_0_1,190301_035_0_2,190301_035_0_3,190301_035_0_4,190301_037_0_1,190301_037_0_2,190301_037_0_3,190301_037_0_4,190301_043_0_1,190301_043_0_2,190301_043_0_3,190301_043_0_4,190301_045_0_1,190301_045_0_2,190301_045_0_3,190301_045_0_4"
tomo_training_list1="004_1_0,004_1_2,004_1_3,004_1_4,005_1_0,005_1_2,005_1_3,005_1_4,021_1_0,021_1_2,021_1_3,021_1_4,181119_002_1_0,181119_002_1_2,181119_002_1_3,181119_002_1_4,181119_030_1_0,181119_030_1_2,181119_030_1_3,181119_030_1_4,181126_002_1_0,181126_002_1_2,181126_002_1_3,181126_002_1_4,181126_012_1_0,181126_012_1_2,181126_012_1_3,181126_012_1_4,181126_025_1_0,181126_025_1_2,181126_025_1_3,181126_025_1_4,190301_003_1_0,190301_003_1_2,190301_003_1_3,190301_003_1_4,190301_005_1_0,190301_005_1_2,190301_005_1_3,190301_005_1_4,190301_009_1_0,190301_009_1_2,190301_009_1_3,190301_009_1_4,190301_016_1_0,190301_016_1_2,190301_016_1_3,190301_016_1_4,190301_022_1_0,190301_022_1_2,190301_022_1_3,190301_022_1_4,190301_028_1_0,190301_028_1_2,190301_028_1_3,190301_028_1_4,190301_031_1_0,190301_031_1_2,190301_031_1_3,190301_031_1_4,190301_032_1_0,190301_032_1_2,190301_032_1_3,190301_032_1_4,190301_033_1_0,190301_033_1_2,190301_033_1_3,190301_033_1_4,190301_035_1_0,190301_035_1_2,190301_035_1_3,190301_035_1_4,190301_037_1_0,190301_037_1_2,190301_037_1_3,190301_037_1_4,190301_043_1_0,190301_043_1_2,190301_043_1_3,190301_043_1_4,190301_045_1_0,190301_045_1_2,190301_045_1_3,190301_045_1_4"
tomo_training_list2="004_2_1,004_2_0,004_2_3,004_2_4,005_2_1,005_2_0,005_2_3,005_2_4,021_2_1,021_2_0,021_2_3,021_2_4,181119_002_2_1,181119_002_2_0,181119_002_2_3,181119_002_2_4,181119_030_2_1,181119_030_2_0,181119_030_2_3,181119_030_2_4,181126_002_2_1,181126_002_2_0,181126_002_2_3,181126_002_2_4,181126_012_2_1,181126_012_2_0,181126_012_2_3,181126_012_2_4,181126_025_2_1,181126_025_2_0,181126_025_2_3,181126_025_2_4,190301_003_2_0,190301_003_2_1,190301_003_2_3,190301_003_2_4,190301_005_2_0,190301_005_2_1,190301_005_2_3,190301_005_2_4,190301_009_2_0,190301_009_2_1,190301_009_2_3,190301_009_2_4,190301_016_2_0,190301_016_2_1,190301_016_2_3,190301_016_2_4,190301_022_2_0,190301_022_2_1,190301_022_2_3,190301_022_2_4,190301_028_2_0,190301_028_2_1,190301_028_2_3,190301_028_2_4,190301_031_2_0,190301_031_2_1,190301_031_2_3,190301_031_2_4,190301_032_2_0,190301_032_2_1,190301_032_2_3,190301_032_2_4,190301_033_2_0,190301_033_2_1,190301_033_2_3,190301_033_2_4,190301_035_2_0,190301_035_2_1,190301_035_2_3,190301_035_2_4,190301_037_2_0,190301_037_2_1,190301_037_2_3,190301_037_2_4,190301_043_2_0,190301_043_2_1,190301_043_2_3,190301_043_2_4,190301_045_2_0,190301_045_2_1,190301_045_2_3,190301_045_2_4"
tomo_training_list3="004_3_1,004_3_2,004_3_0,004_3_4,005_3_1,005_3_2,005_3_0,005_3_4,021_3_1,021_3_2,021_3_0,021_3_4,181119_002_3_1,181119_002_3_0,181119_002_3_2,181119_002_3_4,181119_030_3_1,181119_030_3_0,181119_030_3_2,181119_030_3_4,181126_002_3_1,181126_002_3_0,181126_002_3_2,181126_002_3_4,181126_012_3_1,181126_012_3_0,181126_012_3_2,181126_012_3_4,181126_025_3_1,181126_025_3_0,181126_025_3_2,181126_025_3_4,190301_003_3_0,190301_003_3_1,190301_003_3_2,190301_003_3_4,190301_005_3_0,190301_005_3_1,190301_005_3_2,190301_005_3_4,190301_009_3_0,190301_009_3_1,190301_009_3_2,190301_009_3_4,190301_016_3_0,190301_016_3_1,190301_016_3_2,190301_016_3_4,190301_022_3_0,190301_022_3_1,190301_022_3_2,190301_022_3_4,190301_028_3_0,190301_028_3_1,190301_028_3_2,190301_028_3_4,190301_031_3_0,190301_031_3_1,190301_031_3_2,190301_031_3_4,190301_032_3_0,190301_032_3_1,190301_032_3_2,190301_032_3_4,190301_033_3_0,190301_033_3_1,190301_033_3_2,190301_033_3_4,190301_035_3_0,190301_035_3_1,190301_035_3_2,190301_035_3_4,190301_037_3_0,190301_037_3_1,190301_037_3_2,190301_037_3_4,190301_043_3_0,190301_043_3_1,190301_043_3_2,190301_043_3_4,190301_045_3_0,190301_045_3_1,190301_045_3_2,190301_045_3_4"
tomo_training_list4="004_4_1,004_4_2,004_4_3,004_4_0,005_4_1,005_4_2,005_4_3,005_4_0,021_4_1,021_4_2,021_4_3,021_4_0,181119_002_4_1,181119_002_4_0,181119_002_4_3,181119_002_4_2,181119_030_4_1,181119_030_4_0,181119_030_4_3,181119_030_4_2,181126_002_4_1,181126_002_4_0,181126_002_4_3,181126_002_4_2,181126_012_4_1,181126_012_4_0,181126_012_4_3,181126_012_4_2,181126_025_4_1,181126_025_4_0,181126_025_4_3,181126_025_4_2,190301_003_4_0,190301_003_4_1,190301_003_4_2,190301_003_4_3,190301_005_4_0,190301_005_4_1,190301_005_4_2,190301_005_4_3,190301_009_4_0,190301_009_4_1,190301_009_4_2,190301_009_4_3,190301_016_4_0,190301_016_4_1,190301_016_4_2,190301_016_4_3,190301_022_4_0,190301_022_4_1,190301_022_4_2,190301_022_4_3,190301_028_4_0,190301_028_4_1,190301_028_4_2,190301_028_4_3,190301_031_4_0,190301_031_4_1,190301_031_4_2,190301_031_4_3,190301_032_4_0,190301_032_4_1,190301_032_4_2,190301_032_4_3,190301_033_4_0,190301_033_4_1,190301_033_4_2,190301_033_4_3,190301_035_4_0,190301_035_4_1,190301_035_4_2,190301_035_4_3,190301_037_4_0,190301_037_4_1,190301_037_4_2,190301_037_4_3,190301_043_4_0,190301_043_4_1,190301_043_4_2,190301_043_4_3,190301_045_4_0,190301_045_4_1,190301_045_4_2,190301_045_4_3"


export path_to_dataset_table="/struct/mahamid/Irene/fractions/fas_fractions_data.csv"
export segmentation_names='fas'
export split=0.8

# Data for the new model
export fractions_name="fas_fractions_004_005_021_ED_and_def"
export log_dir="/g/scb2/zaugg/trueba/3d-cnn/log_"$fractions_name
export model_path="./models/"$fractions_name
export n_epochs=300
export depth=3
export initial_features=8
export output_classes=1
export shuffle=false

# Data for old models for resuming training:
export retrain="False"
export path_to_old_model="None"
export models_notebook="/struct/mahamid/Irene/fractions/models_004_005_021_ED_and_def_fas.csv"

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