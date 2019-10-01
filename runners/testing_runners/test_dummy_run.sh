#! /bin/bash
#module load Anaconda3
#export QT_QPA_PLATFORM='offscreen'
#
## To be modified by user
#echo 'starting virtual environment'
#source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
#echo "... done loading venv."
#
#echo "Starting dummy script"
#python3 /g/scb2/zaugg/trueba/3d-cnn/runners/testing_runners/test_dummy.py
#echo "done!"

#! /bin/bash

FRACTIONS="0"
tomo_training_list0="181119_002_0_1"


export path_to_dataset_table="/struct/mahamid/Irene/fractions/fas_fractions_data.csv"
export segmentation_names='fas'
export split=0.8

# Data for the new model
export fractions_name="dummy_test"
export log_dir="/g/scb2/zaugg/trueba/3d-cnn/log_"$fractions_name
export model_path="./models/"$fractions_name
export n_epochs=3
export depth=1
export initial_features=2
export output_classes=1
export shuffle=false

# Data for old models for resuming training:
export retrain="False"
export path_to_old_model="None"
export models_notebook="/struct/mahamid/Irene/fractions/dummy.csv"

for fraction in $FRACTIONS
do
    echo "Submitting job for fraction "$fraction
    if [ $fraction == 0 ]; then
        tomo_training_list=$tomo_training_list0
    else
        echo "Inexistent fraction!"
    fi

    echo 'Job for fraction' $fraction
    export model_initial_name="shuffle_"$shuffle"_frac_"$fraction"_"
    sbatch ./submission_scripts/dataset_tables/training/training_runner.sh -fraction $fraction -path_to_dataset_table $path_to_dataset_table -tomo_training_list $tomo_training_list -split $split -output_classes $output_classes -log_dir $log_dir -model_initial_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -models_notebook $models_notebook
done
