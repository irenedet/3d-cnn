#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 300G
#SBATCH --time 0-8:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu
#SBATCH -C gpu=1080Ti
#SBATCH --gres=gpu:1

module load GCC
module load Anaconda3
echo 'activating virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo '... done.'

# Training set parameters:
training_list="/scratch/trueba/liang_data/172/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/173/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/174/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/175/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/176/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/177/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/178/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/179/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/180/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/190/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/191/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/192/train_and_test_partitions/full_partition.h5"
#/scratch/trueba/liang_data/203/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/204/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/205/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/206/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/207/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/208/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/240/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/241/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/242/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/243/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/244/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/245/train_and_test_partitions/full_partition.h5"
#/scratch/trueba/liang_data/246/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/247/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/248/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/249/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/250/train_and_test_partitions/full_partition.h5
#/scratch/trueba/liang_data/251/train_and_test_partitions/full_partition.h5"

export label_name="ribo"
export segmentation_names='ribo'
export split=0.8
export skip=-1 #4 except 180711_003 (healthy), except 181126_012 (ED)

# Data for the new model
export log_dir="/g/scb2/zaugg/trueba/log_lang_unets"
export model_initial_name="2_NO_DA_BN_172_192_"
export model_path="/g/scb2/zaugg/trueba/3d-cnn/models/lang_unets"
export n_epochs=80
export depth=2
export initial_features=8
export output_classes=1

#Special parameters
export BN="True"
export dropout=0 #0.15

# Data for old models for resuming training:
export retrain="False"
export path_to_old_model="not_needed"

# Models notebook
export models_notebook="/g/scb2/zaugg/trueba/3d-cnn/models/lang_unets/models_notebook.csv"

if [ $BN == "True" ]; then
    echo "The network includes Batch Normalization"
    echo 'Training is starting...'
    python3 ./pipelines/dice_multi-class/training/dice_multi_class_BN_training.py -training_paths_list "${training_list[@]}" -label $label_name -split $split -classes $output_classes -log_dir $log_dir -model_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -skip $skip -models_notebook $models_notebook

elif [ $BN == "False" ]; then
    echo "The network does not include Batch Normalization"
    echo 'Training is starting...'
    python3 ./pipelines/dice_multi-class/training/dice_multi_class_unet_training.py -training_paths_list "${training_list[@]}" -label $label_name -split $split -classes $output_classes -log_dir $log_dir -model_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -skip $skip -models_notebook $models_notebook
elif [$dropout > 0]; then
    echo "Training a U-Net with dropout"
    python3 ./pipelines/dice_multi-class/training/dice_multi_class_dropout_training.py -training_paths_list "${training_list[@]}" -label $label_name -split $split -classes $output_classes -log_dir $log_dir -model_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -skip $skip -models_notebook $models_notebook -dropout $dropout
else
    echo "Batch normalization is not well specified"
fi

echo "... done."

echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $model_path"/SCRIPT_"$model_initial_name".txt"

