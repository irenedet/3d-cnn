#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 300G
#SBATCH --time 0-14:00
#SBATCH -o slurm_outputs/training_multiclass.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/training_multiclass.slurm.%N.%j.err
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
training_list="/scratch/trueba/liang_data/200/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/201/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/203/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/204/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/205/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/206/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/207/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/208/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/240/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/241/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/242/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/243/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/244/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/245/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/246/multi_class/train_and_test_partitions/full_partition.h5
/scratch/trueba/liang_data/247/multi_class/train_and_test_partitions/full_partition.h5"

export segmentation_names='70S,50S,memb'
export split=0.8
export skip=3 #4 except 180711_003 (healthy), except 181126_012 (ED)

# Data for the new model
export log_dir="/g/scb2/zaugg/trueba/3d-cnn/log_liang_multi_label"
export final_activation="softmax"
export model_initial_name="NO_DA_"$final_activation"_"
export model_path="models/liang_multi_label"
export n_epochs=200
export depth=3
export initial_features=8
export output_classes=3

# Data for old models for resuming training:
export retrain="False"
export path_to_old_model="None"

# TODO: Save data in notebook
export models_notebook=$model_path"/models_notebook.csv"

echo 'Training dice multi-label network'
python3 ./pipelines/dice_multi-class/training/dice_multi_class_unet_training.py -training_paths_list "${training_list[@]}" -split $split -classes $output_classes -log_dir $log_dir -model_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -skip $skip -models_notebook $models_notebook -final_activation $final_activation

echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $model_path"/SCRIPT_"$model_initial_name".txt"