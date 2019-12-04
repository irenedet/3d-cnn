#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 120G
#SBATCH --time 0-25:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu
#SBATCH -C gpu=2080Ti
#SBATCH --gres=gpu:4

module load GCC
module load Anaconda3
echo 'activating virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo '... done.'

# Training set parameters:
#tomo_training_list="190301/005
#190301/031
#190301/033"

tomo_training_list="200,201,203,204,205,206,207,208,240,241,242,243,244,245,246,247"

export path_to_dataset_table="/struct/mahamid/Irene/liang_data/multiclass/liang_data_multiclass.csv"
export segmentation_names='70S,50S,memb'
export split=0.7

# Data for the new model
export log_dir="/g/scb2/zaugg/trueba/3d-cnn/log_liang_multi_label"
export model_initial_name="NO_DA_"
export model_path="./models/liang_multi_class"
export n_epochs=200
export depth=2
export initial_features=8
export output_classes=3

# Data for old models for resuming training:
export retrain="False"
export path_to_old_model="None"


echo 'Training dice multi-label network'
python3 $UPICKER_PATH/runners/dataset_tables/training/dice_unet_training.py -dataset_table $path_to_dataset_table -tomo_training_list "${tomo_training_list[@]}" -split $split -classes $output_classes -log_dir $log_dir -model_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features

echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $model_path"/SCRIPT_"$model_initial_name".txt"
