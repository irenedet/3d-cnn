#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 200G
#SBATCH --time 0-10:00
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

#training path number to be ignored while training (e.g. to be used as test set)
export fraction=4

export skip=$fraction","$(($fraction + 5))","$(($fraction + 10))
# Training set parameters:
training_list=$(ls "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_"$fraction"/"*)

export label_name="ribo_fas_memb"
export segmentation_names='ribo,fas,memb'
export split=0.8


# Data for the new model
export log_dir="/g/scb2/zaugg/trueba/3d-cnn/log_dice_multi_label/cross_validation"
export model_initial_name="021_005_004_fraction_"$fraction"_200_epochs"
export model_path="/g/scb2/zaugg/trueba/3d-cnn/models/dice_multi_label/cross_validation"
export n_epochs=200
export depth=2
export initial_features=8
export output_classes=3

# Data for old models for resuming training:
export retrain="False"
export path_to_old_model="None"

# Save unet description on models notebook
export models_notebook=$model_path"/models_notebook.csv"

echo 'Training dice multi-label network'
python3 ./pipelines/dice_multi-class/training/dice_multi_class_unet_training.py -training_paths_list "${training_list[@]}" -label $label_name -split $split -classes $output_classes -log_dir $log_dir -model_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -skip $skip -models_notebook $models_notebook

echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $model_path"/SCRIPT_"$model_initial_name".txt"
