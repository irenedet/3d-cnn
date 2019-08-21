#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 200G
#SBATCH --time 0-04:00
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
training_list="/struct/mahamid/Irene/yeast/healthy/180426/004/G_sigma1_non_sph/train_and_test_partitions/DA_elastic_gauss_rot.h5"

export label_name="ribo_fas_memb"
export segmentation_names='ribo,fas,memb'
export split=0.8
export skip=3 #4 except 180711_003 (healthy), except 181126_012 (ED)

# Data for the new model
export log_dir="/g/scb2/zaugg/trueba/3d-cnn/log_dice_multi_label"
export model_initial_name="elastic_rot_DA_004_600N_"
export model_path="/g/scb2/zaugg/trueba/3d-cnn/models/dice_multi_label"
export n_epochs=40
export depth=2
export initial_features=8
export output_classes=3

# Data for old models for resuming training:
export retrain="False"
export path_to_old_model="/g/scb2/zaugg/trueba/3d-cnn/models/dice_multi_label/retrained/Retrain_retrained_except_180711_003ribo_fas_memb_D_2_IF_8.pkl"

# Todo:
export models_notebook="/g/scb2/zaugg/trueba/3d-cnn/models/dice_multi_label/yeast_models_notebook.csv"

echo 'Training dice multi-label network'
python3 ./pipelines/dice_multi-class/training/dice_multi_class_unet_training.py -training_paths_list "${training_list[@]}" -label $label_name -split $split -classes $output_classes -log_dir $log_dir -model_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -skip $skip -models_notebook $models_notebook

echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $model_path"/SCRIPT_"$model_initial_name".txt"
