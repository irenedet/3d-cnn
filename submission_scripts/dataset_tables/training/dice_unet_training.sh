#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 200G
#SBATCH --time 0-7:00
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
#tomo_training_list="190301/005
#190301/031
#190301/033"

tomo_training_list="181119/002
181119/030
181126/002
181126/012
181126/025"

export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
export segmentation_names='ribo,fas,memb'
export split=0.7

# Data for the new model
export log_dir="/g/scb2/zaugg/trueba/3d-cnn/log_dice_multi_label"
export model_initial_name="R_ED_191119_002_030_191126_002_012_025__R_all_but_003_"
export model_path="/g/scb2/zaugg/trueba/3d-cnn/models/dice_multi_label/retrained/ED"
export n_epochs=40
export depth=2
export initial_features=8
export output_classes=3

# Data for old models for resuming training:
export retrain="True"
export path_to_old_model="/g/scb2/zaugg/trueba/3d-cnn/models/dice_multi_label/retrained/Retrain_D2_IF8_NA_except_180711_003ribo_fas_memb_D_2_IF_8.pkl"


echo 'Training dice multi-label network'
python3 ./runners/dataset_tables/training/dice_unet_training.py -dataset_table $path_to_dataset_table -tomo_training_list "${tomo_training_list[@]}" -split $split -classes $output_classes -log_dir $log_dir -model_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features

echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $model_path"/SCRIPT_"$model_initial_name".txt"
