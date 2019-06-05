#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 200G
#SBATCH --time 0-17:00
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
training_list="/struct/mahamid/Irene/yeast/vpp/180426_004/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5
/struct/mahamid/Irene/yeast/vpp/180426_005/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5
/struct/mahamid/Irene/yeast/vpp/180426_021/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5
/struct/mahamid/Irene/yeast/vpp/180426_024/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5
/struct/mahamid/Irene/yeast/vpp/180711_003/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5
/struct/mahamid/Irene/yeast/vpp/180711_004/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5
/struct/mahamid/Irene/yeast/vpp/180711_005/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5
/struct/mahamid/Irene/yeast/vpp/180711_018/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5"

export label_name="ribo_fas_memb"
export segmentation_names='ribo,fas,memb'
export split=0.7
export skip=4 #except 180711_003

# Data for the new model
export log_dir="/g/scb2/zaugg/trueba/3d-cnn/log_dice_multi_label"
export model_initial_name="RRrR_180711_all_except_180711_003_"
export model_path="/g/scb2/zaugg/trueba/3d-cnn/models/dice_multi_label/retrained"
export n_epochs=40
export depth=2
export initial_features=8
export output_classes=3

# Data for old models for resuming training:
export retrain="True"
export path_to_old_model="/g/scb2/zaugg/trueba/3d-cnn/models/dice_multi_label/retrained/Retrain_retrained_except_180711_003ribo_fas_memb_D_2_IF_8.pkl"


echo 'Training dice multi-label network'
python3 ./pipelines/dice_multi-class/training/dice_multi_class_unet_training.py -training_paths_list "${training_list[@]}" -label $label_name -split $split -classes $output_classes -log_dir $log_dir -model_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -skip $skip

echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT "/g/scb2/zaugg/trueba/3d-cnn/models/dice_multi_label/retrained/SCRIPT_"$model_initial_name".txt"
