#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 128G
#SBATCH --time 0-05:00
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
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
echo '... done.'

export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn

export training_data_path="/scratch/trueba/3d-cnn/training_data/dice-multi-class/004/G_sigma1/train_and_test_partitions/train_partition.h5"
export label_name="ribo_fas_memb"
export split=80
export output_classes=3 #background is not a class
export log_dir="/g/scb2/zaugg/trueba/3d-cnn/log_dice_multi_label"
export model_initial_name="1_w_1_1_1_"
export model_path="models/dice_multi_label"
export n_epochs=10
export weight='1,1,1' #background, ribos, fas, membranes
#'0.01,1,10,1' #background, ribos, fas, membranes

echo 'Training dice multi-label network'
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/dice_multi-class/training/dice_multi_class_unet_training.py -data_path $training_data_path -label $label_name -split $split -classes $output_classes -log_dir $log_dir -model_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -weight $weight

