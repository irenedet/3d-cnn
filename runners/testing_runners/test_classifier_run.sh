#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 228G
#SBATCH --time 0-0:20
#SBATCH -o classifier_training.slurm.%N.%j.out
#SBAtCH -e classifier_training.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu
#SBATCH -C gpu=1080Ti
#SBATCH --gres=gpu:1



module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"


export volume_side=64
export spl=0.7  # training set percentage
export epochs=40  # 40  # 100
export depth=3  # 2
export initial_features=8  # 4
export output_classes=2
export
export train_log_dir="/g/scb2/zaugg/trueba/3d-cnn/logs_classification3D/"
export data_path="/scratch/trueba/3Dclassifier/liang_data/training_data/200/training_set.h5"
export semantic_classes='70S,50S'
export model_initial_name="Test_CroSSEntLoss_"



echo "starting python script:"
python3 ./runners/testing_runners/test_classifier.py -train_log_dir $train_log_dir -data_path $data_path -semantic_classes $semantic_classes -model_initial_name $model_initial_name -volume_side $volume_side -spl $spl -epochs $epochs -output_classes $output_classes -depth $depth -initial_features $initial_features
echo "... done."

train_log_dir
data_path
semantic_classes
model_initial_name
volume_side
spl
epochs
output_classes
depth
initial_features