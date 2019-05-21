#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 400G
#SBATCH --time 0-10:00
#SBATCH -o /g/scb2/zaugg/trueba/3d-cnn/slurm_outputs/1_cnn_training_submission.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu
#SBATCH -C gpu=1080Ti
#SBATCH --gres=gpu:1

module load GCC
module load Anaconda3

echo "Starting virtual environment"
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

export split=0.8
export label_name="ribosomes"
export model_nickname='8TOMOS_DATA_AUG_'
export model_path="/g/scb2/zaugg/trueba/3d-cnn/mixed_models"
export depth=2
export init_features=4
export iterations=1
export elu='True'
export log_dir="/g/scb2/zaugg/trueba/3d-cnn/mixed_logs"
echo "Calling python script..."
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/generate_mixed_training_set/1_cnn_training.py -split $split -label_name $label_name -model_nickname $model_nickname -model_path $model_path -depth $depth -init_features $init_features -iterations $iterations -elu $elu -log_dir $log_dir
echo "... done."
