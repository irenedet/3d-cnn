#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 64G
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
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
echo '... done.'

export training_data_path="/scratch/trueba/shrec/0_sph_masks/training_sets/top5_differentiated_training.h5"
export label_name="all_particles"
export split=850
export output_classes=6
export log_dir="./shrec_sph_logs"
export model_initial_name="just_checking_"
export model_path="shrec_models/multi-class"
export n_epochs=200

echo 'Testing multi-class segmentation'
python3 /g/scb2/zaugg/trueba/3d-cnn/SHREC_challenge/multi_class.py -data_path $training_data_path -label $label_name -split $split -classes $output_classes -log_dir $log_dir -model_name $model_initial_name -model_path $model_path -n_epochs $n_epochs
source deactivate
