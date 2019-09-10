#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 128G
#SBATCH --time 0-2:50
#SBATCH -o slurm_outputs/cnn_evaluation.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/cnn_evaluation.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

#SBAtCH -p gpu
#SBAtCH -C gpu=1080Ti
#SBAtCH --gres=gpu:1

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"

export QT_QPA_PLATFORM='offscreen'

export label_name="NO_DA_ribo_D_2_IF_8"
export output_dir="/g/scb2/zaugg/trueba/3d-cnn/"
export min_cluster_size=1500
export max_cluster_size=4000
export test_partition="/scratch/trueba/liang_data/192/train_and_test_partitions/full_partition.h5"
export path_to_model="/g/scb2/zaugg/trueba/3d-cnn/models/lang_unets/NO_DA_ribo_D_2_IF_8.pkl"
export output_classes=1
export xdim=928
export ydim=928
export zdim=450
export class_number=0
export min_peak_distance=25
export new_loader=True
export initial_features=8
export unet_depth=2
export box_overlap=12



