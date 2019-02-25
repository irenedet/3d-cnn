#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 0-00:30
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de



export path_to_motl='/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_16_5_bis_/motl_4621.csv'
export output_dir='/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_16_5_bis_/'
export path_to_clean='/scratch/trueba/3d-cnn/clean/180426_006/motl_clean_4b.em'
export label_name='ribosomes'

module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

bash ./prec_recall_runner.sh -motl $path_to_motl -output $output_dir -clean $path_to_clean -label $label_name
