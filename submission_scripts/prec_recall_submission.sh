#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 0-01:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de



export path_to_motl='/scratch/trueba/3d-cnn/pipeline_TEST/TEST2/motl_4443.csv'
export path_to_clean='/scratch/trueba/cnn/004/4bin/cnn/motl_clean_4b.em'
export output_dir='/scratch/trueba/3d-cnn/pipeline_TEST/TEST3/'
export label_name='ribosomes'

module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

bash ./prec_recall_runner.sh -motl $path_to_motl -output $output_dir -clean $path_to_clean -label $label_name
