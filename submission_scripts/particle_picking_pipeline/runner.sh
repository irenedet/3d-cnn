#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 0-02:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de





export path_to_raw='/scratch/trueba/cnn/004/4bin/cnn/rawtomogram/180426_004_4bin.hdf'
export output_dir='/scratch/trueba/3d-cnn/pipeline_TEST/TEST3/'
export path_to_model="/g/scb2/zaugg/trueba/3d-cnn/models/0_lay_6_len_128_32_DiceLoss_ELUactiv_2ndtry.pkl"
export label_name="ribosomes"


module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

bash ./particle_picking_pipeline/runner.sh -raw $path_to_raw -output $output_dir -model $path_to_model -label $label_name



