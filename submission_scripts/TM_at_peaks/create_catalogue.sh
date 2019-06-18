#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 128G
#SBATCH --time 0-2:00
#SBATCH -o slurm_outputs/cnn_evaluation.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/cnn_evaluation.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"

export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn

echo "Starting python script"
python3 ./pipelines/template_matching_at_peaks/create_catalogue.py
echo "... done."

