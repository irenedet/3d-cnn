#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 30G
#SBATCH --time 0-02:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"
export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn

echo "Starting python script"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/generate_label_masks/mask_map.py
