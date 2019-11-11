#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 5G
#SBATCH --time 0-00:10
#SBATCH -o slurm_outputs/threshold.%N.%j.out
#SBAtCH -e slurm_outputs/threshold.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"

echo "Starting python script"
python3 pipelines/generate_label_masks/threshold_maps.py


