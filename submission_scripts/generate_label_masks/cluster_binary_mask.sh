#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 25G
#SBATCH --time 0-03:10
#SBATCH -o slurm_outputs/cluster.%N.%j.out
#SBAtCH -e slurm_outputs/cluster.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"

echo "Starting python script"
python3 pipelines/generate_label_masks/cluster_binary_mask.py


