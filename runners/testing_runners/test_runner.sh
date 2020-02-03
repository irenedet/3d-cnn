#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 10G
#SBATCH --time 0-0:05
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de



module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"

python3 test.py