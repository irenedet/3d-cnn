#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 128G
#SBATCH --time 0-00:50
#SBATCH -o clustering_slurm.%N.%j.out
#SBAtCH -e clustering_slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


module load Anaconda3
export QT_QPA_PLATFORM='offscreen'

# To be modified by user
echo 'starting virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/


echo "starting python script"
python3 /g/scb2/zaugg/trueba/3d-cnn/runners/testing_runners/clustering_test.py
echo "... done."