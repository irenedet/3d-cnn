#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 128G
#SBATCH --time 0-05:50
#SBATCH -o data_aug_slurm.%N.%j.out
#SBAtCH -e data_aug_slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


module load Anaconda3
export QT_QPA_PLATFORM='offscreen'

# To be modified by user
echo 'starting virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/


echo "starting python script"
python3 /g/scb2/zaugg/trueba/3d-cnn/runners/testing_runners/test_data_augm.py
echo "... done."