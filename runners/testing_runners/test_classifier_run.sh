#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 228G
#SBATCH --time 0-0:50
#SBATCH -o classifier_training.slurm.%N.%j.out
#SBAtCH -e classifier_training.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu
#SBATCH -C gpu=1080Ti
#SBATCH --gres=gpu:1



module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"

echo "starting python script:"
python3 ./runners/testing_runners/test_classifier.py
echo "... done."