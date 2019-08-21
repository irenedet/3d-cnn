#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 400G
#SBATCH --time 0-07:50
#SBATCH -o ynet.%N.%j.out
#SBAtCH -e ynet.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu
#SBATCH -C gpu=1080Ti
#SBATCH --gres=gpu:1

export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn/src/python
module load GCC
module load Anaconda3
echo 'activating virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo '... done.'

echo 'Training dice multi-label network'
python3 ./runners/testing_runners/testing_ynet.py