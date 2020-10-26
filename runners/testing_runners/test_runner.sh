#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 40G
#SBATCH --time 0-3:05
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de



#module load Anaconda3
echo "activating virtual environment"
#source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
#echo "... done"
export PYTHONPATH=/struct/mahamid/Irene/3d-unet/src
source ~/.bashrc
source activate /struct/mahamid/Irene/segmentation_ribo/.snakemake/conda/50db6a03
#python3 runners/testing_runners/hdf2mrc_convert.py
#python3 /struct/mahamid/Irene/3d-cnn/runners/testing_runners/test.py
python3 /struct/mahamid/Irene/3d-unet/scripts/test.py
