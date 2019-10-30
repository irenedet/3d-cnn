#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 10G
#SBATCH --time 0-00:60
#SBATCH -o get_shapes.slurm.%N.%j.out
#SBAtCH -e get_shapes.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"


echo "starting python script:"
python3 runners/dataset_tables/get_shapes.py
echo "... done."



