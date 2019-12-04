#! /bin/bash

#SBATCH -A mbeck
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 20G
#SBATCH --time 0-05:20
#SBATCH -o slurm_outputs/fraction_datasets_cv.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/fraction_datasets_cv.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate $UPICKER_VENV_PATH
echo "... done"
echo UPICKER_PATH = $UPICKER_PATH

python3 $UPICKER_PATH/runners/testing_runners/copy_raw_partition.py

