#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 60G
#SBATCH --time 0-00:45
#SBATCH -o slurm_outputs/print_middle_slice.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/print_middle_slice.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


#module load Anaconda3
#echo "activating virtual environment"
#source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
#echo "... done"

export QT_QPA_PLATFORM='offscreen'

echo "starting script"
python3 runners/testing_runners/print_middle_slice_of_datasets.py
echo "done!"