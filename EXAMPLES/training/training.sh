#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 30G
#SBATCH --time 0-15:15
#SBATCH -o predict.slurm.%N.%j.out
#SBAtCH -e predict.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBAtCH --mail-user=irene.de.teresa@embl.de
#SBAtCH -p gpu
#SBAtCH --gres=gpu:4,-n1,-c4

echo "Activating virtual environment"
module load Anaconda3
source activate $UPICKER_VENV_PATH
echo "done"

export yaml_file="training/config.yaml"

tomo_training_list="180426/027,180426/028"


echo "Submitting job for set" $tomo_training_list
python $UPICKER_PATH/pipelines/training/training.py -yaml_file $yaml_file -tomo_training_list $tomo_training_list
