#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 30G
#SBATCH --time 0-15:15
#SBATCH -o predict.slurm.%N.%j.out
#SBATCH -e predict.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBAtCH --mail-user=irene.de.teresa@embl.de
#SBAtCH -p gpu
#SBAtCH --gres=gpu:4,-n1,-c4

echo "Activating virtual environment"
module load Anaconda3
source activate $UPICKER_VENV_PATH
echo "done"

export yaml_file="/struct/mahamid/Irene/EXAMPLES/training/config.yaml"

tomos_set="1"

echo "Submitting job for set" $tomos_set
python $UPICKER_PATH/PIPELINES/training/training_debugging.py -yaml_file $yaml_file -tomos_set $tomos_set
