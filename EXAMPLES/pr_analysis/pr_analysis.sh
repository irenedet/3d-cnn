#!/usr/bin/env bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 50G
#SBATCH --time 0-1:15
#SBATCH -o predict.slurm.%N.%j.out
#SBAtCH -e predict.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBAtCH --mail-user=irene.de.teresa@embl.de
#SBAtCH -p gpu
#SBAtCH --gres=gpu:4,-n1,-c4

echo "Activating virtual environment"
module load Anaconda3
source activate $UPICKER_VENV_PATH

export yaml_file="/struct/mahamid/Irene/test_3d_cnn/pr_analysis/config.yaml"
TOMOS="180426/027"


for tomo_name in $TOMOS
do
  echo "Performing precision-recall analysis"
  python $UPICKER_PATH/pipelines/pr_analysis/pr_analysis.py -yaml_file $yaml_file -tomo_name $tomo_name
done

