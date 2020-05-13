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

export yaml_file="/struct/mahamid/Irene/3d-cnn/submission_parameters_files/dataset_tables/saras_data/FAS/single_filt/evaluation/partition/config_predict.yaml"
export tomos_set=1

echo "Partitioning dataset"
python $UPICKER_PATH/PIPELINES/prediction/partition.py -yaml_file $yaml_file -tomos_set $tomos_set

echo "Segmenting partition"
python3 $UPICKER_PATH/pipelines/prediction/segment.py -yaml_file $yaml_file -tomo_name $tomo_name

echo "Reconstructing segmentation"
python3 $UPICKER_PATH/pipelines/prediction/assemble.py -yaml_file $yaml_file -tomo_name $tomo_name

echo "Getting cluster centroids motl"
python $UPICKER_PATH/pipelines/prediction/cluster_motl.py -yaml_file $yaml_file -tomo_name $tomo_name

echo "Selecting coordinates within mask"
python $UPICKER_PATH/pipelines/prediction/mask_motl.py -yaml_file $yaml_file -tomo_name $tomo_name

