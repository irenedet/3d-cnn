#!/usr/bin/env bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 60G
#SBATCH --time 0-1:35
#SBATCH -o predict.slurm.%N.%j.out
#SBAtCH -e predict.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBAtCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu
#SBATCH --gres=gpu:4 -n1 -c4

echo "Activating virtual environment"
module load Anaconda3
source activate $UPICKER_VENV_PATH

usage()

{
    echo "usage: [[ [-set][-set set] | [-h]]"
}
while [ "$1" != "" ]; do
    case $1 in
        -set | --set )   shift
                                set=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

export yaml_file="/struct/mahamid/Irene/3d-cnn/submission_parameters_files/with_config/NPC/multi_class/evaluation/config_predict.yaml"
export tomos_set=$set
echo "Analyzing set" $set

echo "Partitioning dataset"
python $UPICKER_PATH/PIPELINES/prediction/partition.py -yaml_file $yaml_file -tomos_set $tomos_set

echo "Segmenting partition"
python $UPICKER_PATH/PIPELINES/prediction/segment.py -yaml_file $yaml_file -tomos_set $tomos_set

echo "Reconstructing segmentation"
python $UPICKER_PATH/PIPELINES/prediction/assemble.py -yaml_file $yaml_file -tomos_set $tomos_set

#echo "Getting cluster centroids motl"
#python $UPICKER_PATH/PIPELINES/prediction/cluster_motl.py -yaml_file $yaml_file -tomos_set $tomos_set
#
#echo "Selecting coordinates within mask"
#python $UPICKER_PATH/PIPELINES/prediction/mask_motl.py -yaml_file $yaml_file -tomos_set $tomos_set
#
#echo "Performing precision-recall analysis"
#python $UPICKER_PATH/PIPELINES/pr_analysis/pr_analysis.py -yaml_file $yaml_file -tomos_set $tomos_set