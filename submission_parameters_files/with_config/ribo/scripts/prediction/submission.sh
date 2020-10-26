#!/usr/bin/env bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 30G
#SBATCH --time 0-1:00
#SBATCH -o predict.slurm.%N.%j.out
#SBAtCH -e predict.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBAtCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu
#SBATCH --gres=gpu:4

echo "Activating virtual environment"
#module load Anaconda3
#export UPICKER_VENV_PATH="/struct/mahamid/Irene/envs/.conda/3d-cnn"
#source activate 3d-cnn
source ~/.bashrc
source activate /struct/mahamid/Irene/segmentation_ribo/.snakemake/conda/50db6a03
export PYTHONPATH="/struct/mahamid/Irene/3d-cnn/src/python"
export UPICKER_PATH="/struct/mahamid/Irene/3d-cnn"
usage()

{
    echo "usage: [[ [-output output_dir][-test_partition test_partition ]
                  [-model path_to_model] [-label label_name]
                  [-out_h5 output_h5_file_path] [-conf conf]] | [-h]]"
}
while [ "$1" != "" ]; do
    case $1 in
        -set | --set )   shift
                                set=$1
                                ;;
        -config | --config )   shift
                                config=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

export yaml_file=$config
export set=$set
echo "Analyzing set" $set

echo "Partitioning dataset"
python $UPICKER_PATH/PIPELINES/prediction/partition.py -yaml_file $yaml_file -tomos_set $set

echo "Segmenting partition" $set
python3 $UPICKER_PATH/PIPELINES/prediction/segment.py -yaml_file $yaml_file -tomos_set $set --gpu $CUDA_VISIBLE_DEVICES

echo "Reconstructing segmentation"
python3 $UPICKER_PATH/PIPELINES/prediction/assemble.py -yaml_file $yaml_file -tomos_set $set

echo "Getting cluster centroids motl"
python3 $UPICKER_PATH/PIPELINES/prediction/cluster_motl.py -yaml_file $yaml_file -tomos_set $set

echo "Selecting coordinates within mask"
python3 $UPICKER_PATH/PIPELINES/prediction/mask_motl.py -yaml_file $yaml_file -tomos_set $set

echo "Performing precision-recall analysis"
python3 $UPICKER_PATH/PIPELINES/pr_analysis/pr_analysis.py -yaml_file $yaml_file -tomos_set $set
