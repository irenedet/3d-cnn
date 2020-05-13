#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 100G
#SBATCH --time 0-20:00
#SBATCH -o training.slurm.%N.%j.out
#SBATCH -e training.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBAtCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu
#SBATCH --gres=gpu:4 -n1 -c4



echo "Activating virtual environment"
module load Anaconda3
source activate $UPICKER_VENV_PATH
echo "done"
export QT_QPA_PLATFORM='offscreen'
usage()

{
    echo "usage: [[ [-tomos_set][-tomos_set tomos_set ]
                  [-yaml_file] [-yaml_file yaml_file] | [-h]]"
}


while [ "$1" != "" ]; do
    case $1 in
        -tomos_set | --tomos_set )   shift
                                tomos_set=$1
                                ;;
        -yaml_file | --yaml_file )   shift
                                yaml_file=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

export tomos_set=$tomos_set
export yaml_file=$yaml_file
echo "Submitting job for set" $tomos_set
python $UPICKER_PATH/PIPELINES/training/training.py -yaml_file $yaml_file -tomos_set $tomos_set
