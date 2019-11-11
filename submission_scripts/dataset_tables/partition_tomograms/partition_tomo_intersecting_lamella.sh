#!/usr/bin/env bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 10G
#SBATCH --time 0-00:25
#SBATCH -o slurm_outputs/partition_tomo_slurm.%N.%j.out
#SBAtCH -e slurm_outputs/partition_tomo_slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

usage()

{
    echo "usage: todo---[[ [-output output_dir][-test_partition test_partition ]
                  [-model path_to_model] [-label label_name]
                  [-out_h5 output_h5_file_path] [-conf conf]] | [-h]]"
}


while [ "$1" != "" ]; do
    case $1 in
        -dataset_table | --dataset_table )   shift
                                dataset_table=$1
                                ;;
        -tomo_name | --tomo_name )   shift
                                tomo_name=$1
                                ;;
        -output_dir | --output_dir )   shift
                                output_dir=$1
                                ;;
        -box_side | --box_side )   shift
                                box_side=$1
                                ;;
        -write_on_table | --write_on_table )   shift
                                write_on_table=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

export overlap=12
export dataset_table=$dataset_table
export box_side=$box_side
export tomo_name=$tomo_name
export output_dir=$output_dir

echo output_dir = $output_dir
export output_dir=$output_dir"/"$tomo_name
export outh5=$output_dir"/tomo_partition.h5"
export write_on_table=$write_on_table

echo tomo_name = $tomo_name
echo output_dir = $output_dir
echo box_side = $box_side
echo overlap = $overlap
echo outh5 = $outh5

module load Anaconda3
echo 'starting virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/

export outh5=$output_dir"/tomo_partition.h5"
echo 'starting python script'
python3 runners/dataset_tables/particle_picking/partition_from_dataset_table.py -dataset_table $dataset_table -tomo_name $tomo_name -outh5 $outh5 -output $output_dir -box $box_side -overlap $overlap -write_on_table $write_on_table
echo 'done'