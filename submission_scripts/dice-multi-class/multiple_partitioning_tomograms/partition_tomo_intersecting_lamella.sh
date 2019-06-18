#!/usr/bin/env bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 32G
#SBATCH --time 0-05:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
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
        -parameters_file | --parameters_file )   shift
                                parameters_file=$1
                                ;;
        -output_dir | --output_dir )   shift
                                output_dir=$1
                                ;;
        -box_side | --box_side )   shift
                                box_side=$1
                                ;;
        -overlap | --overlap )   shift
                                overlap=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


export parameters_file=$parameters_file
export box_side=$box_side
export overlap=$overlap
export output_dir=$output_dir

echo output_dir = $output_dir

source $parameters_file

export tomo_name=$tomo_name
export output_dir=$output_dir"/"$tomo_name
export hdf_lamella_file=$hdf_lamella_file
export path_to_raw=$eman2_filtered_file
export outh5=$output_dir"/tomo_partition.h5"

echo parameters_file = $parameters_file
echo output_dir = $output_dir
echo box_side = $box_side
echo overlap = $overlap
echo hdf_lamella_file = $hdf_lamella_file
echo path_to_raw = $path_to_raw
echo outh5 = $outh5

module load Anaconda3
echo 'starting virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/

export outh5=$output_dir"/tomo_partition.h5"
echo 'starting python script'
python3 ./pipelines/dice_multi-class/particle_picking_pipeline/partition_tomogram_in_intersection_with_lamella.py -raw $path_to_raw -hdf_lamella_file $hdf_lamella_file -outh5 $outh5 -output $output_dir -box $box_side -overlap $overlap
echo 'done'