#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 64G
#SBATCH --time 0-0:50
#SBATCH -o mixed_partition.slurm.%N.%j.out
#SBAtCH -e mixed_partition.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
echo "... done"

export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn
export QT_QPA_PLATFORM='offscreen'
usage()

{
    echo "usage: [[ [-output output_dir][-test_partition test_partition ]
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
        -partition_dir_name | --partition_dir_name )   shift
                                partition_dir_name=$1
                                ;;
        -label_name | --label_name )   shift
                                label_name=$1 #check
                                ;;
        -box_side | --box_side )   shift
                                box_side=$1
                                ;;
        -split | --split )   shift
                                split=$1
                                ;;
        -number_iter | --number_iter )   shift
                                number_iter=$1
                                ;;
        -overlap | --overlap )   shift
                                overlap=$1
                                ;;
        -raw | --path_to_raw )   shift
                                path_to_raw=$1
                                ;;
        -mask_0 | --mask_0 )   shift
                                mask_0=$1
                                ;;
        -tomo_name | --tomo_name )   shift
                                tomo_name=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


echo parameters_file = $parameters_file
echo label_name = $label_name
echo output_dir = $output_dir
echo box_side = $box_side
echo partition_dir_name = $partition_dir_name
echo split = $split
echo number_iter = $number_iter
echo path_to_raw = $path_to_raw
echo mask_0 = $mask_0

source $parameters_file

export tomo_name=$tomo_name
export output_dir=$output_dir"/"$tomo_name"/"$partition_dir_name
echo output_dir = $output_dir
export shape_x=$input_xdim
export shape_y=$input_ydim
export shape_z=$input_zdim

echo 'starting python script'
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/dice_multi-class/generate_training_set/generate_train_and_test_partitions_single_label_file.py -raw $path_to_raw  -labeled $mask_0 -output $output_dir -box $box_side -shapex $shape_x -shapey $shape_y -shapez $shape_z -number_iter $number_iter -split $split -overlap $overlap -label $label_name
echo 'done'