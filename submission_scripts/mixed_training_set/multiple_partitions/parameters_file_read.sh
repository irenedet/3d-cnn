#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 64G
#SBATCH --time 0-0:50
#SBATCH -o slurm_outputs/mixed_partition.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/mixed_partition.slurm.%N.%j.err
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

#-parameters_file
# -output_dir
# -partition_dir_name
# -label_name
# -classes
# -split
# -box_side
# -number_iter
# -overlap
#-raw
# -mask_0
# -mask_1
# -mask_2

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
        -mask_1 | --mask_1 )   shift
                                mask_1=$1
                                ;;
        -mask_2 | --mask_2 )   shift
                                mask_2=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


#export parameters_file=$parameters_file
#export label_name=$label_name
#export output_dir=$output_dir
#export box_side=$box_side
#export partition_dir_name=$partition_dir_name
#export box_side=$box_side
#export split=$split
#export number_iter=$number_iter
#export path_to_raw=$path_to_raw
#export mask_0=$mask_0
#export mask_1=$mask_1
#export mask_2=$mask_2

echo parameters_file = $parameters_file
echo label_name = $label_name
echo output_dir = $output_dir
echo box_side = $box_side
echo partition_dir_name = $partition_dir_name
echo box_side = $box_side
echo split = $split
echo number_iter = $number_iter
echo path_to_raw = $path_to_raw
echo mask_0 = $mask_0
echo mask_1 = $mask_1
echo mask_2 = $mask_2

source $parameters_file

export tomo_name=$tomo_name
export output_dir=$output_dir"/"$tomo_name"/"$partition_dir_name
echo output_dir = $output_dir
export shape_x=$input_xdim
export shape_y=$input_ydim
export shape_z=$input_zdim

echo 'starting python script'
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/dice_multi-class/generate_training_set/generate_train_and_test_partitions_multi_label_files.py -raw $path_to_raw  -label_0 $mask_0 -label_1 $mask_1 -label_2 $mask_2 -output $output_dir -box $box_side -shapex $shape_x -shapey $shape_y -shapez $shape_z -number_iter $number_iter -split $split -overlap $overlap
echo 'done'