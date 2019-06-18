#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 32G
#SBATCH --time 0-0:20
#SBATCH -o slurm_outputs/subtomo2dataset.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/subtomo2dataset.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
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
        -box_side | --box_side )   shift
                                box_side=$1
                                ;;
        -overlap | --overlap )   shift
                                overlap=$1
                                ;;
        -output_classes | --output_classes )   shift
                                output_classes=$1
                                ;;
        -label_name | --label_name )   shift
                                label_name=$1
                                ;;
        -segmentation_names | --segmentation_names )   shift
                                segmentation_names=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


# parameters_file: subtomos_path, output_shape
# output_dir
# box
# overlap_thickness
# output_classes
# label_name
# segmentation_names

export box_overlap=$overlap
export label_name=$label_name
export box_side=$box_side
export output_classes=$output_classes
export output_dir=$output_dir
export segmentation_names

# Reading parameters file and translating to current script variables:
source $parameters_file
export tomo_name=$tomo_name
export output_dir=$output_dir"/"$tomo_name
export subtomos_path=$test_partition
export output_xdim=$input_xdim
export output_ydim=$input_ydim
export output_zdim=$input_zdim

echo tomo_name = $tomo_name
echo output_dir = $output_dir


echo "Starting python script..."
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/dice_multi-class/evaluation/subtomos2dataset.py -subtomos_path $subtomos_path -output_xdim $output_xdim -output_ydim $output_ydim -output_zdim $output_zdim -output_dir $output_dir -box $box_side -overlap_thickness $overlap -output_classes $output_classes -label_name $label_name -segmentation_names $segmentation_names
echo "... done."

