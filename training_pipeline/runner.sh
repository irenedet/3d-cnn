#!/usr/bin/env bash

usage()

{
    echo "usage: [[[-raw path_to_raw ] [-output output_dir]
                  [-model path_to_model] [-label label_name]
                  [-out_h5 output_h5_file_path] [-conf conf]] | [-h]]"
}


while [ "$1" != "" ]; do
    case $1 in
        -raw | --path_to_raw  )   shift
                                path_to_raw=$1
                                ;;
        -output | --output_dir )   shift
                                output_dir=$1
                                ;;
#        -model | --path_to_model )   shift
#                                path_to_model=$1
#                                ;;
        -label | --label_name )   shift
                                label_name=$1
                                ;;
#        -depth | --unet_depth )   shift
#                                unet_depth=$1
#                                ;;
#        -init_feat | --initial_features )   shift
#                                initial_features=$1
#                                ;;
        -xdim | --output_xdim )   shift
                                output_xdim=$1
                                ;;
        -ydim | --output_ydim )   shift
                                output_ydim=$1
                                ;;
        -zdim | --output_zdim )   shift
                                output_zdim=$1
                                ;;
        -box | --box_size )   shift
                                box_size=$1
                                ;;
        -number_iter | --number_iter )   shift
                                number_iter=$1
                                ;;
        -split | --split )   shift
                                split=$1
                                ;;
#        -min_peak_distance | --min_peak_distance )   shift
#                                min_peak_distance=$1
#                                ;;
#        -z_shift | --z_shift_original )   shift
#                                z_shift_original=$1
#                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


echo path_to_raw = $path_to_raw
echo path_to_labeled=$path_to_labeled
echo output_dir = $output_dir
echo label_name = $label_name
echo box = $box_size
echo shapex = $output_xdim
echo shapey = $output_ydim
echo shapez = $output_zdim
echo number_iter = $number_iter
echo split = $split  # Only augment training data

#echo path_to_model = $path_to_model
#echo depth = $unet_depth
#echo init_feat = $initial_features


#echo min_peak_distance = $min_peak_distance
#echo z_shift = $z_shift_original

#export path_to_raw="/scratch/trueba/cnn/004/4bin/cnn/rawtomogram/180426_004_4bin.hdf"
#export path_to_labeled="/scratch/trueba/cnn/004/4bin/cnn/centralregion_004.hdf"
#
#export output_dir="/scratch/trueba/3d-cnn/training_data/TEST/"
#export label_name="ribosomes"
#export box_size=128
#
#export shapez=221
#export shapey=928
#export shapex=928
## For data augmentation:
#export number_iter=5
#export split=40  # Only augment training data



export output_h5_file_name="training.h5"
export output_data_path=$output_dir$output_h5_file_name
export overlap=12
export output_data_path_aug=$output_dir'data_aug.h5'

export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn
echo PYTHONPATH=$PYTHONPATH

echo 'running python3 scripts: 0. Generating training set'
python3 /g/scb2/zaugg/trueba/3d-cnn/training_pipeline/0_create_training_set.py -raw $path_to_raw -labeled $path_to_labeled -output $output_dir -label $label_name -outh5 $output_data_path -outh5_aug $output_data_path_aug -box $box_size -shapex $output_xdim -shapey $output_ydim -shapez $output_zdim -number_iter $number_iter -split $split -overlap $overlap
echo '... done.'