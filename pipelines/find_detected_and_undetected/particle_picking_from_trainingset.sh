#!/usr/bin/env bash

usage()

{
    echo "usage: [[[-raw path_to_raw ] [-output output_dir]
                  [-model path_to_model] [-label label_name]
                  [-out_h5 output_h5_file_path] [-conf conf]] | [-h]]"
}


while [ "$1" != "" ]; do
    case $1 in
        -output | --output_dir )   shift
                                output_dir=$1
                                ;;
        -training_file | --training_file )   shift
                                training_file=$1
                                ;;
        -model | --path_to_model )   shift
                                path_to_model=$1
                                ;;
        -model_nickname | --model_nickname )   shift
                                model_nickname=$1
                                ;;
        -depth | --unet_depth )   shift
                                unet_depth=$1
                                ;;
        -init_feat | --initial_features )   shift
                                initial_features=$1
                                ;;
        -xdim | --output_zdim )   shift
                                output_xdim=$1
                                ;;
        -ydim | --output_zdim )   shift
                                output_ydim=$1
                                ;;
        -zdim | --output_zdim )   shift
                                output_zdim=$1
                                ;;
        -box | --box_side )   shift
                                box_side=$1
                                ;;
        -min_peak_distance | --min_peak_distance )   shift
                                min_peak_distance=$1
                                ;;
        -z_shift | --z_shift_original )   shift
                                z_shift_original=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


echo training_dir = $training_dir
echo output_dir = $output_dir
echo path_to_model = $path_to_model
echo model_nickname = $model_nickname
echo depth = $unet_depth
echo init_feat = $initial_features
echo xdim = $output_xdim
echo ydim = $output_ydim
echo zdim = $output_zdim
echo box = $box_side
echo min_peak_distance = $min_peak_distance
echo z_shift = $z_shift_original


export box_overlap=12
export training_file=$training_file

echo 'running 2_subtomograms_segmentation.py'
python3 particle_picking_pipeline/2_subtomograms_segmentation.py -model $path_to_model -label $model_nickname -outh5 $training_file -init_feat $initial_features -depth $unet_depth
echo '... done.'

echo 'running 3_get_peaks_motive_list.py'
python3 particle_picking_pipeline/3_get_peaks_motive_list.py -output $output_dir -label $model_nickname -subtomo $training_file -box $box_side -xdim $output_xdim -ydim $output_ydim -zdim $output_zdim -min_peak_distance $min_peak_distance -z_shift $z_shift_original -overlap $box_overlap
echo 'finished whole script'
