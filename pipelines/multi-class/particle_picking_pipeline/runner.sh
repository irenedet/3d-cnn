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
        -model | --path_to_model )   shift
                                path_to_model=$1
                                ;;
        -label | --label_name )   shift
                                label_name=$1
                                ;;
        -depth | --unet_depth )   shift
                                unet_depth=$1
                                ;;
        -init_feat | --initial_features )   shift
                                initial_features=$1
                                ;;
        -out_classes | --output_classes )   shift
                                output_classes=$1
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
        -class_number | --class_number )   shift
                                class_number=$1
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


echo path_to_raw = $path_to_raw
echo output_dir = $output_dir
echo path_to_model = $path_to_model
echo label_name = $label_name
echo depth = $unet_depth
echo init_feat = $initial_features
echo output_classes = $output_classes
echo xdim = $output_xdim
echo ydim = $output_ydim
echo zdim = $output_zdim
echo box = $box_side
echo class_number = $class_number
echo min_peak_distance = $min_peak_distance
echo z_shift = $z_shift_original


export box_overlap=12
export output_h5_file_path=$output_dir'partition_subtomograms_.h5'

echo 'running python3 scripts: 1. Partitioning raw tomogram'
python3 ./pipelines/multi-class/particle_picking_pipeline/1_partition_tomogram.py -raw $path_to_raw -output $output_dir -outh5 $output_h5_file_path -box $box_side -overlap $box_overlap
echo '... done.'

echo 'running python3 scripts: 2. Segmenting raw subtomograms'
python3 ./pipelines/multi-class/evaluation/cnn_subtomo_segmentation.py -model $path_to_model -label $label_name -data_path $output_h5_file_path -init_feat $initial_features -depth $unet_depth -out_classes $output_classes
echo '... done.'

echo 'running python3 scripts: 3. getting particles motive list'
python3 ./pipelines/multi-class/particle_picking_pipeline/3_get_peaks_motive_list.py -output $output_dir -label $label_name -subtomo $output_h5_file_path -box $box_side -xdim $output_xdim -ydim $output_ydim -zdim $output_zdim -class_number $class_number -min_peak_distance $min_peak_distance -z_shift $z_shift_original -overlap $box_overlap
echo 'finished whole script'


