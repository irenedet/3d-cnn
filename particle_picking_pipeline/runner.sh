#!/usr/bin/env bash

usage()
{
    echo "usage: [[[-raw path_to_raw ] [-output output_dir]
                  [-model path_to_model]] | [-h]]"
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


echo 'running python3 scripts: 1. Partitioning raw tomogram'
export output_h5_file_name=$(python3 particle_picking_pipeline/1_partition_tomogram.py -raw $path_to_raw -output $output_dir 2>&1)
echo '... done.'

echo 'running python3 scripts: 2. Segmenting raw subtomograms'
python3 particle_picking_pipeline/2_subtomograms_segmentation.py -output $output_h5_file_name -model $path_to_model -label $label_name
echo '... done.'

echo 'running python3 scripts: 3. getting particles motive list'
python3 particle_picking_pipeline/3_get_peaks_motive_list.py -output $output_dir -label $label_name -subtomo $output_h5_file_name 2>&1
echo 'finished whole script'
