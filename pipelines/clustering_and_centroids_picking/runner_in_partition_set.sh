#!/usr/bin/env bash

#usage()
#
#{
#    echo "usage: [[[-raw path_to_raw ] [-output output_dir]
#                  [-model path_to_model] [-label label_name]
#                  [-out_h5 output_h5_file_path] [-conf conf]] | [-h]]"
#}
#
#
#while [ "$1" != "" ]; do
#    case $1 in
#        -test_partition | --test_partition  )   shift
#                                test_partition=$1
#                                ;;
#        -output_dir | --output_dir )   shift
#                                output_dir=$1
#                                ;;
#        -model | --path_to_model )   shift
#                                path_to_model=$1
#                                ;;
#        -label_name | --label_name )   shift
#                                label_name=$1
#                                ;;
#        -depth | --unet_depth )   shift
#                                unet_depth=$1
#                                ;;
#        -init_feat | --initial_features )   shift
#                                initial_features=$1
#                                ;;
#        -out_classes | --output_classes )   shift
#                                output_classes=$1
#                                ;;
#        -xdim | --xdim )   shift
#                                xdim=$1
#                                ;;
#        -ydim | --ydim )   shift
#                                ydim=$1
#                                ;;
#        -zdim | --zdim )   shift
#                                zdim=$1
#                                ;;
#        -min_cluster_size | --min_cluster_size )   shift
#                                min_cluster_size=$1
#                                ;;
#        -max_cluster_size | --max_cluster_size )   shift
#                                max_cluster_size=$1
#                                ;;
#        -class_number | --class_number )   shift
#                                class_number=$1
#                                ;;
#        -min_peak_distance | --min_peak_distance )   shift
#                                min_peak_distance=$1
#                                ;;
#        -new_loader | --new_loader )   shift
#                                new_loader=$1
#                                ;;
#        -h | --help )           usage
#                                exit
#                                ;;
#        * )                     usage
#                                exit 1
#    esac
#    shift
#done
#
#
#
#echo label_name=$label_name
#echo output_dir=$output_dir
#echo min_cluster_size=$min_cluster_size
#echo max_cluster_size=$max_cluster_size
#echo test_partition=$test_partition
#echo path_to_model=$path_to_model
#echo output_classes=$output_classes
#echo xdim=$xdim
#echo ydim=$ydim
#echo zdim=$zdim
#echo class_number=$class_number
#echo min_peak_distance=$min_peak_distance
#echo new_loader=$new_loader
#echo initial_features=$initial_features
#echo unet_depth=$unet_depth

module load Anaconda3
export QT_QPA_PLATFORM='offscreen'

# To be modified by user
echo 'starting virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/

export label_name="NO_DA_ribo_D_2_IF_8"
export output_dir="/g/scb2/zaugg/trueba/3d-cnn/"
export min_cluster_size=1500
export max_cluster_size=4000
export test_partition="/scratch/trueba/liang_data/192/train_and_test_partitions/full_partition.h5"
export path_to_model="/g/scb2/zaugg/trueba/3d-cnn/models/lang_unets/NO_DA_ribo_D_2_IF_8.pkl"
export output_classes=1
export xdim=928
export ydim=928
export zdim=450
export class_number=0
export min_peak_distance=25
export new_loader=True
export initial_features=8
export unet_depth=2
export box_overlap=12


echo 'running python3 scripts: Segmenting raw subtomograms'
python3 ./pipelines/clustering_and_centroids_picking/2_subtomograms_segmentation.py -model $path_to_model -label $label_name -data_path $test_partition -init_feat $initial_features -depth $unet_depth -out_classes $output_classes -new_loader $new_loader
echo '... done.'

echo 'running python3 scripts: getting particles motive list'
python3 ./pipelines/clustering_and_centroids_picking/3_get_cluster_centroids_motl.py -output_dir $output_dir -label_name $label_name -partition $test_partition -min_cluster_size $min_cluster_size -max_cluster_size $max_cluster_size -xdim $xdim -ydim $ydim -zdim $zdim -class_number $class_number -particle_radius $min_peak_distance -overlap $box_overlap
echo 'finished whole script'


