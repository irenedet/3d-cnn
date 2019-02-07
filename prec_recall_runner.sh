#!/usr/bin/env bash


usage()
{
    echo "usage: [[[-raw path_to_motl ] [-output output_dir]] | [-h]]"
}


while [ "$1" != "" ]; do
    case $1 in
        -output | --output_dir )   shift
                                output_dir=$1
                                ;;
        -motl | --path_to_motl )   shift
                                path_to_motl=$1
                                ;;
        -clean | --path_to_clean )   shift
                                path_to_clean=$1
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

echo path_to_motl = $path_to_motl
echo output_dir = $output_dir
echo path_to_clean = $path_to_clean
echo label_name = $label_name

echo 'running precision-recall evaluation'
python3 runners/prec_recall_analysis.py -output $output_dir -motl $path_to_motl -clean $path_to_clean -label $label_name
echo 'finished!'