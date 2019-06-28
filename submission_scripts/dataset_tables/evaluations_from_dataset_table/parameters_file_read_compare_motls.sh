#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --time 0-00:10
#SBATCH -o slurm_outputs/cnn_evaluation.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/cnn_evaluation.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"

export QT_QPA_PLATFORM='offscreen'
usage()

{
    echo "usage: [[ [-output output_dir][-test_partition test_partition ]
                   [-label label_name]
                  [-out_h5 output_h5_file_path] [-conf conf]] | [-h]]"
}


while [ "$1" != "" ]; do
    case $1 in
        -statistics_file | --statistics_file )   shift
                                statistics_file=$1
                                ;;
        -dataset_table | --dataset_table )   shift
                                dataset_table=$1
                                ;;
        -tomo_name | --tomo_name )   shift
                                tomo_name=$1
                                ;;
        -output_dir | --output_dir )   shift
                                output_dir=$1
                                ;;
        -class_number | --class_number )   shift
                                class_number=$1
                                ;;
        -output_classes | --output_classes )   shift
                                output_classes=$1
                                ;;
        -label_name | --label_name )   shift
                                label_name=$1
                                ;;
        -box_side | --box_side )   shift
                                box_side=$1
                                ;;
        -min_peak_distance | --min_peak_distance )   shift
                                min_peak_distance=$1
                                ;;
        -depth | --depth )   shift
                                depth=$1
                                ;;
        -init_feat | --init_feat )   shift
                                init_feat=$1
                                ;;
        -new_loader | --new_loader )   shift
                                new_loader=$1
                                ;;
        -minimum_peak_distance | --minimum_peak_distance )   shift
                                minimum_peak_distance=$1
                                ;;
        -border_xy | --border_xy )   shift
                                border_xy=$1
                                ;;
        -lamella_extension | --lamella_extension )   shift
                                lamella_extension=$1
                                ;;
        -same_peak_distance | --same_peak_distance )   shift
                                same_peak_distance=$1
                                ;;
        -threshold | --threshold )   shift
                                threshold=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done

echo dataset_table = $dataset_table
echo border_xy = $border_xy
echo lamella_extension = $lamella_extension
echo same_peak_distance = $same_peak_distance
echo class_number = $class_number
echo output_classes = $output_classes

export dataset_table=$dataset_table
export threshold=$threshold
export label_name=$label_name
export depth=$depth
export init_feat=$init_feat
export box_side=$box_side
export new_loader=$new_loader
export minimum_peak_distance=$minimum_peak_distance
export border_xy=$border_xy
export lamella_extension=$lamella_extension
export same_peak_distance=$same_peak_distance
export class_number=$class_number
export output_classes=$output_classes
export tomo_name=$tomo_name
export output_dir=$output_dir

export summary_file=$output_dir"/summary_analysis_class"$class_number".txt"
touch $summary_file
export output_dir=$output_dir"/"$tomo_name"/class_"$class_number

echo tomo_name = $tomo_name
echo output_dir = $output_dir


if [ $class_number == 0 ]; then
    echo "class_number is 0"
    export class_name="ribo"
elif [ $class_number == 1 ]; then
    echo "class_number is 1"
    export class_name="fas"
else
    echo "class_number non-supported for now"
fi



export box_overlap=12

# 3. Filter coordinate points with lamella mask
export lamella_output_dir=$output_dir"/in_lamella"

# 3. Precision-Recall analysis
export path_to_csv_motl_in_lamella=$(ls $lamella_output_dir/motl*)
echo "Starting to generate precision recall plots"
python3 ./runners/dataset_tables/pr_analysis/precision_recall_plots.py -dataset_table $dataset_table -tomo_name $tomo_name -statistics_file $statistics_file -label_name $label_name -motl $path_to_csv_motl_in_lamella -output $lamella_output_dir -radius $same_peak_distance -box $box_side -threshold $threshold -class_name $class_name >> $summary_file
echo "...done with precision recall plots."

#!/usr/bin/env bash