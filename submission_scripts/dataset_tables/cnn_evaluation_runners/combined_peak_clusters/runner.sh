#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 128G
#SBATCH --time 0-2:50
#SBATCH -o slurm_outputs/evaluate_peak_and_cluster.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/evaluate_peak_and_cluster.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

#SBAtCH -p gpu
#SBAtCH -C gpu=1080Ti
#SBAtCH --gres=gpu:1

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"

export QT_QPA_PLATFORM='offscreen'
usage()

{
    echo "usage: [[ [-output output_dir][-test_partition test_partition ]
                  [-model path_to_model] [-label label_name]
                  [-out_h5 output_h5_file_path] [-conf conf]] | [-h]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -statistics_file | --statistics_file )   shift
                                statistics_file=$1
                                ;;
        -semantic_classes | --semantic_classes )   shift
                                semantic_classes=$1
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
        -path_to_model | --path_to_model )   shift
                                path_to_model=$1
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
        -min_cluster_size | --min_cluster_size )   shift
                                min_cluster_size=$1
                                ;;
        -max_cluster_size | --max_cluster_size )   shift
                                max_cluster_size=$1
                                ;;
        -BN | --Batch_Normalization )   shift
                                Batch_Normalization=$1
                                ;;
        -cluster_size_threshold | --cluster_size_threshold )   shift
                                cluster_size_threshold=$1
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
echo BN = $Batch_Normalization
echo cluster_size_threshold = $cluster_size_threshold

export min_cluster_size=$min_cluster_size
export max_cluster_size=$max_cluster_size
export dataset_table=$dataset_table
export threshold=$threshold
export path_to_model=$path_to_model
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
export BN=$Batch_Normalization
export cluster_size_threshold=$cluster_size_threshold

export summary_file=$output_dir"/summary_statistics_class"$class_number".csv"
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

# 1. Segmentation with the trained UNet
echo 'running python3 scripts: Segmenting raw subtomograms without activation'
python3 ./runners/dataset_tables/cluster_peaks_particle_peaking/2_subtomograms_segmentation.py -model $path_to_model -label $label_name -dataset_table $dataset_table -tomo_name $tomo_name -init_feat $init_feat -depth $depth -out_classes $output_classes -new_loader $new_loader -BN $BN
echo '... done.'

# 2. Getting peaks
echo 'running python3 scripts: getting particles motive list'
python3 ./runners/dataset_tables/cluster_peaks_particle_peaking/3_get_peaks_motive_list.py -dataset_table $dataset_table -tomo_name $tomo_name -output $output_dir -label $label_name -box $box_side -class_number $class_number -min_peak_distance $minimum_peak_distance -overlap $box_overlap
echo '... done.'

export peaks_motl_path=$(ls $output_dir"/peaks/motl_"*".csv")

# 3. Getting final motive list as combination of cluster centroids and peaks:
echo 'running python3 scripts: getting particles motive list'
python3 ./runners/dataset_tables/cluster_peaks_particle_peaking/3_get_combined_centroids_peaks_motl.py -dataset_table $dataset_table -min_cluster_size $min_cluster_size -max_cluster_size $max_cluster_size -tomo_name $tomo_name -output $output_dir -label $label_name -box $box_side -class_number $class_number -particle_radius $minimum_peak_distance -overlap $box_overlap -peaks_motl_path $peaks_motl_path -cluster_size_threshold $cluster_size_threshold
echo '... done.'

# 4. Filter coordinate points with lamella mask
export combined_motl_dir=$output_dir"/combined_motl_"$cluster_size_threshold
export path_to_csv_motl=$(ls $combined_motl_dir"/combined_motl.csv")
export lamella_output_dir=$output_dir"/in_lamella"

echo "Now filtering points in lamella mask"
python3 ./runners/dataset_tables/pr_analysis/filter_with_lamella_mask.py -dataset_table $dataset_table -tomo_name $tomo_name -csv_motl $path_to_csv_motl -output_dir $output_dir -border_xy $border_xy -lamella_extension $lamella_extension
echo "...done filtering points in lamella mask."


# 5. Precision-Recall analysis
export path_to_csv_motl_in_lamella=$(ls $lamella_output_dir/motl*)
echo "Starting to generate precision recall plots"
python3 ./runners/dataset_tables/pr_analysis/precision_recall_plots.py -dataset_table $dataset_table -tomo_name $tomo_name -statistics_file $statistics_file -label_name $label_name -motl $path_to_csv_motl_in_lamella -output $lamella_output_dir -radius $same_peak_distance -box $box_side -threshold $threshold -class_number $class_number -summary_file $summary_file -semantic_classes $semantic_classes
echo "...done with precision recall plots."

