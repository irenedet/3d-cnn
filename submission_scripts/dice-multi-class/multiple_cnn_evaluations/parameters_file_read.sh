#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 128G
#SBATCH --time 0-2:00
#SBATCH -o slurm_outputs/cnn_evaluation.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/cnn_evaluation.slurm.%N.%j.err
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
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


echo border_xy = $border_xy
echo lamella_extension = $lamella_extension
echo same_peak_distance = $same_peak_distance
echo class_number = $class_number
echo output_classes = $output_classes

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

source $parameters_file
export output_dir=$output_dir
export summary_file=$output_dir"/summary_analysis_class"$class_number".txt"
touch $summary_file

export tomo_name=$tomo_name
export output_dir=$output_dir"/"$tomo_name"/class_"$class_number
export test_partition=$test_partition
export input_xdim=$input_xdim
export input_ydim=$input_ydim
export input_zdim=$input_zdim
export z_shift=$z_shift
export x_shift=$x_shift
export hdf_lamella_file=$hdf_lamella_file
export path_to_motl_clean_0=$path_to_motl_clean_0
export path_to_motl_clean_1=$path_to_motl_clean_1

echo tomo_name = $tomo_name
echo output_dir = $output_dir
echo z_shift = $z_shift
echo x_shift = $x_shift
echo hdf_lamella_file = $hdf_lamella_file
echo path_to_motl_clean_0=$path_to_motl_clean_0
echo path_to_motl_clean_1=$path_to_motl_clean_1


if [ $class_number == 0 ]; then
    echo "class_number is 0"
    export path_to_motl_clean=$path_to_motl_clean_0
elif [ $class_number == 1 ]; then
    echo "class_number is 1"
    export path_to_motl_clean=$path_to_motl_clean_1
else
    echo "class_number non-supported for now"
fi


echo path_to_motl_clean = $path_to_motl_clean
export box_overlap=12


# 1. Segmenting, peak calling and motl writing
echo "Calling particle picking pipeline"
bash ./pipelines/dice_multi-class/particle_picking_pipeline/runner_in_partition_set.sh -test_partition $test_partition -output $output_dir -model $path_to_model -label $label_name -init_feat $init_feat -depth $depth -box $box_side -xdim $input_xdim -ydim $input_ydim -zdim $input_zdim -min_peak_distance $minimum_peak_distance -z_shift $z_shift -class_number $class_number -out_classes $output_classes -new_loader $new_loader
echo "... done with particle picking pipeline."


# 2. Mask coordinate points with lamella mask
export path_to_csv_motl=$(ls $output_dir/motl*)


if [ $hdf_lamella_file == "None" ]; then
    echo "No lamella mask available"
    export lamella_output_dir=$output_dir
else
    echo "Lamella mask available"
    export lamella_output_dir=$output_dir"/in_lamella"
    echo "Now filtering points in lamella mask"
    python3 ./pipelines/performance_nnet_quantification/filter_with_lamella_mask.py -csv_motl $path_to_csv_motl -lamella_file $hdf_lamella_file -output_dir $output_dir -border_xy $border_xy -lamella_extension $lamella_extension -x_dim $input_xdim -y_dim $input_ydim -z_dim $input_zdim -z_shift $z_shift
    echo "...done filtering points in lamella mask."
fi

# 3. Precision-Recall analysis
export path_to_csv_motl_in_lamella=$(ls $lamella_output_dir/motl*)
echo "Starting to generate precision recall plots"
python3 ./pipelines/performance_nnet_quantification/precision_recall_plots.py -motl $path_to_csv_motl_in_lamella -clean $path_to_motl_clean -output $lamella_output_dir -test_file $test_partition -radius $same_peak_distance -shape_x $input_xdim -shape_y $input_ydim -shape_z $input_zdim -x_shift $x_shift -z_shift $z_shift -box $box_side -threshold $threshold >> $summary_file
echo "...done with precision recall plots."

