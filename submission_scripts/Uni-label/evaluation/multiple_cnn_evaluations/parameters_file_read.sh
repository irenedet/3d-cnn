#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 32G
#SBATCH --time 0-01:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
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
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


echo path_to_model = $path_to_model
echo label_name = $label_name
echo depth = $depth
echo init_feat = $init_feat
echo box_side = $box_side
echo new_loader = $new_loader
echo minimum_peak_distance = $minimum_peak_distance
echo border_xy = $border_xy
echo lamella_extension = $lamella_extension
echo same_peak_distance = $same_peak_distance

source $parameters_file
export output_dir=$output_dir
export summary_file=$output_dir"/summary_analysis.txt"
touch $summary_file

export output_dir=$output_dir"/"$tomo_name
export test_partition=$test_partition
export input_xdim=$input_xdim
export input_ydim=$input_ydim
export input_zdim=$input_zdim
export z_shift=$z_shift
export x_shift=$x_shift
export hdf_lamella_file=$hdf_lamella_file
export path_to_motl_clean=$path_to_motl_clean

echo output_dir = $output_dir
echo test_partition = $test_partition
echo input_xdim = $input_xdim
echo input_ydim = $input_ydim
echo input_zdim = $input_zdim
echo z_shift = $z_shift
echo x_shift = $x_shift
echo hdf_lamella_file = $hdf_lamella_file
echo path_to_motl_clean = $path_to_motl_clean



# 1. Segmenting, peak calling and motl writing
echo "Calling particle picking pipeline"
bash /g/scb2/zaugg/trueba/3d-cnn/particle_picking_pipeline/runner_in_partirtion_set.sh -test_partition $test_partition -output $output_dir -model $path_to_model -label $label_name -init_feat $init_feat -depth $depth -box $box_side -xdim $input_xdim -ydim $input_ydim -zdim $input_zdim -min_peak_distance $minimum_peak_distance -z_shift $z_shift -new_loader $new_loader
echo "... done with particle picking pipeline."


# 2. Mask coordinate points with lamella mask
export path_to_csv_motl=$(ls $output_dir/motl*)

echo "Now filtering points in lamella mask"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/performance_nnet_quantification/filter_with_lamella_mask.py -csv_motl $path_to_csv_motl -lamella_file $hdf_lamella_file -output_dir $output_dir -border_xy $border_xy -lamella_extension $lamella_extension -x_dim $input_xdim -y_dim $input_ydim -z_dim $input_zdim -z_shift $z_shift
echo "...done filtering points in lamella mask."


# 3. Precision-Recall analysis
export lamella_output_dir=$output_dir"/in_lamella"
export path_to_csv_motl=$(ls $lamella_output_dir/motl*)


echo "Starting to generate precision recall plots"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/performance_nnet_quantification/precision_recall_plots.py -motl $path_to_csv_motl -clean $path_to_motl_clean -output $lamella_output_dir -test_file $test_partition -radius $same_peak_distance -shape_x $input_xdim -shape_y $input_ydim -shape_z $input_zdim -x_shift $x_shift -z_shift $z_shift -box $box_side >> $summary_file
echo "...done with precision recall plots."

