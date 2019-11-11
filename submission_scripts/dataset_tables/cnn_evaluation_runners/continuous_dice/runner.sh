#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 20G
#SBATCH --time 0-04:35
#SBATCH -o slurm_outputs/evaluate_particle_peaking_peaks.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/evaluate_particle_peaking_peaks.slurm.%N.%j.err
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
        -depth | --depth )   shift
                                depth=$1
                                ;;
        -init_feat | --init_feat )   shift
                                init_feat=$1
                                ;;
        -new_loader | --new_loader )   shift
                                new_loader=$1
                                ;;
        -BN | --Batch_Normalization )   shift
                                Batch_Normalization=$1
                                ;;
        -test_partition_from_table | --test_partition_from_table )   shift
                                test_partition_from_table=$1
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
echo semantic_classes = $semantic_classes
echo test_partition_from_table=$test_partition_from_table

export test_partition_from_table=$test_partition_from_table
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
export semantic_classes=$semantic_classes
export statistics_file=$statistics_file

export output_dir=$output_dir/$label_name/$tomo_name/"class_"$class_number
mkdir -p $output_dir

echo tomo_name = $tomo_name
echo output_dir = $output_dir


echo "class_number is " $class_number

export box_overlap=12
# 1. Segmenting test_partition:
echo 'running python3 scripts: Segmenting raw subtomograms'
python3 runners/dataset_tables/particle_picking_scripts/2_subtomograms_segmentation_no_activation.py -model $path_to_model -label $label_name -dataset_table $dataset_table -tomo_name $tomo_name -init_feat $init_feat -depth $depth -out_classes $output_classes -new_loader $new_loader -BN $BN
echo '... done.'

# 2. Assemble together the full prediction dataset:
echo "Assembling prediction dataset for "$tomo_name
export cluster_labels=False
export reconstruction_type="prediction"
export prediction_path=$output_dir"/prediction.hdf"
python3 runners/dataset_tables/subtomos2datasets/subtomos2dataset.py -dataset_table $dataset_table -tomo_name $tomo_name -class_number $class_number -output_path $prediction_path -box_length $box_side -overlap $box_overlap -label_name $label_name -cluster_labels $cluster_labels -reconstruction_type $reconstruction_type
echo "... done."

# 3. Dice loss computation
echo "Computing our prediction's Dice coefficient"
python3 runners/dataset_tables/dice_loss_evaluation/dice_evaluation.py -dataset_table $dataset_table -tomo_name $tomo_name -class_number $class_number -label_name $label_name -statistics_file $statistics_file -semantic_classes $semantic_classes -prediction_path $prediction_path
echo "... done"