module load Anaconda3
echo "activating virtual environment"
source activate $UPICKER_VENV_PATH
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
        -semantic_classes | --semantic_classes )   shift
                                semantic_classes=$1
                                ;;
        -dataset_table | --dataset_table )   shift
                                dataset_table=$1
                                ;;
        -tomo_name | --tomo_name )   shift
                                tomo_name=$1
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
        -encoder_dropout | --encoder_dropout )   shift
                                encoder_dropout=$1
                                ;;
        -decoder_dropout | --decoder_dropout )   shift
                                decoder_dropout=$1
                                ;;
        -new_loader | --new_loader )   shift
                                new_loader=$1
                                ;;
        -BN | --Batch_Normalization )   shift
                                Batch_Normalization=$1
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
echo class_number = $class_number
echo output_classes = $output_classes
echo BN = $Batch_Normalization
echo semantic_classes = $semantic_classes

export encoder_dropout=$encoder_dropout
export decoder_dropout=$decoder_dropout
#export min_cluster_size=$min_cluster_size
#export max_cluster_size=$max_cluster_size
export dataset_table=$dataset_table
#export threshold=$threshold
export path_to_model=$path_to_model
export label_name=$label_name
export depth=$depth
export init_feat=$init_feat
export box_side=$box_side
export new_loader=$new_loader
#export minimum_peak_distance=$minimum_peak_distance
#export border_xy=$border_xy
#export lamella_extension=$lamella_extension
#export same_peak_distance=$same_peak_distance
export class_number=$class_number
export output_classes=$output_classes
export tomo_name=$tomo_name
export output_dir=$output_dir
export BN=$Batch_Normalization
export semantic_classes=$semantic_classes

#export output_dir=$output_dir/$label_name/$tomo_name/"class_"$class_number

echo tomo_name = $tomo_name
echo output_dir = $output_dir


echo "class_number is " $class_number

export box_overlap=12


echo 'running python3 scripts: Segmenting raw subtomograms'
python3 $UPICKER_PATH/runners/dataset_tables/particle_picking_scripts/2_subtomograms_segmentation_no_activation.py -model $path_to_model -label $label_name -dataset_table $dataset_table -tomo_name $tomo_name -init_feat $init_feat -depth $depth -out_classes $output_classes -new_loader $new_loader -BN $BN -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout
echo '... done.'