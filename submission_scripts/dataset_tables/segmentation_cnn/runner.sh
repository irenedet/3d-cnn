#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 20G
#SBATCH --time 0-00:30
#SBATCH -o segment.slurm.%N.%j.out
#SBAtCH -e segment.slurm.%N.%j.err
#SBAtCH --mail-type=END,FAIL
#SBAtCH --mail-user=irene.de.teresa@embl.de

#SBATCH -p gpu
#SBATCH -C gpu=2080Ti
#SBATCH --gres=gpu:4 -n1 -c4

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
        -semantic_classes | --semantic_classes )   shift
                                semantic_classes=$1
                                ;;
        -dataset_table | --dataset_table )   shift
                                dataset_table=$1
                                ;;
        -tomo_name | --tomo_name )   shift
                                tomo_name=$1
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
        -decoder_dropout | --decoder_dropout )   shift
                                decoder_dropout=$1
                                ;;
        -encoder_dropout | --encoder_dropout )   shift
                                encoder_dropout=$1
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
echo output_classes = $output_classes
echo BN = $Batch_Normalization
echo semantic_classes = $semantic_classes
echo encoder_dropout = $encoder_dropout
echo decoder_dropout = $decoder_dropout

export encoder_dropout=$encoder_dropout
export decoder_dropout=$decoder_dropout
export dataset_table=$dataset_table
export path_to_model=$path_to_model
export label_name=$label_name
export depth=$depth
export init_feat=$init_feat
export box_side=$box_side
export new_loader=$new_loader
export output_classes=$output_classes
export tomo_name=$tomo_name
export BN=$Batch_Normalization
export semantic_classes=$semantic_classes


echo tomo_name = $tomo_name
export box_overlap=12


# 1. Segmenting test_partition:
echo 'running python3 scripts: Segmenting raw subtomograms'
python3 $UPICKER_PATH/runners/dataset_tables/particle_picking_scripts/2_subtomograms_segmentation_no_activation.py -model $path_to_model -label $label_name -dataset_table $dataset_table -tomo_name $tomo_name -init_feat $init_feat -depth $depth -out_classes $output_classes -new_loader $new_loader -BN $BN -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout
echo '... done.'