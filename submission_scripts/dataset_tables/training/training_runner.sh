#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 160G
#SBATCH --time 1-16:00
#SBATCH -o slurm_outputs/training.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/training.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de
#SBATCH -p gpu
#SBATCH -C gpu=2080Ti
#SBATCH --gres=gpu:4 -n1 -c4

#-p is partition
#-C is the card
#--gres means generic resources per node (gpus per node)
# in format name[:type:count]
#--gpus-per-socket
#--gpus-per-task

module load GCC
module load Anaconda3
echo 'activating virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo '... done.'

export QT_QPA_PLATFORM='offscreen'
usage()

{
    echo "usage: [[ [-output output_dir][-test_partition test_partition ]
                  [-model path_to_model] [-label label_name]
                  [-out_h5 output_h5_file_path] [-conf conf]] | [-h]]"
}


while [ "$1" != "" ]; do
    case $1 in
        -tomo_training_list | --tomo_training_list )   shift
                                tomo_training_list=$1
                                ;;
        -path_to_dataset_table | --path_to_dataset_table )   shift
                                path_to_dataset_table=$1
                                ;;
        -segmentation_names | --segmentation_names )   shift
                                segmentation_names=$1
                                ;;
        -split | --split )   shift
                                split=$1
                                ;;
        -shuffle | --shuffle )   shift
                                shuffle=$1
                                ;;
        -log_dir | --log_dir )   shift
                                log_dir=$1
                                ;;
        -model_initial_name | --model_initial_name )   shift
                                model_initial_name=$1
                                ;;
        -model_path | --model_path )   shift
                                model_path=$1
                                ;;
        -n_epochs | --n_epochs )   shift
                                n_epochs=$1
                                ;;
        -depth | --depth )   shift
                                depth=$1
                                ;;
        -initial_features | --initial_features )   shift
                                initial_features=$1
                                ;;
        -output_classes | --output_classes )   shift
                                output_classes=$1
                                ;;
        -retrain | --retrain )   shift
                                retrain=$1
                                ;;
        -path_to_old_model | --path_to_old_model )   shift
                                path_to_old_model=$1
                                ;;
        -fraction | --fraction )   shift
                                fraction=$1
                                ;;
        -models_notebook | --models_notebook )   shift
                                models_notebook=$1
                                ;;
        -BN | --Batch_Normalization )   shift
                                Batch_Normalization=$1
                                ;;
        -encoder_dropout | --encoder_dropout )   shift
                                encoder_dropout=$1
                                ;;
        -decoder_dropout | --decoder_dropout )   shift
                                decoder_dropout=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


export tomo_training_list=$tomo_training_list
export path_to_dataset_table=$path_to_dataset_table
export segmentation_names=$segmentation_names
export split=$split
export shuffle=$shuffle
# Data for the new model
export log_dir=$log_dir
export model_initial_name=$model_initial_name
export model_path=$model_path
export n_epochs=$n_epochs
export depth=$depth
export initial_features=$initial_features
export output_classes=$output_classes
export Batch_Normalization=$Batch_Normalization
# Data for old models for resuming training:
export retrain=$retrain
export path_to_old_model=$path_to_old_model
export models_notebook=$models_notebook
export encoder_dropout=$encoder_dropout
export decoder_dropout=$decoder_dropout

echo tomo_training_list=$tomo_training_list
echo path_to_dataset_table=$path_to_dataset_table
echo segmentation_names=$segmentation_names
echo split=$split
echo shuffle=$shuffle

echo log_dir=$log_dir
echo model_initial_name=$model_initial_name
echo model_path=$model_path
echo n_epochs=$n_epochs
echo depth=$depth
echo initial_features=$initial_features
echo output_classes=$output_classes

echo retrain=$retrain
echo path_to_old_model=$path_to_old_model
echo models_notebook=$models_notebook

echo 'Training dice multi-label network for fraction='$fraction
echo 'and model_name='$model_initial_name
echo UPICKER_PATH=$UPICKER_PATH
python3 $UPICKER_PATH/runners/dataset_tables/training/dice_unet_training.py -dataset_table $path_to_dataset_table -tomo_training_list "${tomo_training_list[@]}" -split $split -classes $output_classes -log_dir $log_dir -model_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -models_notebook $models_notebook -shuffle $shuffle -BN $Batch_Normalization -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout


