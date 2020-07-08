#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 20G
#SBATCH --time 0-01:30
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

#module load GCC
#module load Anaconda3
#echo 'activating virtual environment'
#source activate $UPICKER_VENV_PATH
#echo '... done.'

export QT_QPA_PLATFORM='offscreen'

TOMOS="180426/004
180426/005
180426/021
180426/024"


export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
export global_output_dir="/struct/mahamid/Irene/test-3d-unet/out/training_data"
export write_on_table='false'
## Training set parameters:
export segmentation_names='memb' #separated by commas
export split=0.7
export box_length=64
export number_iterations_data_aug=0
export min_label_fraction=0.002
export max_label_fraction=1
export processing_tomo="alex_filter"
# Data for the new model

# Data for old models for resuming training:

for tomo_name in $TOMOS
do
    export output_dir=$global_output_dir"/"$tomo_name
    mkdir -p $output_dir
    echo 'Generating training partition for dataset' $tomo_name
    python3 $UPICKER_PATH/runners/dataset_tables/generate_training_partitions/generate_train_and_test_partitions_multi_label_files.py  -output $output_dir -tomo_name $tomo_name -dataset_table $path_to_dataset_table -split $split -segmentation_names $segmentation_names -box $box_length -number_iter $number_iterations_data_aug -write_on_table $write_on_table -min_label_fraction $min_label_fraction -max_label_fraction $max_label_fraction -processing_tomo $processing_tomo
    echo "... done."
done
#
