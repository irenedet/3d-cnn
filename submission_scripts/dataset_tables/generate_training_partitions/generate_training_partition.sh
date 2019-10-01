#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 35G
#SBATCH --time 0-1:50
#SBATCH -o slurm_outputs/generate_training_partition.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/generate_training_partition.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load GCC
module load Anaconda3
echo 'activating virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo '... done.'

export PYTHONPATH=/g/scb2/zaugg/trueba/3d-cnn


TOMOS="181119/002
181119/030
181126/002
181126/012
181126/025"
#190301/009
#190301/012
#190301/016
#190301/022
#190301/028
#190301/032
#190301/035
#190301/037
#190301/043
#190301/045"

export path_to_dataset_table="/struct/mahamid/Irene/yeast/fas_yeast_table.csv"
export global_output_dir="/struct/mahamid/Irene/yeast/ED"
export write_on_table='True'
# Training set parameters:
export segmentation_names='fas'
export split=0.7
export box_length=128
export number_iterations_data_aug=0
# Data for the new model

# Data for old models for resuming training:

for tomo_name in $TOMOS
do
    export output_dir=$global_output_dir"/"$tomo_name"/fas_class/"
    mkdir -p $output_dir
    echo 'Generating training partition for dataset' $tomo_name
    python3 ./runners/dataset_tables/generate_training_partitions/generate_train_and_test_partitions_multi_label_files.py  -output $output_dir -tomo_name $tomo_name -dataset_table $path_to_dataset_table -split $split -segmentation_names $segmentation_names -box $box_length -number_iter $number_iterations_data_aug -write_on_table $write_on_table
    echo "... done."
done

