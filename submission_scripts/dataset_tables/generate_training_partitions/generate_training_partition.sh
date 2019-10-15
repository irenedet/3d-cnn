#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 20G
#SBATCH --time 0-00:40
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

TOMOS="181119/002"

export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
export global_output_dir="/scratch/trueba/3d-cnn/cross-validation/training-data"
export write_on_table='false'
# Training set parameters:
export segmentation_names='ribo,fas,memb' #separated by commas
export split=0.7
export box_length=128
export number_iterations_data_aug=0
export min_label_fraction=0.002
# Data for the new model

# Data for old models for resuming training:

for tomo_name in $TOMOS
do
    export output_dir=$global_output_dir"/"$tomo_name"/multi_class/strongly_labeled_$min_label_fraction"
    mkdir -p $output_dir
    echo 'Generating training partition for dataset' $tomo_name
    python3 runners/dataset_tables/generate_training_partitions/generate_train_and_test_partitions_multi_label_files.py  -output $output_dir -tomo_name $tomo_name -dataset_table $path_to_dataset_table -split $split -segmentation_names $segmentation_names -box $box_length -number_iter $number_iterations_data_aug -write_on_table $write_on_table -min_label_fraction $min_label_fraction
    echo "... done."
done

