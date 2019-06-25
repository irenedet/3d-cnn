#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 128G
#SBATCH --time 0-0:50
#SBATCH -o slurm_outputs/partition_training.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/partition_training.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load GCC
module load Anaconda3
echo 'activating virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo '... done.'


TOMOS="190301/005"

export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
export global_output_dir="/scratch/trueba/3d-cnn/training_data/dice-multi-class"
export write_on_table='True'
# Training set parameters:
export segmentation_names='ribo,fas,memb'
export split=0.7
export box_length=128
export number_iterations_data_aug=6
# Data for the new model

# Data for old models for resuming training:

for tomo_name in $TOMOS
do
    export output_dir=$global_output_dir"/"$tomo_name
    echo 'Generating training partition for dataset' $tomo_name
    python3 ./runners/dataset_tables/generate_training_partitions/generate_train_and_test_partitions_multi_label_files.py  -output $output_dir -tomo_name $tomo_name -dataset_table $path_to_dataset_table -split $split -segmentation_names $segmentation_names -box $box_length -number_iter $number_iterations_data_aug -write_on_table $write_on_table
    echo "... done."
done

echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $global_output_dir"/SCRIPT_training_set.txt"
