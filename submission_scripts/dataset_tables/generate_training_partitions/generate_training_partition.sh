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


TOMOS="190301/001
190301/002
190301/004
190301/006
190301/007
190301/010
190301/011
190301/013
190301/014
190301/015
190301/017
190301/020
190301/021
190301/026
190301/029
190301/030
190329/001
190329/004
190329/005
190329/007
190329/010
190329/012
190329/013
190329/015
190329/017
190329/021
190329/022
190329/023
190329/025
190329/028
190329/032
190329/036"

export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table_ribo_single_class.csv"
export global_output_dir="/scratch/trueba/3d-cnn/training_data/dice-multi-class/ribo"
export write_on_table='True'
# Training set parameters:
export segmentation_names='ribo' #,fas,memb'
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
