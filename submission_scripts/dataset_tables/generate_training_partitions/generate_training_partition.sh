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

#TOMOS="180713/043"
#180426/014
#180426/043
#180711/005
#180713/005
#180713/020
#180713/027"

TOMOS="180413/006
180413/007
180426/005
180426/006
180426/008
180426/014
180426/038
180426/027
180426/037
180426/040
180426/043
180711/005
180711/006
180711/007
180711/012
180711/017
180711/022
180713/002
180713/005
180713/007
180713/015
180713/018
180713/020
180713/025
180713/027
180713/031
180713/035
180713/037
180713/039
180713/041
180713/043
180713/050"

export path_to_dataset_table="/struct/mahamid/Irene/yeast/npc/npc_yeast_data.csv"
export global_output_dir="/struct/mahamid/Irene/yeast/healthy"
export write_on_table='true'
# Training set parameters:
export segmentation_names='npc' #separated by commas
export split=0.7
export box_length=128
export number_iterations_data_aug=0
export min_label_fraction=0.02
# Data for the new model

# Data for old models for resuming training:

for tomo_name in $TOMOS
do
    export output_dir=$global_output_dir"/"$tomo_name"/"$segmentation_names"_class/strongly_labeled$min_label_fraction/"
    mkdir -p $output_dir
    echo 'Generating training partition for dataset' $tomo_name
    python3 runners/dataset_tables/generate_training_partitions/generate_train_and_test_partitions_multi_label_files.py  -output $output_dir -tomo_name $tomo_name -dataset_table $path_to_dataset_table -split $split -segmentation_names $segmentation_names -box $box_length -number_iter $number_iterations_data_aug -write_on_table $write_on_table -min_label_fraction $min_label_fraction
    echo "... done."
done

