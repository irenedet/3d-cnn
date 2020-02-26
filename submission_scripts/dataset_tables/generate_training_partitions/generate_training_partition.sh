#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 20G
#SBATCH --time 0-01:30
#SBATCH -o generate_training_partition.slurm.%N.%j.out
#SBAtCH -e generate_training_partition.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load GCC
module load Anaconda3
echo 'activating virtual environment'
source activate $UPICKER_VENV_PATH
echo '... done.'

export QT_QPA_PLATFORM='offscreen'

TOMOS="180426/026
180426/027
180426/028
180426/029
180426/030
180426/034
180426/037
180426/041
180426/043
180426/045"

export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_64.csv"
export global_output_dir="/struct/mahamid/Irene/scratch/3d-cnn/cross-validation/fas/original-training-data/"
export write_on_table='true'
# Training set parameters:
export segmentation_names='fas' #separated by commas
export split=0.7
export box_length=64
export number_iterations_data_aug=0
export min_label_fraction=0.001
export max_label_fraction=1
# Data for the new model

# Data for old models for resuming training:

for tomo_name in $TOMOS
do
    export output_dir=$global_output_dir"/"$tomo_name"/strongly_labeled_min"$min_label_fraction"_max"$max_label_fraction"/eman_filter_64pix"
    mkdir -p $output_dir
    echo 'Generating training partition for dataset' $tomo_name
    python3 runners/dataset_tables/generate_training_partitions/generate_train_and_test_partitions_multi_label_files.py  -output $output_dir -tomo_name $tomo_name -dataset_table $path_to_dataset_table -split $split -segmentation_names $segmentation_names -box $box_length -number_iter $number_iterations_data_aug -write_on_table $write_on_table -min_label_fraction $min_label_fraction -max_label_fraction $max_label_fraction
    echo "... done."
done
#
