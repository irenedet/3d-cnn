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

TOMOS="180426/026"
#TOMOS="180426/027
#180426/028
#180426/029
#180426/030
#180426/034
#180426/037
#180426/041
#180426/043
#180426/045"

export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_4bin_fas_single_filter.csv"
export global_output_dir="/struct/mahamid/Irene/scratch/3d-cnn/cross-validation/fas/original-training-data"
export write_on_table='true'
# Training set parameters:
export segmentation_names='fas' #separated by commas
export box_length=64
export min_label_fraction=0.01
export max_label_fraction=1
export processing_tomo='1xf_tomo'
export image_acquisition_parameter='vpp'
export n_total_cubes=4400
# Data for old models for resuming training:

for tomo_name in $TOMOS
do
    export output_dir=$global_output_dir"/"$tomo_name"/strongly_labeled_min"$min_label_fraction"_max"$max_label_fraction"/random_partition/1xfilter_64pix"
    mkdir -p $output_dir
    echo 'Generating training partition for dataset' $tomo_name
    python3 $UPICKER_PATH/runners/dataset_tables/generate_training_partitions/generate_random_training_partition.py  -output $output_dir -tomo_name $tomo_name -dataset_table $path_to_dataset_table -segmentation_names $segmentation_names -box $box_length -write_on_table $write_on_table -min_label_fraction $min_label_fraction -max_label_fraction $max_label_fraction -processing_tomo $processing_tomo -image_acquisition_parameter $image_acquisition_parameter -n_total $n_total_cubes
    echo "... done."
done
