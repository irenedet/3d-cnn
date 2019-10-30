#!/usr/bin/env bash

#except
TOMOS="190301/045"

export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
export global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/ED_DEFOCUS/"
export write_on_table='True'

# Partition parameters
export box_side=128


for tomo in $TOMOS
do
	echo "Submitting job for tomogram $tomo"
	sbatch submission_scripts/dataset_tables/partition_tomograms/partition_tomo_intersecting_lamella.sh -dataset_table $path_to_dataset_table -tomo_name $tomo -output_dir $global_output_dir -box_side $box_side -write_on_table $write_on_table
done


