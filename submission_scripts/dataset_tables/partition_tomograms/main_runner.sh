#!/usr/bin/env bash

#except
TOMOS="190301/003"
#TOMOS="190301/022
#190301/028
#190301/031
#190301/032
#190301/033
#190301/035
#190301/037"

export path_to_dataset_table="/struct/mahamid/Irene/yeast/fas_yeast_table.csv"
export global_output_dir="/struct/mahamid/Irene/yeast/ED/"
export write_on_table='True'

# Partition parameters
export box_side=128


for tomo in $TOMOS
do
	echo "Submitting job for tomogram $tomo"
	bash submission_scripts/dataset_tables/partition_tomograms/partition_tomo_intersecting_lamella.sh -dataset_table $path_to_dataset_table -tomo_name $tomo -output_dir $global_output_dir -box_side $box_side -write_on_table $write_on_table
done