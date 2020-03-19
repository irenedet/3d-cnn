#!/usr/bin/env bash


TOMOS="0 1 2 3 4 5 6 7 8 9"


export path_to_dataset_table="/struct/mahamid/Irene/shrec2020/data/table.csv"
export global_output_dir="/struct/mahamid/Irene/shrec2020/data/partitions/testing"
mkdir -p $global_output_dir
export write_on_table='false'

# Partition parameters
export box_side="512,512,512"

for tomo in $TOMOS
do
	echo "Submitting job for tomogram $tomo"
	bash $UPICKER_PATH/submission_scripts/dataset_tables/partition_tomograms/partition_tomo_intersecting_lamella.sh -dataset_table $path_to_dataset_table -tomo_name $tomo -output_dir $global_output_dir -box_side $box_side -write_on_table $write_on_table
done
