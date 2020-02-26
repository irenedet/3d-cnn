#!/usr/bin/env bash

#except
#TOMOS="190301/003"
#190301/005
#190301/009
#190301/012
#190301/016
#190301/022
#190301/028
#190301/031
#190301/032
#190301/033
#190301/035
#190301/037
#190301/043
TOMOS="180426/004
180426/005
180426/021
180426/024"


export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_64.csv"
export global_output_dir="/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/test_partitions/64pix/"
mkdir -p $global_output_dir
export write_on_table='true'

# Partition parameters
export box_side=64

for tomo in $TOMOS
do
	echo "Submitting job for tomogram $tomo"
	bash submission_scripts/dataset_tables/partition_tomograms/partition_tomo_intersecting_lamella.sh -dataset_table $path_to_dataset_table -tomo_name $tomo -output_dir $global_output_dir -box_side $box_side -write_on_table $write_on_table
done
