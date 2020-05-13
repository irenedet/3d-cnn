#!/usr/bin/env bash


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
export global_output_dir="/struct/mahamid/Irene/yeast/healthy/"
mkdir -p $global_output_dir
export write_on_table='True'
export processing_tomo="1xf_tomo"
# Partition parameters
export box_side="960,928,1000"

for tomo in $TOMOS
do
	echo "Submitting job for tomogram $tomo"
	bash $UPICKER_PATH/submission_scripts/dataset_tables/partition_tomograms/partition_tomo_intersecting_lamella.sh -dataset_table $path_to_dataset_table -tomo_name $tomo -output_dir $global_output_dir -box_side $box_side -write_on_table $write_on_table -processing_tomo $processing_tomo
done
