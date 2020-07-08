#!/usr/bin/env bash

#TOMOS with annotated fas and ribos:
#TOMOS="190301/005
#190301/031
#190301/033"


# VPP, ED, TOMOS (with fas and ribos):
TOMOS=""

sample_type="" #ED or healthy
global_output_dir="/struct/mahamid/Irene/yeast/"$sample_type
path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
coords_in_tom_format="True"
values_in_motl="False"
write_on_dataset_table="False"
class_name="ribo"
sphere_radius=8

#dataset_table
#tomo_name
#class_name
#output_path
#radius
#coords_in_tom_format
#write_on_dataset_table
#values_in_motl


for tomo_name in $TOMOS
do
	echo "Generating mask for tomo $tomo_name"
	export output_path=$global_output_dir"/"$tomo_name"/clean_masks/ribo_sph_mask.mrc"
	bash $UPICKER_PATH/submission_scripts/dataset_tables/generate_label_masks/generate_sph_masks.sh -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -coords_in_tom_format $coords_in_tom_format -values_in_motl $values_in_motl -write_on_dataset_table $write_on_dataset_table -class_name $class_name -sphere_radius $sphere_radius -output_path $output_path
done
