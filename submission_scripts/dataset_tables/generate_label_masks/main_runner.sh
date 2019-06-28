#!/usr/bin/env bash

#TOMOS with annotated fas and ribos:
TOMOS="190301/005
190301/031
190301/033"


# VPP, ED, TOMOS (with fas and ribos):
#TOMOS="181119/002"
#181119/030
#181126/002
#181126/012
#181126/025"
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

sample_type="ED" #ED or healthy
global_output_dir="/struct/mahamid/Irene/yeast/"$sample_type
path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table_ribo_single_class.csv"
coords_in_tom_format="True"
values_in_motl="False"
class_number=0
write_on_dataset_table="True"

for tomo_name in $TOMOS
do
	echo "Submitting job for tomo $tomo_name"
	sbatch ./submission_scripts/dataset_tables/generate_label_masks/generate_sph_masks.sh -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -class_number $class_number -coords_in_tom_format $coords_in_tom_format -values_in_motl $values_in_motl -write_on_dataset_table $write_on_dataset_table
done

# ... Finally:
echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $global_output_dir"/SCRIPT.txt"