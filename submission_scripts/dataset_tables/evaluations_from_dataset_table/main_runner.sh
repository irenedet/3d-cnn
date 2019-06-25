#!/usr/bin/env bash

#file-specifics:
#
#TOMOS="180426/004
#180426/005
#180426/021
#180426/024
#180711/003
#180711/004
#180711/005
#180711/018
#180713/027"
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

# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
export statistics_file="/struct/mahamid/Irene/yeast/ribos_yeast_statistics_corrected_motls.csv"

# CNN parameters:
path_to_model="./models/dice_multi_label/retrained/Retrain_D2_IF8_NA_except_180711_003ribo_fas_memb_D_2_IF_8.pkl"
depth=2
init_feat=8
output_classes=3
box_side=128
new_loader='True'

# Evaluation and particle picking parameters:
label_name="Retrained_D4_IF8_NA_except_180711_003"
class_number=0 # 0=ribo, 1=fas, 2=memb
particle_picking_radius=16 #for particle picking
border_xy=20
lamella_extension=40
same_peak_radius_pr_analysis=10
score_threshold=-1
global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/"$label_name


mkdir $global_output_dir

for tomo_name in $TOMOS
do
	echo "Submitting job for tomo $tomo_name"
	sbatch ./submission_scripts/dataset_tables/evaluations_from_dataset_table/parameters_file_read.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -minimum_peak_distance $particle_picking_radius -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_radius_pr_analysis -threshold $score_threshold
done

# ... Finally:
echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $global_output_dir"/SCRIPT.txt"
