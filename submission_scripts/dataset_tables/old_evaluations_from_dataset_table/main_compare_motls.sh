#!/usr/bin/env bash

#file-specifics:
# RIBO FAS HEALTHY VPP:
#TOMOS="180426/004
#180426/005
#180426/021
#180426/024
#180711/003
#180711/004
#180711/005
#180711/018
#180713/027" # RIBO
# FAS ED DEFOCUS:
#
#TOMOS="180426/005
#180426/021
#180426/024"
#TOMOS="180426/004
#180711/003
#180711/004
#180711/005
#180711/018
#181119/002
#181119/030
#181126/002
#181126/012
#181126/025
#190301/003
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
#190301/045"
# RIBO ED DEFOCUS:
TOMOS="180426/004"
#180426/005
#TOMOS="180711/003"
#TOMOS="180711/004"
#180711/005
#"180711/018"
#180713/027"

# Tomograms data
export corrected=True # True or False
export class_number=0 # 0=ribo, 1=fas, 2=memb
export semantic_classes="ribo,fas,memb"


# CNN parameters:
depth=2
init_feat=8
output_classes=3
box_side=128
new_loader='True'

# Evaluation and particle picking parameters:
#label_name="test_R_ED_190301_005_031_033_"
#label_name="Retrained_D4_IF8_NA_except_180711_003"
label_name="Retrain_retrained_D4_IF8_NA_except_180711_003"
#label_name="R_ED_191119_002_030_191126_002_012_025__R_all_but_003_"
#label_name="D2_IF8_NA"
particle_picking_radius=16 #for particle picking
border_xy=20
lamella_extension=40
same_peak_radius_pr_analysis=10
score_threshold=-1


if [ $class_number == 0 ]; then
    echo "class_number is 0"
    if [$corrected == True]; then
        export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
        export statistics_file="/struct/mahamid/Irene/yeast/motl_comparison/ribos_corrected_stats.csv"
        global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/"$label_name"/corrected_motls"
    else
        export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_uncorrected_motls.csv"
        export statistics_file="/struct/mahamid/Irene/yeast/motl_comparison/ribos_uncorrected_stats.csv"
        global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/"$label_name"/uncorrected_motls"
    fi
elif [ $class_number == 1 ]; then
    echo "class_number is 1"
    export statistics_file="/struct/mahamid/Irene/yeast/fas_yeast_statistics.csv"
else
    echo "class_number non-supported for now"
fi




mkdir $global_output_dir

for tomo_name in $TOMOS
do
	echo "Submitting job for tomo $tomo_name"
	bash  ./submission_scripts/dataset_tables/evaluations_from_dataset_table/parameters_file_read_compare_motls.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -minimum_peak_distance $particle_picking_radius -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_radius_pr_analysis -threshold $score_threshold -semantic_classes $semantic_classes
done

# ... Finally:
echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $global_output_dir"/SCRIPT.txt"
