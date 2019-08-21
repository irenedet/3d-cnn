#!/usr/bin/env bash

#file-specifics:
# RIBO FAS HEALTHY VPP:
TOMOS="180426/004
180426/005
180426/021
180426/024"
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
#TOMOS="180426/004"
#180426/005
#TOMOS="180711/003"
#TOMOS="180711/004"
#180711/005
#"180711/018"
#180713/027"

# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
export class_number=0 # 0=ribo, 1=fas, 2=memb

if [ $class_number == 0 ]; then
    echo "class_number is 0"
    export statistics_file="/struct/mahamid/Irene/yeast/ribos_yeast_statistics_corrected_motls.csv"
elif [ $class_number == 1 ]; then
    echo "class_number is 1"
    export statistics_file="/struct/mahamid/Irene/yeast/fas_yeast_statistics.csv"
else
    echo "class_number non-supported for now"
fi



# CNN parameters:
path_to_model="./models/dice_multi_label/elastic_rot_DA_004_600N_ribo_fas_memb_D_2_IF_8.pkl"
#path_to_model="./models/dice_multi_label/ED/2_corrected_membribo_fas_memb__D_2_IF_8.pkl"

#path_to_model="./models/dice_multi_label/retrained/ED/test_R_ED_190301_005_031_033__R_all_but_003_ribo_fas_memb__D_2_IF_8.pkl"
#path_to_model="./models/dice_multi_label/retrained/Retrain_retrained_except_180711_003ribo_fas_memb_D_2_IF_8.pkl"
#path_to_model="./models/dice_multi_label/retrained/Retrain_D2_IF8_NA_except_180711_003ribo_fas_memb_D_2_IF_8.pkl"
#path_to_model="./models/dice_multi_label/retrained/ED/R_ED_191119_002_030_191126_002_012_025__R_all_but_003_ribo_fas_memb__D_2_IF_8.pkl"
#path_to_model="./models/dice_multi_label/0_w_1_1_1_ribo_fas_memb_D_2_IF_8.pkl"
depth=2
init_feat=8
output_classes=3
box_side=128
new_loader='True'

# Evaluation and particle picking parameters:
label_name="elastic_rot_DA_004_600N_ribo_fas_memb_D_2_IF_8"
#label_name="2_corrected_memb"

#label_name="test_R_ED_190301_005_031_033_"
#label_name="Retrained_D4_IF8_NA_except_180711_003"
#label_name="Retrain_retrained_D4_IF8_NA_except_180711_003"
#label_name="R_ED_191119_002_030_191126_002_012_025__R_all_but_003_"
#label_name="D2_IF8_NA"
particle_picking_radius=16 #for particle picking
border_xy=20
lamella_extension=40
same_peak_radius_pr_analysis=10
score_threshold=-1
global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/"$label_name"/corrected_motls"



mkdir -p $global_output_dir

for tomo_name in $TOMOS
do
	echo "Submitting job for tomo $tomo_name"
	sbatch  ./submission_scripts/dataset_tables/evaluations_from_dataset_table/parameters_file_read.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -minimum_peak_distance $particle_picking_radius -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_radius_pr_analysis -threshold $score_threshold
done

# ... Finally:
echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $global_output_dir"/SCRIPT.txt"
