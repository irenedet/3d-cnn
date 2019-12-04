#!/usr/bin/env bash

TOMOS="190223/178
190223/183
190218/044
190223/132
190223/148
190223/177
190223/190
190223/191
190223/192
190223/194"

training_set_type="NPC_SU_gauss_0.06_0.001_strongly_labeled0.002"
# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/NPC/SPombe/npc_gauss_0.06_0.01_masks/DA_NPC_SU_gauss0.06_0.01_masks_table.csv"
export class_number=0 #
export statistics_file="/struct/mahamid/Irene/NPC/SPombe/npc_gauss_0.06_0.01_masks/dice_statistics_class_"$class_number".csv"

BN=false
depth=4
init_feat=8
output_classes=1
semantic_classes="npc"
box_side=128
new_loader='True'
shuffle=false

test_partition_from_table=false

for tomo_name in $TOMOS
do
  echo $tomo_name
  # CNN parameters:
  model_nickname="R_false_DA_DA_G1.5_E2_R180_shuffle_false_npc__D_4_IF_8"
  path_to_model="/struct/mahamid/Irene/NPC/SPombe/models/"$training_set_type"/"$model_nickname".pkl"
  label_name=$training_set_type"_"$model_nickname
  global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/NPC_SU/"$training_set_type
  mkdir -p $global_output_dir
  echo "Submitting job for tomo" $tomo_name
  sbatch $UPICKER_PATH/submission_scripts/dataset_tables/cnn_evaluation_runners/continuous_dice/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -BN $BN -semantic_classes $semantic_classes -test_partition_from_table $test_partition_from_table
done

#training_set_type="NPC_SU_gauss_0.06_strongly_labeled0.002"
## Tomograms data
#export path_to_dataset_table="/struct/mahamid/Irene/NPC/SPombe/npc_gauss_0.06_masks/NPC_SU_gauss0.06_masks_table.csv"
#export class_number=0 #
#export statistics_file="/struct/mahamid/Irene/NPC/SPombe/npc_gauss_0.06_masks/dice_statistics_class_"$class_number".csv"
#
#BN=false
#depth=4
#init_feat=8
#output_classes=1
#semantic_classes="npc"
#box_side=128
#new_loader='True'
#shuffle=false
#
#test_partition_from_table=false
#
#for tomo_name in $TOMOS
#do
#  echo $tomo_name
#  # CNN parameters:
#  model_nickname="R_false_DA_none_shuffle_false_npc__D_4_IF_8"
#  path_to_model="/struct/mahamid/Irene/NPC/SPombe/models/"$training_set_type"/"$model_nickname".pkl"
#  label_name=$training_set_type"_"$model_nickname
#  global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/NPC_SU/"$training_set_type
#  mkdir -p $global_output_dir
#  echo "Submitting job for tomo" $tomo_name
#  sbatch $UPICKER_PATH/submission_scripts/dataset_tables/cnn_evaluation_runners/continuous_dice/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -BN $BN -semantic_classes $semantic_classes -test_partition_from_table $test_partition_from_table
#done















