#!/usr/bin/env bash

TOMOS="190223/132 190223/148 190223/178 190223/183 190223/177 190223/191 190223/192 190223/194 190223/190"

training_set_type="strongly_labeled0.002"
# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/NPC/SPombe/NPC_SU_table.csv"
export class_number=0 #
export statistics_file="/struct/mahamid/Irene/NPC/SPombe/dice_statistics_class_"$class_number".csv"

BN=false
depth=2
init_feat=8
output_classes=1
semantic_classes="npc"
box_side=128
new_loader='True'
shuffle=false

test_partition_from_table=true

for tomo_name in $TOMOS
do
  echo $tomo_name
  # CNN parameters:
  model_nickname="R_DA_G1.5_E2_R180_DArounds4_shuffle_"$shuffle"_"$semantic_classes"__D_"$depth"_IF_"$init_feat
  path_to_model="models/npc_christian_yeast_npc/strongly_labeled0.002/"$model_nickname".pkl"
  label_name=$training_set_type"_"$model_nickname
  global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/"$training_set_type
  mkdir -p $global_output_dir
  echo "Submitting job for tomo" $tomo_name
  sbatch submission_scripts/dataset_tables/cnn_evaluation_runners/continuous_dice/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -BN $BN -semantic_classes $semantic_classes -test_partition_from_table $test_partition_from_table
done


















