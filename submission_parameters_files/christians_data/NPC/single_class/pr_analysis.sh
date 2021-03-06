#!/usr/bin/env bash

TOMOS="190223/148 190223/178 190223/183 190223/177 190223/191 190223/192 190223/194 190223/190"
#190223/132
training_set_type="strongly_labeled0.002"
# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/NPC/SPombe/NPC_SU_table.csv"
export class_number=0 #
export semantic_classes="npc"
export statistics_file="/struct/mahamid/Irene/NPC/SPombe/peak_statistics_class_"$class_number".csv"



BN=false
depth=3
init_feat=8
output_classes=1
box_side=128
new_loader='True'
shuffle=false

# Evaluation and particle picking parameters:
particle_picking_radius=40 #for particle picking
border_xy=10
lamella_extension=0
same_peak_radius_pr_analysis=30
score_threshold=-10
test_partition_from_table=true

for tomo_name in $TOMOS
do
  echo $tomo_name
  # CNN parameters:
  model_nickname="DA_G1.5_E2_R180_DArounds4_shuffle_"$shuffle"_"$semantic_classes"__D_"$depth"_IF_"$init_feat
  path_to_model="models/npc_christian_yeast_npc/strongly_labeled0.002/"$model_nickname".pkl"
  label_name=$training_set_type"_"$model_nickname
  global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/"$training_set_type"/"$tomo_name"/"$model_nickname"/pr_radius_"$same_peak_radius_pr_analysis"/peak_calling"
  mkdir -p $global_output_dir
  echo "Submitting job for tomo" $tomo_name
  sbatch submission_scripts/dataset_tables/cnn_evaluation_runners/peak_calling/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -minimum_peak_distance $particle_picking_radius -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_radius_pr_analysis -threshold $score_threshold -BN $BN -semantic_classes $semantic_classes -test_partition_from_table $test_partition_from_table
done


