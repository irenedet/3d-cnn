#! /bin/bash


#tomo_training_list="180413/006,180413/007,180426/005,180426/006,180426/008,180426/014,180426/027,180426/037,180426/038,180426/040,180426/043,180711/005,180711/006,180711/007,180711/012,180711/017,180711/022,180713/002,180713/005,180713/007,180713/015,180713/018,180713/020,180713/025,180713/027,180713/031,180713/035,180713/037,180713/039,180713/041"

tomo_training_list="190218/049"

#export path_to_dataset_table="/struct/mahamid/Irene/yeast/npc/npc_yeast_data.csv"
#export path_to_dataset_table="/struct/mahamid/Irene/yeast/npc/npc_DA_data.csv"
#export path_to_dataset_table="/struct/mahamid/Irene/NPC/SPombe/NPC_SU_table.csv"
#export path_to_dataset_table="/struct/mahamid/Irene/NPC/SPombe/DA_NPC_SU_table.csv"
export path_to_dataset_table="/struct/mahamid/Irene/NPC/SPombe/npc_gauss_0.06_0.01_masks/DA_NPC_SU_gauss0.06_0.01_masks_table.csv"
export segmentation_names='npc'
export split=0.8
# Data for the new model
export models_tag="log_test_data_loading"
export DA_tag="G1.5_E2_R180_DArounds4"
export log_dir="log_test_data_loading"
export model_path="model_path_test_data_loading/"$models_tag
export n_epochs=3
export depth=1
export initial_features=1
export output_classes=1
export shuffle=false

# Data for old models for resuming training:
export retrain="false"
export path_to_old_model="none"
export models_notebook="/struct/mahamid/Irene/NPC/SPombe/log_test_data_loading_models_npc.csv"
export fraction="None"

echo 'Job for segmentation_names' $segmentation_names
export model_initial_name="R_"$retrain"_DA_"$DA_tag"_shuffle_"$shuffle"_"
bash submission_scripts/dataset_tables/training/training_runner.sh -fraction $fraction -path_to_dataset_table $path_to_dataset_table -tomo_training_list $tomo_training_list -split $split -output_classes $output_classes -log_dir $log_dir -model_initial_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -models_notebook $models_notebook



