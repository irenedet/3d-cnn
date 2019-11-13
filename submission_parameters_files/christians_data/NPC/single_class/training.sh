#! /bin/bash


#tomo_training_list="180413/006,180413/007,180426/005,180426/006,180426/008,180426/014,180426/027,180426/037,180426/038,180426/040,180426/043,180711/005,180711/006,180711/007,180711/012,180711/017,180711/022,180713/002,180713/005,180713/007,180713/015,180713/018,180713/020,180713/025,180713/027,180713/031,180713/035,180713/037,180713/039,180713/041"

tomo_training_list="190218/043,190218/048,190218/049,190218/051,190218/052,190218/054,190218/056,190218/059,190218/061,190218/062,190218/064,190218/065,190218/066,190218/067,190218/068,190218/069,190218/070,190218/071,190218/072,190218/073,190218/075,190218/076,190218/078,190218/081,190218/082,190218/084,190218/085,190218/086,190218/087,190218/088,190218/089,190218/091,190218/092,190218/093,190218/094,190218/095,190218/096,190218/097,190218/098,190218/099,190218/101,190218/102,190218/103,190218/104,190218/105,190218/108,190218/110,190218/111,190218/112,190218/114,190218/115,190218/116,190218/117,190218/118,190218/119,190218/120,190218/121,190218/122,190218/123,190218/124,190218/125,190223/129,190223/130,190223/131,190223/133,190223/135,190223/136,190223/139,190223/140,190223/141,190223/142,190223/143,190223/144,190223/145,190223/146,190223/149,190223/151,190223/152,190223/153,190223/154,190223/155,190223/156,190223/157,190223/159,190223/160,190223/162,190223/163,190223/165,190223/166,190223/168,190223/169,190223/171,190223/172,190223/173,190223/174,190223/175,190223/176,190223/179,190223/180,190223/181,190223/182,190223/184,190223/185,190223/186,190223/187,190223/188,190223/189,190218/060,190218/063,190218/077,190218/100,190218/050,190218/090,190218/106,190218/083,190218/113"

#TEST="190223/132
#190223/148
#190223/178
#190223/183
#190223/177
#190223/190
#190223/191
#190223/192
#190223/193
#190223/194
#190218/044"

#export path_to_dataset_table="/struct/mahamid/Irene/yeast/npc/npc_yeast_data.csv"
#export path_to_dataset_table="/struct/mahamid/Irene/yeast/npc/npc_DA_data.csv"
#export path_to_dataset_table="/struct/mahamid/Irene/NPC/SPombe/NPC_SU_table.csv"
#export path_to_dataset_table="/struct/mahamid/Irene/NPC/SPombe/DA_NPC_SU_table.csv"
export path_to_dataset_table="/struct/mahamid/Irene/NPC/SPombe/npc_gauss_0.06_0.01_masks/DA_NPC_SU_gauss0.06_0.01_masks_table.csv"
export segmentation_names='npc'
export split=0.8
# Data for the new model
export models_tag="gauss_0.06_0.01_strongly_labeled0.02"
export DA_tag="G1.5_E2_R180_DArounds4"
export log_dir="/g/scb2/zaugg/trueba/3d-cnn/log_christian_yeast_npc"
export model_path="models/npc_christian_yeast_npc/"$models_tag
export n_epochs=300
export depth=4
export initial_features=8
export output_classes=1
export shuffle=false

# Data for old models for resuming training:
export retrain="false"
export path_to_old_model="none"
export models_notebook="/struct/mahamid/Irene/NPC/SPombe/models_npc.csv"
export fraction="None"

echo 'Job for segmentation_names' $segmentation_names
export model_initial_name="R_"$retrain"_DA_"$DA_tag"_shuffle_"$shuffle"_"
sbatch submission_scripts/dataset_tables/training/training_runner.sh -fraction $fraction -path_to_dataset_table $path_to_dataset_table -tomo_training_list $tomo_training_list -split $split -output_classes $output_classes -log_dir $log_dir -model_initial_name $model_initial_name -model_path $model_path -n_epochs $n_epochs -segmentation_names $segmentation_names -retrain $retrain -path_to_old_model $path_to_old_model -depth $depth -initial_features $initial_features -models_notebook $models_notebook



