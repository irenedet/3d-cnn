#!/usr/bin/env bash

#TOMOS="190301/003"
#TOMOS="190301/005
#190301/009
#190301/012
#TOMOS="190301/016
#190301/022
#190301/028
#190301/031
#190301/032
#190301/033
#190301/035
#190301/037
#190301/043
TOMOS="180426/004
180426/005
180426/021
180426/024"

#fraction=2
fractions_name="FAS_non_sph_masks"
export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
DA=none

#export class_number=0 #
#export particle_picking_radius=10 #for particle picking
#export same_peak_radius_pr_analysis=8
export semantic_classes="fas"
#export semantic_classes="ribo,fas,memb"


BN=false
depth=2
init_feat=8
output_classes=1
box_side=128
new_loader='True'
shuffle=true
retrain=false
encoder_dropout=0
decoder_dropout=0

model_nickname="R_false_encoder_dropout_0_decoder_dropout_0_BN_false_DA_none_shuffle_true_frac_None_fas__D_2_IF_8"
#model_nickname="R_false_encoder_dropout_0.2_decoder_dropout_0.2_BN_false_DA_none_shuffle_true_frac_2_ribo_fas_memb__D_2_IF_8"
# model_nickname="R_"$retrain"_encoder_dropout_"$encoder_dropout"_decoder_dropout_"$decoder_dropout"_BN_"$BN"_DA_"$DA"_shuffle_"$shuffle"_frac_"$fraction"_ribo_fas_memb__D_"$depth"_IF_"$init_feat
label_name=$fractions_name"_"$model_nickname


for tomo_name in $TOMOS
do
  path_to_model="/struct/mahamid/Irene/cross-validation/multiclass/models/"$fractions_name"/"$model_nickname".pkl"
  export tomo_name=$tomo_name
  echo "Submitting job for tomo $tomo_name"
  bash submission_scripts/dataset_tables/segmentation_cnn/runner.sh -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $label_name -depth $depth -init_feat $init_feat -output_classes $output_classes -box_side $box_side -new_loader $new_loader -BN $BN -semantic_classes $semantic_classes -encoder_dropout $encoder_dropout -decoder_dropout $decoder_dropout
done


# For reconstruction of segmentation:
export cluster_labels=False
export class_number=0
export box_overlap=12
#reconstruction_type is either "prediction", "raw", or "labels":
export reconstruction_type="prediction"
export global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/"$fractions_name"/"$model_nickname"/"

for tomo_name in $TOMOS
do
    export output_dir=$global_output_dir"/"$tomo_name"/class_"$class_number
    mkdir -p $output_dir
    export output_path=$output_dir"/prediction.hdf"
	  echo "reconstructing prediction for $tomo_name"
	  echo "Running python script"
    python3 $UPICKER_PATH/runners/dataset_tables/subtomos2datasets/subtomos2dataset.py -semantic_names $semantic_classes -dataset_table $dataset_table -tomo_name $tomo_name -class_number $class_number -output_path $output_path -box_length $box_side -overlap $box_overlap -label_name $label_name -cluster_labels $cluster_labels -reconstruction_type $reconstruction_type
    echo "... done."
done



