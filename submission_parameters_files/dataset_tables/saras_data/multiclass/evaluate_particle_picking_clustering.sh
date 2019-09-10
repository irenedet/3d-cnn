#!/usr/bin/env bash

TOMOS="248
249
250
251"

# Tomograms data
export path_to_dataset_table="/struct/mahamid/Irene/liang_data/liang_data_multiclass.csv"
export class_number=0 # 0=ribo, 1=fas, 2=memb
export semantic_classes="70S,50S,memb"
export statistics_file="/struct/mahamid/Irene/liang_data/multiclass/clustering_statistics_class_"$class_number".csv"


# CNN parameters:
path_to_model="/g/scb2/zaugg/trueba/3d-cnn/models/lang_unets/NO_DA_ribo_D_2_IF_8.pkl"
model_nickname="NO_DA_ribo_D_2_IF_8"

BN=False
depth=2
init_feat=8
output_classes=1
box_side=128
new_loader='True'
min_cluster_size=2500
max_cluster_size=1000000

# Evaluation and particle picking parameters:
particle_picking_radius=30 #for particle picking
border_xy=10
lamella_extension=0
same_peak_radius_pr_analysis=20
score_threshold=-1
global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/"$model_nickname"/clustering/pr_radius_"$same_peak_radius_pr_analysis

mkdir -p $global_output_dir

for tomo_name in $TOMOS
do
	echo "Submitting job for tomo $tomo_name"
	sbatch  ./submission_scripts/dataset_tables/cnn_evaluation_runners/cluster_centroids/runner.sh -statistics_file $statistics_file -output_dir $global_output_dir -dataset_table $path_to_dataset_table -tomo_name $tomo_name -path_to_model $path_to_model -label_name $model_nickname -depth $depth -init_feat $init_feat -output_classes $output_classes -class_number $class_number -box_side $box_side -new_loader $new_loader -minimum_peak_distance $particle_picking_radius -border_xy $border_xy -lamella_extension $lamella_extension -same_peak_distance $same_peak_radius_pr_analysis -threshold $score_threshold -BN $BN -min_cluster_size $min_cluster_size -max_cluster_size $max_cluster_size -semantic_classes $semantic_classes
done

# ... Finally:
echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $global_output_dir"/SCRIPT.txt"
