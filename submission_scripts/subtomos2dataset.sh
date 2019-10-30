#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 20G
#SBATCH --time 0-00:10
#SBATCH -o slurm_outputs/subtomos2dataset.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3

echo "Activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done."
# # For NPC
#export cluster_labels=False
#export class_number=0
#export output_shape=(928,928,400)
#export box_length=128
#export box_overlap=12
##reconstruction_type is either "prediction" or "labels" or "raw":
#export reconstruction_type="prediction"
#export tomo_name="180713/050"
#export label_name="strongly_labeled0.02_0413_006-0713_043_DA_G1.5_E2_R180_shuffle_false_npc__D_4_IF_8"
#export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/strongly_labeled0.02_0413_006-0713_043/"$tomo_name"/DA_G1.5_E2_R180_shuffle_false_npc__D_4_IF_8/pr_radius_120/peak_calling/"$tomo_name
#mkdir -p $output_dir
#DIRS="/struct/mahamid/Irene/yeast/healthy/"$tomo_name"/npc_class/train_and_test_partitions/full_partition.h5"
#
#for dir in $DIRS
#do
#    export subtomos_path=$dir
##    export output_path=$output_dir"/class_"$class_number"/prediction.hdf"
#    export output_path=$output_dir"/prediction.hdf"
#	  echo "Reading file $subtomos_path"
#	  echo "Running python script"
#    python3 runners/subtomos2dataset_new.py -subtomos_path $subtomos_path -class_number $class_number -output_path $output_path -output_shape $output_shape -box_length $box_length -overlap $box_overlap -label_name $label_name -cluster_labels $cluster_labels -reconstruction_type $reconstruction_type
#    echo "... done."
#done
#
# # For FAS
export cluster_labels=False
export class_number=0
export output_shape=(928,928,500)
export box_length=128
export box_overlap=12
#reconstruction_type is either "prediction" or "labels" or "raw":
export reconstruction_type="prediction"
export label_name="fas_fractions_004_005_021_ED_and_def_shuffle_false_frac_4_fas__D_1_IF_12"
export tomo_name="181126/002"
export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/yeast_dataset/"$label_name"/peak_calling/pr_radius_10/"$tomo_name
mkdir -p $output_dir
DIRS="/struct/mahamid/Irene/yeast/ED/"$tomo_name"/eman_filt_eman_filt_tomo_partition.h5"

count=0
for dir in $DIRS
do
    echo $count
    export subtomos_path=$dir
    export output_path=$output_dir"/prediction.hdf"
	  echo "Reading file $subtomos_path"
	  echo "Running python script"
    python3 runners/subtomos2dataset_new.py -subtomos_path $subtomos_path -class_number $class_number -output_path $output_path -output_shape $output_shape -box_length $box_length -overlap $box_overlap -label_name $label_name -cluster_labels $cluster_labels -reconstruction_type $reconstruction_type
    echo "... done."
    count=$((count+1))
done


