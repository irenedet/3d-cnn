#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 40G
#SBATCH --time 0-00:30
#SBATCH -o slurm_outputs/subtomos2dataset.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3

echo "Activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done."

#export subtomos_path="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_005/tomo_partition.h5"
#export output_path="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_005/class_0/predicted_segmentation.hdf"
export cluster_labels=False
export class_number=0
export output_shape=(960,931,500)
export box_length=128
export box_overlap=12
export label_name="004_005_021_ED_shuffle_false_frac_1_fas__D_1_IF_8"
export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/yeast_dataset/004_005_021_ED_shuffle_false_frac_1_fas__D_1_IF_8/peak_calling/pr_radius_10/190301/005/"
DIRS="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/ED_DEFOCUS/190301/005/tomo_partition.h5"
#/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180426_021
#/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180426_024
#/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_003
#/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_004
#/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_005
#/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_018
#"

for dir in $DIRS
do
    export subtomos_path=$dir
    export output_path=$output_dir"/class_"$class_number"/prediction.hdf"
	echo "Reading file $subtomos_path"
	echo "Running python script"
    python3 ./runners/subtomos2dataset_new.py -subtomos_path $subtomos_path -class_number $class_number -output_path $output_path -output_shape $output_shape -box_length $box_length -overlap $box_overlap -label_name $label_name -cluster_labels $cluster_labels
    echo "... done."
done


