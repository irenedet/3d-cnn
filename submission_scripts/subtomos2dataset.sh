#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --time 0-02:00
#SBATCH -o slurm_outputs/subtomos2dataset.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3

echo "Activating virtual environment"
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
echo "... done."

#export subtomos_path="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_005/tomo_partition.h5"
#export output_path="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_005/class_0/predicted_segmentation.hdf"

export class_number=0
export output_shape=(960,928,1000)
export box_length=128
export box_overlap=12
export label_name="D2_IF8_NA"

DIRS="
/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180426_004
"
#/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180426_021
#/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180426_024
#/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_003
#/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_004
#/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_005
#/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_018
#"

for dir in $DIRS
do
    export subtomos_path=$dir"/tomo_partition.h5"
    export output_path=$dir"/class_"$class_number"/predicted_segmentation.hdf"
	echo "Reading file $subtomos_path"
	echo "Running python script"
    python3 /g/scb2/zaugg/trueba/3d-cnn/runners/subtomos2dataset_new.py -subtomos_path $subtomos_path -class_number $class_number -output_path $output_path -output_shape $output_shape -box_length $box_length -overlap $box_overlap -label_name $label_name
    echo "... done."
done


