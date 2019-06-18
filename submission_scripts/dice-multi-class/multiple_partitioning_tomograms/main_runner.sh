#!/usr/bin/env bash

FILES="/g/scb2/zaugg/trueba/3d-cnn/submission_data/SPOMBE/ED_DEFOCUS/190301_005.sh"


export global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/ED_DEFOCUS"

# Partition parameters
export box_side=128
export overlap=12

for param_file in $FILES
do
	echo "Reading file $param_file"
	sbatch ./submission_scripts/dice-multi-class/multiple_partitioning_tomograms/partition_tomo_intersecting_lamella.sh -parameters_file $param_file -output_dir $global_output_dir -box_side $box_side -overlap $overlap
done

# ... Finally:
echo "Save a copy of this script for future reference"
SCRIPT=`realpath $0`
cp $SCRIPT $global_output_dir"/SCRIPT.txt"

