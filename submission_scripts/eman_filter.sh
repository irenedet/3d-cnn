#!/usr/bin/env bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 32G
#SBATCH --time 0-05:00
#SBATCH -o eman2_filter.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


source_files="/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180426_005.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_003.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_004.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_005.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180711_018.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/9_VPP_TOMOS/180713_027.sh"

module load EMAN2

for param_file in $source_files
do
	echo "Reading file $param_file"
	source $param_file
	export origin_file=$origin_file
	export eman2_filtered_file=$eman2_filtered_file
    export mult_factor=$mult_factor
	echo origin_file=$origin_file
	echo eman2_filtered_file=$eman2_filtered_file
	rm $eman2_filtered_file
	e2proc3d.py $origin_file $eman2_filtered_file --mult=$mult_factor --process filter.lowpass.gauss:cutoff_abs=.25 --process filter.highpass.gauss:cutoff_pixels=5 --process normalize --process threshold.clampminmax.nsigma:nsigma=3
done