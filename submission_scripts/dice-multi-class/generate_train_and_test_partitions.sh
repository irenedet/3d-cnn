#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --time 0-05:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

#export segmentation_names="ribo,fas,memb"
export split=0.8  # Between partitions of testing and training data
export box_side=128

# For data augmentation:
export number_iter=6
export overlap=12
export global_output_dir="/scratch/trueba/3d-cnn/training_data/dice-multi-class/"

module load Anaconda3
echo 'starting virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/

export PYTHONPATH=/g/scb2/zaugg/trueba/3d-cnn


FILES="/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/ED_TOMOS/181119_002.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/ED_TOMOS/181119_030.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/ED_TOMOS/181126_002.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/ED_TOMOS/181126_012.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/ED_TOMOS/181126_025.sh
/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/ED_TOMOS/190301_005.sh"

for param_file in $FILES
do
	echo "Reading file $param_file"
	source $param_file
	export tomo_name=$tomo_name
	export path_to_raw=$eman2_filtered_file
	export label_0=$class_0_mask
	export label_1=$class_1_mask
	export label_2=$class_2_mask
    export output_dir=$global_output_dir$tomo_name"/NA/"
    export shape_x=$input_xdim
    export shape_y=$input_ydim
    export shape_z=$input_zdim
    echo 'starting python script'
    python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/dice_multi-class/generate_training_set/generate_train_and_test_partitions_multi_label_files.py -raw $path_to_raw  -label_0 $label_0 -label_1 $label_1 -label_2 $label_2 -output $output_dir -box $box_side -shapex $shape_x -shapey $shape_y -shapez $shape_z -number_iter $number_iter -split $split -overlap $overlap
    echo 'done'
done
