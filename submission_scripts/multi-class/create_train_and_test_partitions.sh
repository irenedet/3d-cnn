#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --time 0-10:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

export path_to_raw="/scratch/trueba/3d-cnn/clean/180426_004/subtomo380-600.hdf"
export path_to_labeled="/struct/mahamid/Irene/yeast/180426/004/training/multi_class/ribos_corrected_fas_memb_mask.hdf"
export output_dir="/scratch/trueba/3d-cnn/training_data/multi-class/004/G_sigma1"
export shape_x=928
export shape_y=928
export shape_z=221

#export path_to_raw="/scratch/trueba/3d-cnn/clean/180426_005/subtomo370-620.hdf"
#export path_to_labeled="/scratch/trueba/3d-cnn/clean/180426_005/4b_ribos_mask.hdf"
#export output_dir="/scratch/trueba/3d-cnn/training_data/TEST/mixed_training/005/G_sigma1"
#export shape_x=928
#export shape_y=928
#export shape_z=251

#export path_to_raw="/scratch/trueba/3d-cnn/clean/180426_006/subtomo330-650.hdf"
#export path_to_labeled="/scratch/trueba/3d-cnn/clean/180426_006/4b_ribos_mask.hdf"
#export output_dir="/scratch/trueba/3d-cnn/training_data/TEST/mixed_training/006/G_sigma1/"
#export shape_x=928
#export shape_y=928
#export shape_z=321

export label_name="ribos_corrected_fas_memb"
export split=180  # Between partitions of testing and training data
export box_side=128

# For data augmentation:
export number_iter=6
export train_split=144  # within training data for nnet training & data aug
export overlap=12


module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn
echo "... done"

echo "Running python script..."
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/multi-class/training/create_train_and_test_partitions.py -raw $path_to_raw -labeled $path_to_labeled -output $output_dir -label $label_name -box $box_side -shapex $shape_x -shapey $shape_y -shapez $shape_z -number_iter $number_iter -split $split -train_split $train_split -overlap $overlap

cp /g/scb2/zaugg/trueba/3d-cnn/submission_scripts/multi-class/create_train_and_test_partitions.sh $output_dir'/script.run'