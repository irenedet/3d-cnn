#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 0-00:50
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


module load Anaconda3
export QT_QPA_PLATFORM='offscreen'
export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn

# To be modified by user
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/uni-class-9TOMOS/confs_D5_IF4_005/in_lamella/"
export path_to_motl_clean="/scratch/trueba/3d-cnn/clean/180426_005/motl_clean_4b.em"
export path_to_csv_motl="/scratch/trueba/3d-cnn/cnn_evaluation/uni-class-9TOMOS/confs_D5_IF4_005/in_lamella/motl_3358.csv"
export testing_set_data_path="/struct/mahamid/Irene/yeast/ribosomes/180426_005/G_sigma1/train_and_test_partitions/test_partition.h5" #set the partition produced by particle picking pipeline
export radius=10
export z_shift=0  # shift between original tomogram and subtomogram of analysis
export x_shift=16 #check if it is -16 or 16
export shape_x=928
export shape_y=928
export shape_z=1000
export box=128

echo "Starting to generate precision recall plots"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/performance_nnet_quantification/precision_recall_plots.py -motl $path_to_csv_motl -clean $path_to_motl_clean -output $output_dir -test_file $testing_set_data_path -radius $radius -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -x_shift $x_shift -z_shift $z_shift -box $box
echo "...done."

echo "Save a copy of this script for future reference"
cp /g/scb2/zaugg/trueba/3d-cnn/submission_scripts/performance_nnet_quantification_pipeline/prec_recall_plots.sh $output_dir"/PR_submission_script.txt"