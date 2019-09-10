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

# To be modified by user
echo 'starting virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/

tomo="180711_005"
#/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/for_sara/Retrained_D4_IF8_NA_except_180711_003/180711_005/class_0/in_lamella/motl_833.csv

export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/clusters/test/clustering_test_NO_DA_ribo_D_2_IF_8_pr_radius_20/full_dataset/246/class_0/combined_motl_1.5sph"
#export path_to_motl_predicted="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180426_004/class_0/in_lamella/motl_2608.csv"
export path_to_motl_true="/struct/mahamid/Liang/tomogram/3D_CNN/network_training/data/00246_motl_ribo_clean.em"
export path_to_motl_predicted=$output_dir"/combined_motl.csv"
export testing_set_data_path="None" #set the partition produced by particle picking pipeline
export radius=20
export z_shift=0 # z_shift between original tomogram and subtomogram of partition
export x_shift=0 # -16 for 004, 16 for 005 (if dataset is square)
export shape_x=928
export shape_y=928
export shape_z=450
export box=128

echo "Starting to generate precision recall plots"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/performance_nnet_quantification/precision_recall_plots.py -motl $path_to_motl_predicted -clean $path_to_motl_true -output $output_dir -test_file $testing_set_data_path -radius $radius -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -x_shift $x_shift -z_shift $z_shift -box $box
echo "...done."

echo "Save a copy of this script for future reference"
cp /g/scb2/zaugg/trueba/3d-cnn/submission_scripts/performance_nnet_quantification_pipeline/prec_recall_plots.sh $output_dir"/PR_submission_script.txt"