#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 0-00:20
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


# ToDo check why this does not run in cluster
module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse


export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn

#export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/mixed_trainset/no_shuffle/G1_confs_4_5_/peaks_in_training_partition"
#export path_to_motl_clean='/scratch/trueba/3d-cnn/clean/180426_006/motl_clean_4b.em'
#export path_to_csv_motl=$(ls $output_dir/motl*)
#export testing_set_data_path="/scratch/trueba/3d-cnn/training_data/TEST/mixed_training/006/train_and_test_partitions/partition_training.h5" #$(ls $output_dir/partition_subtomograms_.h5)
#export radius=8
#export z_shift=330  # shift between original tomogram and subtomogram of analysis
#export x_shift=0
#export shape_x=927
#export shape_y=927
#export shape_z=321
#export box=128

#export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/multi-class/G_sigma1_D5_IF8/fas/test_and_train"
##export path_to_motl_clean="/scratch/trueba/3d-cnn/clean/180426_004/motl_clean_fas_4b_iniavg.em"
#export path_to_motl_clean='/scratch/trueba/cnn/004/4bin/cnn/motl_clean_4b.em'
#export path_to_csv_motl="/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/multi-class/G_sigma1_D5_IF8/motl_3852_class_2.csv"
##"/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/multi-class/G_sigma1_D4_IF8/motl_4629_class_1.csv"
#export testing_set_data_path="/scratch/trueba/3d-cnn/training_data/multi-class/004/G_sigma1/train_and_test_partitions/partition_training.h5" #$(ls $output_dir/partition_subtomograms_.h5)
#export radius=10
#export z_shift=380  # shift between original tomogram and subtomogram of analysis
#export x_shift=0
#export shape_x=928
#export shape_y=928
#export shape_z=221
#export box=128

#export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/confs_4_5_bis_/peaks_in_training_partition"
#export path_to_motl_clean='/scratch/trueba/3d-cnn/clean/180426_005/motl_clean_4b.em'
#export path_to_csv_motl=$(ls $output_dir/motl*)
#export testing_set_data_path="/scratch/trueba/3d-cnn/cnn_evaluation/180426_005/gaussian_aug/confs_4_5_sigma0-8/partition_subtomograms_.h5" #$(ls $output_dir/partition_subtomograms_.h5)
#export radius=8
#export z_shift=370  # shift between original tomogram and subtomogram of analysis
#export x_shift=16
#export shape_x=927
#export shape_y=927
#export shape_z=251
#export box=128

#echo "Starting to generate precision recall plots"
#python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/performance_nnet_quantification/precision_recall_plots.py -motl $path_to_csv_motl -clean $path_to_motl_clean -output $output_dir -test_file $testing_set_data_path -radius $radius -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -x_shift $x_shift -z_shift $z_shift -box $box
#echo "...done."

echo "Starting to generate precision recall plots"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/performance_nnet_quantification/filter_with_lamella_mask.py
echo "...done."