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
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse


export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_4_5_bis_/"
export path_to_csv_motl="/scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_4_5_bis_/motl_4896.csv"
export path_to_motl_clean='/scratch/trueba/3d-cnn/clean/180426_006/motl_clean_4b.em'
export radius=8
export label_name="ribosomes"
echo "running script to find undetected and detected particles"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/find_detected_and_undetected/get_detected.py -motl $path_to_csv_motl -clean $path_to_motl_clean -output $output_dir -radius $radius -label $label_name
echo "...done."

export path_to_detected_motl=$(ls /scratch/trueba/3d-cnn/cnn_evaluation/180426_006/confs_4_5_bis_/detected/motl*.csv)
export output_dir_detected=$output_dir"detected/"
export z_shift=-330  # shift between original tomogram and subtomogram of analysis
export shape_x=927
export shape_y=927
export shape_z=321
export radius=8
echo "starting to generate hdf of detected particles"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/find_detected_and_undetected/generate_hdf_from_motl.py -motl $path_to_detected_motl -output $output_dir_detected -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -radius $radius -z_shift $z_shift
echo "...done."

export output_dir_undetected=$output_dir"undetected"
export path_to_undetected_motl=$(ls $output_dir_undetected/motl*.csv)
echo "starting to generate hdf of undetected particles"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/find_detected_and_undetected/generate_hdf_from_motl.py -motl $path_to_undetected_motl -output $output_dir_undetected -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -radius $radius -z_shift $z_shift
echo "...done."

