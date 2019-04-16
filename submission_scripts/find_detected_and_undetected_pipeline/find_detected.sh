#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 2G
#SBATCH --time 0-00:10
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn

export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/180426_004/multi-class/G_sigma1_D4_IF8/fas/in_lamella_wide"
export path_to_csv_motl=$(ls $output_dir/motl*.csv)
export path_to_motl_clean="/scratch/trueba/3d-cnn/clean/180426_004/motl_clean_fas_4b_iniavg.em"
export min_peak_distance=10
export x_shift=0 # only different from 0 for tomo 005
export label_name="fas"
echo "running script to find undetected and detected particles"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/find_detected_and_undetected/get_detected.py -motl $path_to_csv_motl -clean $path_to_motl_clean -output $output_dir -min_peak_distance $min_peak_distance -label $label_name -x_shift $x_shift
echo "...done."

export path_to_detected_motl=$(ls $output_dir/detected/motl*.csv)
export output_dir_detected=$output_dir"/detected/"
export z_shift=-380  # shift between original tomogram and subtomogram of analysis
export shape_x=928
export shape_y=928
export shape_z=221
export radius=10


echo "starting to generate hdf of detected particles"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/find_detected_and_undetected/generate_hdf_from_motl.py -motl $path_to_detected_motl -output $output_dir_detected -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -radius $radius -z_shift $z_shift
echo "...done."

export output_dir_undetected=$output_dir"/undetected"
export path_to_undetected_motl=$(ls $output_dir/undetected/motl*.csv)
echo "starting to generate hdf of undetected particles"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/find_detected_and_undetected/generate_hdf_from_motl.py -motl $path_to_undetected_motl -output $output_dir_undetected -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -radius $radius -z_shift $z_shift
echo "...done."

