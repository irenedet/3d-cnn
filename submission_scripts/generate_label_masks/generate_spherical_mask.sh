#! /bin/bash

#SBATCH -A zaugg
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 0-00:30
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"
export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn

export coords_in_tom_format='True'
export class_number=0

export hdf_output_path="/scratch/trueba/3d-cnn/cnn_evaluation/yeast_dataset/004_005_021_ED_shuffle_false_frac_0_fas__D_2_IF_8/peak_calling/pr_radius_10/190301/005/class_0/peaks/in_lamella/motl_843.hdf"
export path_to_motl="/scratch/trueba/3d-cnn/cnn_evaluation/yeast_dataset/004_005_021_ED_shuffle_false_frac_0_fas__D_2_IF_8/peak_calling/pr_radius_10/190301/005/class_0/peaks/in_lamella/motl_843.csv"
export radius=10
export z_shift=0
export shape_x=928
export shape_y=928
export shape_z=500
export values_in_motl=True #True if the motl peak scores = value at spheres

echo "starting to generate hdf of particles in the motl"
python3 ./pipelines/generate_label_masks/generate_hdf_from_motl.py -motl $path_to_motl -hdf_output_path $hdf_output_path -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -radius $radius -z_shift $z_shift -coords_in_tom_format $coords_in_tom_format -values_in_motl $values_in_motl
echo "...done."

# ... Finally:
#echo "Save a copy of this script for future reference"
#SCRIPT=`realpath $0`
#cp $SCRIPT $output_dir"/SCRIPT_SPH_PARTICLE.txt"