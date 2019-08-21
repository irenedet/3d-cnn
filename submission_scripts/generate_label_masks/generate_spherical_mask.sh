#! /bin/bash

#SBATCH -A zaugg
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 64G
#SBATCH --time 0-00:10
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

export hdf_output_path="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/for_sara/Retrained_D4_IF8_NA_except_180711_003/180711_005/class_0/in_lamella/mask_motl_833.hdf"
export path_to_motl="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/for_sara/Retrained_D4_IF8_NA_except_180711_003/180711_005/class_0/in_lamella/motl_833.csv"
export radius=8
export z_shift=0
export shape_x=960
export shape_y=930
export shape_z=500

echo "starting to generate hdf of particles in the motl"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/generate_label_masks/generate_hdf_from_motl.py -motl $path_to_motl -hdf_output_path $hdf_output_path -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -radius $radius -z_shift $z_shift -coords_in_tom_format $coords_in_tom_format
echo "...done."

# ... Finally:
#echo "Save a copy of this script for future reference"
#SCRIPT=`realpath $0`
#cp $SCRIPT $output_dir"/SCRIPT_SPH_PARTICLE.txt"