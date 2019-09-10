#! /bin/bash

#SBATCH -A zaugg
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 64G
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

export hdf_output_path="/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/clusters/test/clustering_test_NO_DA_ribo_D_2_IF_8_pr_radius_20/full_dataset/246/class_0/combined_motl_1.5sph/big_clusters_motl_mask.hdf"
export path_to_motl="/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/clusters/test/clustering_test_NO_DA_ribo_D_2_IF_8_pr_radius_20/full_dataset/246/class_0/combined_motl_1.5sph/big_clusters_motl.csv"
export radius=20
export z_shift=0
export shape_x=928
export shape_y=928
export shape_z=450
export values_in_motl=False #True if the motl peak scores = value at spheres

echo "starting to generate hdf of particles in the motl"
python3 ./pipelines/generate_label_masks/generate_hdf_from_motl.py -motl $path_to_motl -hdf_output_path $hdf_output_path -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -radius $radius -z_shift $z_shift -coords_in_tom_format $coords_in_tom_format -values_in_motl $values_in_motl
echo "...done."

# ... Finally:
#echo "Save a copy of this script for future reference"
#SCRIPT=`realpath $0`
#cp $SCRIPT $output_dir"/SCRIPT_SPH_PARTICLE.txt"