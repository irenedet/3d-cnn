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


#/struct/mahamid/Shared/For_Irene/predictions/180426/004/motl_482_checked.csv
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/004/TM/motl_clean_4b.em
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/005/TM/motl_clean_4b.em
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/021/TM/motl_clean_4b.em
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/024/TM/motl_clean_4b.em
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/003/TM/motl_clean_4b.em
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/TM/motl_clean_4b.em
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/005/TM/motl_clean_4b.em
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/018/TM/motl_clean_4b.em
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/TM/motl_clean_4b.em
export output_dir="/scratch/trueba/3d-cnn/clean/180713_027/ribos/"
export path_to_motl="/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/TM/motl_clean_4b.em"
echo output_dir = $output_dir
echo path_to_motl = $path_to_motl
#export path_to_motl="/struct/mahamid/Shared/For_Irene/predictions/180426/004/motl_482_checked.csv"
#export path_to_motl="/scratch/trueba/3d-cnn/clean/180426_004/motl_clean_4b.em"
export z_shift=0  # shift between original tomogram and subtomogram of analysis
export shape_x=960
export shape_y=928
export shape_z=500
export radius=10

export coords_in_tom_format='True'


module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn

echo "starting to generate hdf of particles in the motl"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/generate_label_masks/generate_hdf_from_motl.py -motl $path_to_motl -output $output_dir -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -radius $radius -z_shift $z_shift -coords_in_tom_format $coords_in_tom_format
echo "...done."

# ... Finally:
#echo "Save a copy of this script for future reference"
#SCRIPT=`realpath $0`
#cp $SCRIPT $output_dir"/SCRIPT_SPH_PARTICLE.txt"