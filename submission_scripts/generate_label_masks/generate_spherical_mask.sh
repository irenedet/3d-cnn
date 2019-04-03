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
#source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse


export output_dir="/struct/mahamid/Irene/predictions/180426/004"
export path_to_motl="/struct/mahamid/Shared/For_Irene/predictions/180426/004/motl_482_checked.csv"

export z_shift=-380  # shift between original tomogram and subtomogram of analysis
export shape_x=928
export shape_y=928
export shape_z=221
export radius=8

export coords_in_tom_format=True

echo "starting to generate hdf of undetected particles"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/generate_label_masks/generate_hdf_from_motl.py -motl $path_to_motl -output $output_dir -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -radius $radius -z_shift $z_shift -coords_in_tom_format $coords_in_tom_format
echo "...done."

