#! /bin/bash

#SBATCH -A zaugg
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 0-00:05
#SBATCH -o slurm_outputs/generate_sph_mask.%N.%j.out
#SBAtCH -e slurm_outputs/generate_sph_mask.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"
export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn


TOMO_NAMES="181119/002
181119/030
181126/002
181126/012
181126/025"
export output_dir="/struct/mahamid/Irene/yeast/ED"

#TOMO_NAMES="180426/021
#180426/024"
#180426/005
#180426/021
#180426/024"
#export output_dir="/struct/mahamid/Irene/yeast/healthy"




export coords_in_tom_format=true
export class_number=1
for tomo in $TOMO_NAMES
do
    echo tomo=$tomo
    export hdf_output_path=$output_dir/$tomo"/clean_masks/class_$class_number/corrected_motl_191108_sph_mask.hdf"

    if [ $class_number == 0 ]; then
        echo "class_number is 0"
        export path_to_motl=$path_to_motl_clean_0
        export radius=8
    elif [ $class_number == 1 ]; then
        echo "class_number is 1"
        export path_to_motl=$path_to_motl_clean_1
        export radius=10
    else
        echo "class_number non-supported for now"
    fi

    export z_shift=0  # shift between original tomogram and subtomogram of analysis
    export shape_x=928
    export shape_y=928
    export shape_z=500
    export values_in_motl=false
    export path_to_motl=$output_dir/$tomo"/fas/motl/corrected_motl_191108.csv"
    echo path_to_motl = $path_to_motl

    echo "starting to generate hdf of particles in the motl"
    python3 pipelines/generate_label_masks/generate_hdf_from_motl.py -motl $path_to_motl -hdf_output_path $hdf_output_path -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -radius $radius -z_shift $z_shift -coords_in_tom_format $coords_in_tom_format -values_in_motl $values_in_motl
    echo "...done."
done
