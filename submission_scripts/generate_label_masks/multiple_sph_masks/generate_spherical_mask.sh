#! /bin/bash

#SBATCH -A zaugg
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 64G
#SBATCH --time 0-00:40
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"
export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn


TOMO_NAMES=( "181119_002"
             "181119_030"
             "181126_002"
             "181126_012"
             "181126_025"
             "190301_005" )


export coords_in_tom_format='True'
export class_number=0

for tomo in ${TOMO_NAMES[@]}; do
    echo tomo=$tomo
    export hdf_output_path="/struct/mahamid/Irene/yeast/ED/"$tomo"/clean_masks/class_$class_number/spherical_mask.hdf"
    export parameters_file="/g/scb2/zaugg/trueba/3d-cnn/submission_data/dice_multilabel/ED_TOMOS/"$tomo".sh"
    source $parameters_file

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

    export z_shift=$z_shift  # shift between original tomogram and subtomogram of analysis
    export shape_x=$input_xdim
    export shape_y=$input_ydim
    export shape_z=$input_zdim
    echo path_to_motl = $path_to_motl

    echo "starting to generate hdf of particles in the motl"
    python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/generate_label_masks/generate_hdf_from_motl.py -motl $path_to_motl -hdf_output_path $hdf_output_path -shape_x $shape_x -shape_y $shape_y -shape_z $shape_z -radius $radius -z_shift $z_shift -coords_in_tom_format $coords_in_tom_format
    echo "...done."
done

# ... Finally:
#echo "Save a copy of this script for future reference"
#SCRIPT=`realpath $0`
#cp $SCRIPT $output_dir"/SCRIPT_SPH_PARTICLE.txt"