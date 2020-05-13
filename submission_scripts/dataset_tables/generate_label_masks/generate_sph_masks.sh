#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 128G
#SBATCH --time 0-0:30
#SBATCH -o slurm_outputs/generate_sph_mask.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/generate_sph_mask.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate $UPICKER_VENV_PATH
echo "... done"

export QT_QPA_PLATFORM='offscreen'
usage()

{
    echo "usage: [[ [-output output_dir][-test_partition test_partition ]
                  [-model path_to_model] [-label label_name]
                  [-out_h5 output_h5_file_path] [-conf conf]] | [-h]]"
}


while [ "$1" != "" ]; do
    case $1 in
        -dataset_table | --dataset_table )   shift
                                dataset_table=$1
                                ;;
        -tomo_name | --tomo_name )   shift
                                tomo_name=$1
                                ;;
        -output_dir | --output_dir )   shift
                                output_dir=$1
                                ;;
        -coords_in_tom_format | --coords_in_tom_format )   shift
                                coords_in_tom_format=$1
                                ;;
        -values_in_motl | --values_in_motl )   shift
                                values_in_motl=$1
                                ;;
        -class_name | --class_name )   shift
                                class_name=$1
                                ;;
        -sphere_radius | --sphere_radius )   shift
                                sphere_radius=$1
                                ;;
        -output_path | --output_path )   shift
                                output_path=$1
                                ;;
        -write_on_dataset_table | --write_on_dataset_table )   shift
                                write_on_dataset_table=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done



echo dataset_table = $dataset_table
echo tomo_name = $tomo_name
export dataset_table=$dataset_table
export tomo_name=$tomo_name
export coords_in_tom_format=$coords_in_tom_format
export values_in_motl=$values_in_motl
export output_dir=$output_dir
export write_on_dataset_table=$write_on_dataset_table
export class_name=$class_name
export sphere_radius=$sphere_radius
export output_path=$output_path


echo "starting python script:"
python3 $UPICKER_PATH/runners/dataset_tables/generate_mask/generate_mask_from_motl.py -tomo_name $tomo_name -dataset_table $dataset_table -class_name $class_name -coords_in_tom_format $coords_in_tom_format -radius $sphere_radius -output_path $output_path -values_in_motl $values_in_motl -write_on_dataset_table $write_on_dataset_table
echo "... done."

