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
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
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
        -class_number | --class_number )   shift
                                class_number=$1
                                ;;
        -coords_in_tom_format | --coords_in_tom_format )   shift
                                coords_in_tom_format=$1
                                ;;
        -values_in_motl | --values_in_motl )   shift
                                values_in_motl=$1
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
export class_number=$class_number
export write_on_dataset_table=$write_on_dataset_table

if [ $class_number == 0 ]; then
    echo "class_number is 0"
    export class_name="ribo"
    export sphere_radius=8
    export hdf_output_path=$output_dir"/"$tomo_name"/clean_masks/ribo_sph_mask.hdf"
elif [ $class_number == 1 ]; then
    echo "class_number is 1"
    export class_name="fas"
    export sphere_radius=10
    export hdf_output_path=$output_dir"/"$tomo_name"/clean_masks/fas_sph_mask.hdf"
else
    echo "class_number non-supported for now"
fi

echo "starting python script:"
python3 ./runners/dataset_tables/generate_mask/generate_hdf_from_motl.py -tomo_name $tomo_name -dataset_table $dataset_table -class_name $class_name -coords_in_tom_format $coords_in_tom_format -radius $sphere_radius -hdf_output_path $hdf_output_path -values_in_motl $values_in_motl -write_on_dataset_table $write_on_dataset_table
echo "... done."

