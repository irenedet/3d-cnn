#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 128G
#SBATCH --time 0-2:00
#SBATCH -o slurm_outputs/TM_at_peaks.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/TM_at_peaks.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"
export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn


export QT_QPA_PLATFORM='offscreen'
usage()

{
    echo "usage: [[ [-output output_dir][-test_partition test_partition ]
                  [-model path_to_model] [-label label_name]
                  [-out_h5 output_h5_file_path] [-conf conf]] | [-h]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -tomo_name | --tomo_name )   shift
                                tomo_name=$1
                                ;;
        -path_to_motl | --path_to_motl )   shift
                                path_to_motl=$1
                                ;;
        -path_to_dataset | --path_to_dataset )   shift
                                path_to_dataset=$1
                                ;;
        -path_to_output_csv | --path_to_output_csv )   shift
                                path_to_output_csv=$1
                                ;;
        -catalogue_path | --catalogue_path )   shift
                                catalogue_path=$1
                                ;;
        -ref_angles | --ref_angles )   shift
                                ref_angles=$1
                                ;;
        -angles_in_degrees | --angles_in_degrees )   shift
                                angles_in_degrees=$1
                                ;;
        -path_to_mask | --path_to_mask )   shift
                                path_to_mask=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


echo tomo_name=$tomo_name
echo path_to_motl=$path_to_motl
echo path_to_dataset=$path_to_dataset
echo path_to_output_csv=$path_to_output_csv
echo catalogue_path=$catalogue_path
echo angles_in_degrees=$angles_in_degrees
echo ref_angles=$ref_angles
echo path_to_mask=$path_to_mask

export tomo_name=$tomo_name
export path_to_motl=$path_to_motl
export path_to_dataset=$path_to_dataset
export path_to_output_csv=$path_to_output_csv
export catalogue_path=$catalogue_path
export angles_in_degrees=$angles_in_degrees
export ref_angles=$ref_angles
export path_to_mask=$path_to_mask

echo "Starting python script..."
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/template_matching_at_peaks/TM_from_motl.py -tomo_name $tomo_name -path_to_motl $path_to_motl -path_to_dataset $path_to_dataset -path_to_output_csv $path_to_output_csv -catalogue_path $catalogue_path -path_to_mask $path_to_mask -ref_angles $ref_angles -angles_in_degrees $angles_in_degrees
echo "...done."