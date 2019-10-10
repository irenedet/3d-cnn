#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 40G
#SBATCH --time 0-00:30
#SBATCH -o slurm_outputs/subtomos2dataset.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3

echo "Activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done."

#export subtomos_path="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_005/tomo_partition.h5"
#export output_path="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180711_005/class_0/predicted_segmentation.hdf"
export cluster_labels=False
export class_number=0
export output_shape=(928,928,400)
export box_length=128
export box_overlap=12
#reconstruction_type is either "prediction" or "labels":
export reconstruction_type="raw"
export label_name="none"
export output_dir="/struct/mahamid/Irene/check_raw_data190301/"
mkdir -p $output_dir
DIRS="/struct/mahamid/Irene/yeast/ED//190301/003/tomo_partition.h5"

for dir in $DIRS
do
    export subtomos_path=$dir
#    export output_path=$output_dir"/class_"$class_number"/prediction.hdf"
    export output_path=$output_dir"/raw_reconstruction003newtrain_part.hdf"
	  echo "Reading file $subtomos_path"
	  echo "Running python script"
    python3 runners/subtomos2dataset_new.py -subtomos_path $subtomos_path -class_number $class_number -output_path $output_path -output_shape $output_shape -box_length $box_length -overlap $box_overlap -label_name $label_name -cluster_labels $cluster_labels -reconstruction_type $reconstruction_type
    echo "... done."
done


