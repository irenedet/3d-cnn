#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 10G
#SBATCH --time 0-00:10
#SBATCH -o slurm_outputs/subtomos2dataset.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "Activating virtual environment"
source activate $UPICKER_VENV_PATH
echo "... done."

export dataset_table="/struct/mahamid/Irene/cross-validation/multiclass/CV_data.csv"
export cluster_labels=False
export class_number=0
export box_length=128
export box_overlap=12
#reconstruction_type is either "prediction", "raw", or "labels":
export reconstruction_type="labels"
export label_name="fractions"
export global_output_dir="/scratch/trueba/3d-cnn/cross-validation/original-training-data/180426/004/strongly_labeled_0.002"
#TOMOS="190223/132
#190223/148
#190223/178
#190223/183
#190223/177
#190223/190
#190223/191
#190223/192
#190223/193
#190223/194
#190218/044"

for tomo_name in $TOMOS
do
    export output_dir=$global_output_dir
    mkdir -p $output_dir
    export output_path=$output_dir"/prediction.hdf"
    rm $output_path
	  echo "reconstructing prediction for $tomo_name"
	  echo "Running python script"
    python3 $UPICKER_PATH/runners/dataset_tables/subtomos2datasets/subtomos2dataset.py -dataset_table $dataset_table -tomo_name $tomo_name -class_number $class_number -output_path $output_path -box_length $box_length -overlap $box_overlap -label_name $label_name -cluster_labels $cluster_labels -reconstruction_type $reconstruction_type
    echo "... done."
done