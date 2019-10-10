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

export dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
export cluster_labels=False
export class_number=0
export box_length=128
export box_overlap=12
#reconstruction_type is either "prediction" or "labels":
export reconstruction_type="prediction"
export label_name="fas_fractions_004_005_021_ED_and_def_shuffle_false_frac_3_fas__D_1_IF_12"
export global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/yeast_dataset/"$label_name"/peak_calling/pr_radius_10"
#TOMOS="190301/003 190301/022 190301/035"
#frac_3_fas__D_1_IF_8
TOMOS="181119/002 181119/030 181126/002 181126/012 181126/025"
#_frac_3_fas__D_1_IF_12

#180426/026
#180426/027
#180426/028
#180426/029
#180426/030
#180426/034
#180426/037
#180426/041
#180426/043
#180426/045"

for tomo_name in $TOMOS
do
    export output_dir=$global_output_dir/$tomo_name
    mkdir -p $output_dir
    export output_path=$output_dir"/prediction.hdf"
	  echo "reconstructing prediction for $tomo_name"
	  echo "Running python script"
    python3 runners/dataset_tables/subtomos2datasets/subtomos2dataset.py -dataset_table $dataset_table -tomo_name $tomo_name -class_number $class_number -output_path $output_path -box_length $box_length -overlap $box_overlap -label_name $label_name -cluster_labels $cluster_labels -reconstruction_type $reconstruction_type
    echo "... done."
done