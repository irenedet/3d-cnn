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

export dataset_table="/struct/mahamid/Irene/yeast/yeast_table.csv"
export cluster_labels=False
export class_number=2
export box_length=128
export box_overlap=12
#reconstruction_type is either "prediction", "raw", or "labels":
export reconstruction_type="prediction"
export semantic_names="ribo,fas,memb"
export model_nickname="R_false_encoder_dropout_0.2_decoder_dropout_0.2_BN_false_DA_none_shuffle_true_frac_2_ribo_fas_memb__D_2_IF_8"
export label_name="cv_fractions_"$model_nickname
export global_output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/cv_fractions/2/"$model_nickname"/"

#TOMOS="190301/003"
TOMOS="190301/005
190301/009
190301/012
190301/016
190301/022
190301/028
190301/031
190301/032
190301/033
190301/035
190301/037
190301/043
190301/045"

for tomo_name in $TOMOS
do
    export output_dir=$global_output_dir"/"$tomo_name"/class_"$class_number
    mkdir -p $output_dir
    export output_path=$output_dir"/prediction.hdf"
#    rm $output_path
	  echo "reconstructing prediction for $tomo_name"
	  echo "Running python script"
    python3 $UPICKER_PATH/runners/dataset_tables/subtomos2datasets/subtomos2dataset.py -semantic_names $semantic_names -dataset_table $dataset_table -tomo_name $tomo_name -class_number $class_number -output_path $output_path -box_length $box_length -overlap $box_overlap -label_name $label_name -cluster_labels $cluster_labels -reconstruction_type $reconstruction_type
    echo "... done."
done