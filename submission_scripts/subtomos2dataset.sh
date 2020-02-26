#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 20G
#SBATCH --time 0-00:10
#SBATCH -o slurm_outputs/subtomos2dataset.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3

echo "Activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done."

export cluster_labels=False
export class_number=2
export output_shape=(928,928,500)
export box_length=128
export box_overlap=12
#reconstruction_type is either "prediction" or "labels" or "raw":
export reconstruction_type="prediction"
export label_name="cv_fractions_R_false_encoder_dropout_0.2_decoder_dropout_0.2_BN_false_DA_none_shuffle_true_frac_2_ribo_fas_memb__D_2_IF_8"
export dataset_type="cv_fractions"
TOMOS="190301/005"

count=0

for tomo_name in $TOMOS
do
    export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/cv_fractions/2"
    export subtomos_path="/scratch/trueba/3d-cnn/cnn_evaluation/yeast//190301/005/tomo_partition.h5"
    export output_path=$output_dir"/R_false_encoder_dropout_0.2_decoder_dropout_0.2_BN_false_DA_none_shuffle_true_frac_2_ribo_fas_memb__D_2_IF_8/"$tomo_name"/class_"$class_number"/prediction.hdf"
    echo "Reading file $subtomos_path"
    echo "Running python script"
    python3 $UPICKER_PATH/runners/subtomos2dataset_new.py -subtomos_path $subtomos_path -class_number $class_number -output_path $output_path -output_shape $output_shape -box_length $box_length -overlap $box_overlap -label_name $label_name -cluster_labels $cluster_labels -reconstruction_type $reconstruction_type
    echo "... done."
    count=$((count+1))
done


