#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 38G
#SBATCH --time 0-05:50
#SBATCH -o slurm_outputs/data_aug_slurm.%N.%j.out
#SBAtCH -e slurm_outputs/data_aug_slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


module load Anaconda3
export QT_QPA_PLATFORM='offscreen'

# To be modified by user
echo 'starting virtual environment'
source activate $UPICKER_VENV_PATH

export dataset_table="/struct/mahamid/Irene/cross-validation/multiclass/DA_G1.5_E2_R180_DArounds4/DA_CV_data.csv"

fractions="0 1 2 3 4"

#export tomo_names="180426/004 180426/005 180426/021 181119/002"
export tomo_names="181119/030 181126/002 181126/012"


export write_on_table=true
export segmentation_names='ribo_fas_memb'

export data_aug_rounds=4
export rot_angle=180
export elastic_alpha=2
export sigma_noise=1.5

for tomo_name in $tomo_names
do
    for fraction in $fractions
    do
        export src_data_path="/struct/mahamid/Irene/cross-validation/multiclass/original-training-data/"$tomo_name"/strongly_labeled_0.002/fraction_"$fraction".h5"
        export dst_data_path="/struct/mahamid/Irene/cross-validation/multiclass/DA_G1.5_E2_R180_DArounds4/original-training-data/"$tomo_name"/strongly_labeled_0.002/DA_fraction_"$fraction".h5"

        echo "starting python script for "$tomo_name" and "$fraction
        python3 $UPICKER_PATH/runners/testing_runners/test_data_augm.py -tomo_name $tomo_name -dataset_table $dataset_table -dst_data_path $dst_data_path -segmentation_names $segmentation_names -data_aug_rounds $data_aug_rounds -rot_angle $rot_angle -sigma_noise  $sigma_noise -elastic_alpha $elastic_alpha -src_data_path $src_data_path  -write_on_table $write_on_table
        echo "... done."
    done
done











