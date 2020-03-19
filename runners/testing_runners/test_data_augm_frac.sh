#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --time 0-05:20
#SBATCH -o data_aug_slurm.%N.%j.out
#SBAtCH -e data_aug_slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


module load Anaconda3
export QT_QPA_PLATFORM='offscreen'

# To be modified by user
echo 'starting virtual environment'
source activate $UPICKER_VENV_PATH

export dataset_table="/struct/mahamid/Irene/yeast/yeast_4bin_fas_single_filter_DA_G5_E0_R180.csv"

tomo_names="180426/045"

export write_on_table="true"
export segmentation_names='fas'

export data_aug_rounds=2
export rot_angle=180
export elastic_alpha=0
export sigma_noise=8
export epsilon=0.1

for tomo_name in $tomo_names
do
        export src_data_path="/struct/mahamid/Irene/yeast/healthy/"$tomo_name"/strongly_labeled_min0.01_max1/single_filter_64pix/full_partition.h5"
        export dst_data_path="/struct/mahamid/Irene/yeast/healthy/"$tomo_name"/strongly_labeled_min0.01_max1/single_filter_64pix/G5_E0_R180_DArounds4/full_partition.h5"

        echo "starting python script for "$tomo_name
        python3 $UPICKER_PATH/runners/testing_runners/test_data_augm.py -tomo_name $tomo_name -dataset_table $dataset_table -dst_data_path $dst_data_path -segmentation_names $segmentation_names -data_aug_rounds $data_aug_rounds -rot_angle $rot_angle -sigma_noise  $sigma_noise -elastic_alpha $elastic_alpha -src_data_path $src_data_path  -write_on_table $write_on_table -epsilon $epsilon
        echo "... done."
done








