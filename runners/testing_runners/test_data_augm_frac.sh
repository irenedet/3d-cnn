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
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/

export dataset_table="/struct/mahamid/Irene/fractions/fas_DA_fractions_data.csv"

fractions="0
1
2
3
4"

export tomo_names="181119/002 181119/030 181126/002 181126/012 181126/025"


export write_on_table=false
export segmentation_names='fas'

export data_aug_rounds=3
export rot_angle=90
export elastic_alpha=2
export sigma_noise=1.5
#export src_data_path="/struct/mahamid/Irene/yeast/healthy/"$tomo_name"/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5"

for tomo_name in $tomo_names
do
    for fraction in $fractions
    do
        export src_data_path="/struct/mahamid/Irene/yeast/ED/"$tomo_name"/fas_class/train_and_test_partitions/fraction_"$fraction".h5"
        export dst_data_path="/struct/mahamid/Irene/yeast/ED/"$tomo_name"/fas_class/train_and_test_partitions/G"$sigma_noise"_E"$elastic_alpha"_R"$rot_angle"/fraction_"$fraction".h5"

        echo "starting python script for "$tomo_name" and "$fraction
        python3 /g/scb2/zaugg/trueba/3d-cnn/runners/testing_runners/test_data_augm.py -tomo_name $tomo_name -dataset_table $dataset_table -dst_data_path $dst_data_path -segmentation_names $segmentation_names -data_aug_rounds $data_aug_rounds -rot_angle $rot_angle -sigma_noise  $sigma_noise -elastic_alpha $elastic_alpha -src_data_path $src_data_path  -write_on_table $write_on_table
        echo "... done."
    done
done











