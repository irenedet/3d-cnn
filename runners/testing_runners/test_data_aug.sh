#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 68G
#SBATCH --time 0-02:50
#SBATCH -o slurm_outputs/data_aug_slurm.%N.%j.out
#SBAtCH -e slurm_outputs/data_aug_slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


module load Anaconda3
export QT_QPA_PLATFORM='offscreen'

# To be modified by user
echo 'starting virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/

export dataset_table="/struct/mahamid/Irene/yeast/npc/npc_DA_data.csv"

tomo_names="180413/006
180413/007
180426/005
180426/006
180426/008
180426/014
180426/027
180426/037
180426/038
180426/040
180426/043
180711/005
180711/006
180711/007
180711/012
180711/017
180711/022
180713/002
180713/005
180713/007
180713/015
180713/018
180713/020
180713/025
180713/027
180713/031
180713/035
180713/037
180713/039
180713/041
180713/043
180713/050"

export write_on_table=true
export segmentation_names='npc'

export data_aug_rounds=4
export rot_angle=180
export elastic_alpha=2
export sigma_noise=1.5
#export src_data_path="/struct/mahamid/Irene/yeast/healthy/"$tomo_name"/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5"

for tomo_name in $tomo_names
do
        export src_data_path="/struct/mahamid/Irene/yeast/healthy/"$tomo_name"/npc_class/strongly_labeled0.02/train_and_test_partitions/full_partition.h5"
        export dst_data_path="/struct/mahamid/Irene/yeast/healthy/"$tomo_name"/npc_class/strongly_labeled0.02/train_and_test_partitions/G"$sigma_noise"_E"$elastic_alpha"_R"$rot_angle"/full_partition.h5"

        echo "starting python script for "$tomo_name" and "$fraction
        python3 /g/scb2/zaugg/trueba/3d-cnn/runners/testing_runners/test_data_augm.py -tomo_name $tomo_name -dataset_table $dataset_table -dst_data_path $dst_data_path -segmentation_names $segmentation_names -data_aug_rounds $data_aug_rounds -rot_angle $rot_angle -sigma_noise  $sigma_noise -elastic_alpha $elastic_alpha -src_data_path $src_data_path  -write_on_table $write_on_table
        echo "... done."
done











