#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 8G
#SBATCH --time 0-00:20
#SBATCH -o slurm_outputs/data_aug_slurm.%N.%j.out
#SBAtCH -e slurm_outputs/data_aug_slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


module load Anaconda3
export QT_QPA_PLATFORM='offscreen'

# To be modified by user
echo 'starting virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/

export dataset_table="/struct/mahamid/Irene/NPC/SPombe/npc_gauss_0.06_0.01_masks/DA_NPC_SU_gauss0.06_0.01_masks_table.csv"

tomo_names="190218/044"


export write_on_table=true
export segmentation_names='npc'

export data_aug_rounds=4
export rot_angle=180
export elastic_alpha=2
export sigma_noise=1.5

for tomo_name in $tomo_names
do
        export src_data_path="/scratch/trueba/3d-cnn/SPombe_NPC_SU/npc_gauss_0.06_0.01_masks/"$tomo_name"/training_data/strongly_labeled_0.02/full_partition.h5"
        export dst_data_path="/scratch/trueba/3d-cnn/SPombe_NPC_SU/npc_gauss_0.06_0.01_masks/"$tomo_name"/training_data/strongly_labeled_0.02/G"$sigma_noise"_E"$elastic_alpha"_R"$rot_angle"_DArounds"$data_aug_rounds"/full_partition.h5"

        echo "starting python script for "$tomo_name
        python3 /g/scb2/zaugg/trueba/3d-cnn/runners/testing_runners/test_data_augm.py -tomo_name $tomo_name -dataset_table $dataset_table -dst_data_path $dst_data_path -segmentation_names $segmentation_names -data_aug_rounds $data_aug_rounds -rot_angle $rot_angle -sigma_noise  $sigma_noise -elastic_alpha $elastic_alpha -src_data_path $src_data_path  -write_on_table $write_on_table
        echo "... done."
done











