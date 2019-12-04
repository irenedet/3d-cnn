#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 8G
#SBATCH --time 0-02:20
#SBATCH -o slurm_outputs/data_aug_slurm.%N.%j.out
#SBAtCH -e slurm_outputs/data_aug_slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


module load Anaconda3
export QT_QPA_PLATFORM='offscreen'

# To be modified by user
echo 'starting virtual environment'
source activate $UPICKER_VENV_PATH

export dataset_table="/struct/mahamid/Irene/NPC/SPombe/npc_gauss_0.06_masks/DA_NPC_SU_gauss0.06_masks_table.csv"

##tomo_names="190218/061 190218/081 190218/043 190218/048 190218/049 190218/051
##190218/052 190218/054 190218/056 190218/059"
#tomo_names="190218/066 190218/067 190218/068 190218/069 190218/070 190218/071 190218/072
#190218/073 190218/075 190218/076 190218/078"
#tomo_names="190218/094 190218/095 190218/096 190218/097 190218/098 190218/099 190218/101
#190218/102 190218/103 190218/104 190218/105"
#tomo_names="190218/120 190218/121 190218/122 190218/123 190218/124 190218/125 190223/129
#190223/130 190223/131 190223/133 190223/135 190223/136 190223/139"
#tomo_names="190223/151 190223/152 190223/153 190223/154 190223/155 190223/156 190223/157
#190223/159 190223/160 190223/162 190223/163 190223/165 190223/166"
#tomo_names="190223/179 190223/180 190223/181"
#tomo_names="190223/182 190223/184 190223/185 190223/186
#190223/187 190223/188 190223/189 190218/060 190218/063"

#tomo_names="190218/062 190218/064 190218/065 190218/077 190218/100
#190218/050 190218/090 190218/106 190218/083 190218/113"
#tomo_names="190223/140
#190223/141 190223/142 190223/143 190223/144 190223/145 190223/146 190223/149"
#tomo_names="190223/168
#190223/169 190223/171 190223/172 190223/173 190223/174 190223/175 190223/176"
#tomo_names="190218/108 190218/110 190218/111
#190218/112 190218/114 190218/115 190218/116 190218/117 190218/118 190218/119"
tomo_names="190218/082 190218/084 190218/085
190218/086 190218/087 190218/088 190218/089 190218/091 190218/092 190218/093"

export write_on_table=true
export segmentation_names='npc'

export data_aug_rounds=4
export rot_angle=180
export elastic_alpha=2
export sigma_noise=1.5

for tomo_name in $tomo_names
do
        export src_data_path="/scratch/trueba/3d-cnn/cnn-evaluation/SPombe_NPC_SU/npc_gauss_0.06_masks/"$tomo_name"/strongly_labeled_0.002/full_partition.h5"
        export dst_data_path="/scratch/trueba/3d-cnn/cnn-evaluation/SPombe_NPC_SU/npc_gauss_0.06_masks/"$tomo_name"/strongly_labeled_0.002/G"$sigma_noise"_E"$elastic_alpha"_R"$rot_angle"_DArounds"$data_aug_rounds"/full_partition.h5"

        echo "starting python script for "$tomo_name
        python3 $UPICKER_PATH/runners/testing_runners/test_data_augm.py -tomo_name $tomo_name -dataset_table $dataset_table -dst_data_path $dst_data_path -segmentation_names $segmentation_names -data_aug_rounds $data_aug_rounds -rot_angle $rot_angle -sigma_noise  $sigma_noise -elastic_alpha $elastic_alpha -src_data_path $src_data_path  -write_on_table $write_on_table
        echo "... done."
done








