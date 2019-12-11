#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 20G
#SBATCH --time 0-02:30
#SBATCH -o generate_training_partition.slurm.%N.%j.out
#SBAtCH -e generate_training_partition.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load GCC
module load Anaconda3
echo 'activating virtual environment'
source activate $UPICKER_VENV_PATH
echo '... done.'

#TOMOS="190218/043
#190218/044
#190218/048
#190218/049
#190218/050
#190218/051
#190218/052
#190218/054
#190218/056
#190218/059
#190218/060
#190218/061
#190218/062
#190218/063
#190218/064
#190218/065
#190218/066
#190218/067
#190218/068
#190218/069
#190218/070
#190218/071
#190218/072"
#TOMOS="190218/073
#190218/075
#190218/076
#190218/077
#190218/078
#190218/081
#190218/082
#190218/083
#190218/084
#190218/085
#190218/086
#190218/087
#190218/088
#190218/089
#190218/090
#190218/091
#190218/092
#190218/093
#190218/094
#190218/095
#190218/096
#190218/097
#190218/098
#190218/099"
#TOMOS="190218/100
#190218/101
#190218/102
#190218/103
#190218/104
#190218/105
#190218/106
#190218/108
#190218/110
#190218/111
#190218/112
#190218/113
#190218/114
#190218/115
#190218/116
#190218/117
#190218/118
#190218/119
#190218/120
#190218/121"
#TOMOS="190218/122
#190218/123
#190218/124
#190218/125
#190223/129
#190223/130
#190223/131
#190223/132
#190223/133
#190223/134
#190223/135
#190223/136
#190223/139
#190223/140
#190223/141
#190223/142
#190223/143
#190223/144
#190223/145
#190223/146
#190223/148"
#TOMOS="190223/149
#190223/151
#190223/152
#190223/153
#190223/154
#190223/155
#190223/156
#190223/157
#190223/159
#190223/160
#190223/162
#190223/163
#190223/165
#190223/166
#190223/168
#190223/169
#190223/171
#190223/172
#190223/173
#190223/174"
#TOMOS="190223/175
#190223/176
#190223/177
#190223/178
#190223/179
#190223/180
#190223/181
#190223/182
#190223/183
#190223/184
#190223/185
#190223/186
#190223/187
#190223/188
#190223/189
#190223/190
#190223/191
#190223/192
#190223/194"
#
#export path_to_dataset_table="/struct/mahamid/Irene/NPC/SPombe/npc_gauss_0.06_masks/NPC_SU_gauss0.06_masks_table.csv"
#export global_output_dir="/scratch/trueba/3d-cnn/cnn-evaluation/SPombe_NPC_SU/npc_gauss_0.06_masks"
#export write_on_table='true'
## Training set parameters:
#export segmentation_names='npc' #separated by commas
#export split=0.7
#export box_length=128
#export number_iterations_data_aug=0
#export min_label_fraction=0.002
## Data for the new model
#
## Data for old models for resuming training:
#
#for tomo_name in $TOMOS
#do
#    export output_dir=$global_output_dir"/"$tomo_name"/strongly_labeled_$min_label_fraction"
#    mkdir -p $output_dir
#    echo 'Generating training partition for dataset' $tomo_name
#    python3 runners/dataset_tables/generate_training_partitions/generate_train_and_test_partitions_multi_label_files.py  -output $output_dir -tomo_name $tomo_name -dataset_table $path_to_dataset_table -split $split -segmentation_names $segmentation_names -box $box_length -number_iter $number_iterations_data_aug -write_on_table $write_on_table -min_label_fraction $min_label_fraction
#    echo "... done."
#done

#180426/004
#180426/005
#180426/021
#180426/024
#181119/002
#181119/030
#181126/002
#TOMOS="181126/012"
TOMOS="181126/025"

export path_to_dataset_table="/struct/mahamid/Irene/yeast/yeast_table_no_eman.csv"
export global_output_dir="/scratch/trueba/3d-cnn/cross-validation/original-training-data/"
export write_on_table='true'
# Training set parameters:
export segmentation_names='ribo,fas,memb' #separated by commas
export split=0.7
export box_length=256
export number_iterations_data_aug=0
export min_label_fraction=0.002
# Data for the new model

# Data for old models for resuming training:

for tomo_name in $TOMOS
do
    export output_dir=$global_output_dir"/"$tomo_name"/strongly_labeled_"$min_label_fraction"/no_eman_filter_256pix"
    mkdir -p $output_dir
    echo 'Generating training partition for dataset' $tomo_name
    python3 runners/dataset_tables/generate_training_partitions/generate_train_and_test_partitions_multi_label_files.py  -output $output_dir -tomo_name $tomo_name -dataset_table $path_to_dataset_table -split $split -segmentation_names $segmentation_names -box $box_length -number_iter $number_iterations_data_aug -write_on_table $write_on_table -min_label_fraction $min_label_fraction
    echo "... done."
done
#
