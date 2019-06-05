#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --time 0-00:20
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

export tomo_number=6
export j="06"

export path_to_raw="/struct/mahamid/Irene/NPC/"$j"/"$j"_dose-filt_eman2filter.hdf"
export labels_dataset="/struct/mahamid/Irene/NPC/"$j"/clean/thin_cylindrical_mask.hdf"
export output_dir="/struct/mahamid/Irene/NPC/"$j"/3_ALL_DA_test/"



export path_to_full_motl="/scratch/trueba/data/NPC/allmotl_bin4.txt"
export label_name="npc"
export shape_x=928
export shape_y=928
export shape_z=300

#export segmentation_names="ribo,fas,memb"
export split=-1  # Between partitions of testing and training data
export box_side=160

# For data augmentation:
export number_iter=10
export overlap=12


module load Anaconda3
echo 'starting virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/

export PYTHONPATH=/g/scb2/zaugg/trueba/3d-cnn

echo 'starting python script'
python3 ./pipelines/dice_multi-class/generate_training_set/generate_partition_from_motl_single_label.py -tomo_number $tomo_number -raw $path_to_raw  -labeled $labels_dataset -motl $path_to_full_motl -label $label_name -output $output_dir -box $box_side -shapex $shape_x -shapey $shape_y -shapez $shape_z -number_iter $number_iter -split $split -overlap $overlap
echo 'done'