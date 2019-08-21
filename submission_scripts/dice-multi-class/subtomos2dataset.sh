#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 0-02:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual env"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "...done."

export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn

tomo=195
label_name="NO_DA_ribo_D_2_IF_8"
subtomos_path="/scratch/trueba/liang_data/"$tomo"/train_and_test_partitions/full_partition.h5"
output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/"$label_name"/"$tomo
box=128
overlap_thickness=12
output_classes=1
output_xdim=928
output_ydim=928
output_zdim=450
#segmentation_names="ribo,fas,memb"
segmentation_names="ribo"





echo "starting python script"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/dice_multi-class/evaluation/subtomos2dataset.py -label_name $label_name -subtomos_path $subtomos_path -output_dir $output_dir -box $box -overlap_thickness $overlap_thickness -output_classes $output_classes -output_xdim $output_xdim -output_ydim $output_ydim -output_zdim $output_zdim -segmentation_names $segmentation_names
echo "... done."

