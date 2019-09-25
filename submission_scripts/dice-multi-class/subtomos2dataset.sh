#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 0-00:30
#SBATCH -o slurm_outputs/subtomo2dataset.%N.%j.out
#SBAtCH -e slurm_outputs/subtomo2dataset.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual env"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "...done."

export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn


#TOMOS="181119/002
#181119/030"
TOMOS="190301/009"

box=128
overlap_thickness=12
output_classes=3
output_xdim=927
output_ydim=927
output_zdim=500
segmentation_names="ribo,fas,memb"
label_name="Retrain_retrained_D4_IF8_NA_except_180711_003"

for tomo in $TOMOS
do
    echo $tomo
    subtomos_path="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/ED_DEFOCUS/"$tomo"/tomo_partition.h5"
    #echo $(ls $subtomos_path)
    output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/"$label_name"/"$tomo
    echo "starting python script"
    python3 pipelines/dice_multi-class/evaluation/subtomos2dataset.py -label_name $label_name -subtomos_path $subtomos_path -output_dir $output_dir -box $box -overlap_thickness $overlap_thickness -output_classes $output_classes -output_xdim $output_xdim -output_ydim $output_ydim -output_zdim $output_zdim -segmentation_names $segmentation_names
    echo "... done."
done







