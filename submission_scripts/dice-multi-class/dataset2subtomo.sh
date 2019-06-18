#!/usr/bin/env bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 32G
#SBATCH --time 0-05:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

#export segmentation_names="ribo,fas,memb"
export box_side=128
export overlap=12

module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

export PYTHONPATH=/g/scb2/zaugg/trueba/3d-cnn

RAW="/struct/mahamid/Irene/yeast/ED/181119_030/eman_filt_neg_030_sq_df_sorted.hdf"
#/struct/mahamid/Irene/yeast/ED/181119_002/eman_filt_eman_filt_002_sq_df_sorted.hdf
#/struct/mahamid/Irene/yeast/ED/181119_030/eman_filt_eman_filt_030_sq_df_sorted.hdf
#/struct/mahamid/Irene/yeast/ED/181126_002/eman_filt_eman_filt_002_sq_df_sorted.hdf
#/struct/mahamid/Irene/yeast/ED/181126_012/eman_filt_eman_filt_012_sq_df_sorted.hdf
#/struct/mahamid/Irene/yeast/ED/181126_025/eman_filt_eman_filt_025_sq_df_sorted.hdf"
#/struct/mahamid/Irene/yeast/ED/190301_005/eman_filt_eman_filt_005_sq_df_sorted.hdf
#"

for path_to_raw in $RAW
do
    export output_dir="/struct/mahamid/Irene/yeast/ED/"${path_to_raw:31:10}"/"
    export outh5=$output_dir"/eman_filt_tomo_partition.h5"
    echo 'starting python script'
    python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/dice_multi-class/particle_picking_pipeline/1_partition_tomogram.py -raw $path_to_raw -outh5 $outh5 -output $output_dir -box $box_side -overlap $overlap
    echo 'done'
done