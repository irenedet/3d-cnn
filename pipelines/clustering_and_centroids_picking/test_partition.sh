#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 128G
#SBATCH --time 0-00:50
#SBATCH -o slurm_outputs/cnn_evaluation.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/cnn_evaluation.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

#SBAtCH -p gpu
#SBAtCH -C gpu=1080Ti
#SBAtCH --gres=gpu:1

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"

export QT_QPA_PLATFORM='offscreen'

export path_to_raw="/struct/mahamid/Irene/liang_data/00246_sq_df_sorted_deconv.hdf"
export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/liang_dataset/clusters/test/clustering_test_NO_DA_175-245ribo_D_2_IF_8_pr_radius_30/246/"
export label_name="test_clustering"
export box=128


export box_overlap=12
export output_h5_file_path=$output_dir'/partition.h5'

echo 'running python3 scripts: 1. Partitioning raw tomogram'
python3 particle_picking_pipeline/1_partition_tomogram.py -raw $path_to_raw -output $output_dir -outh5 $output_h5_file_path -box $box -overlap $box_overlap
echo '... done.'
