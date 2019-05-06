#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 0-00:10
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


# ToDo check why this does not run in cluster
module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse


export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn

export output_dir="/scratch/trueba/3d-cnn/cnn_evaluation/uni-class-9TOMOS/confs_D5_IF4_021"
export path_to_csv_motl=$(ls $output_dir/motl*)
export lamella_file="/scratch/trueba/3d-cnn/clean/180426_021/021_lamellamask.hdf"
export border_xy=20
export lamella_extension=40
export shape_x=959
export shape_y=927
export shape_z=1000
export z_shift=0

echo "filtering points in lamella"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/performance_nnet_quantification/filter_with_lamella_mask.py -csv_motl $path_to_csv_motl -lamella_file $lamella_file -output_dir $output_dir -border_xy $border_xy -lamella_extension $lamella_extension -x_dim $shape_x -y_dim $shape_y -z_dim $shape_z -z_shift $z_shift
echo "...done."
