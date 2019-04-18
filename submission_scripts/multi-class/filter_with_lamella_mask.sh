#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --time 0-04:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

export lamella_file="/scratch/trueba/3d-cnn/clean/180426_004/004_lamellamask_subtomo.hdf"
export csv_motl="/scratch/trueba/test/motl/motl_3852_class_2.csv"
export output_dir="/scratch/trueba/test/motl/fas"
export dataset_border_xy=20
export lamella_extension=40 # because lamella mask does not cover all the region of interest
export x_dim=928
export y_dim=928
export z_dim=221
export z_shift=380


export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn

module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse


echo 'starting python3 script'
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/performance_nnet_quantification/filter_with_lamella_mask.py -lamella_file $lamella_file -csv_motl $csv_motl -output_dir $output_dir -border_xy $dataset_border_xy -lamella_extension $lamella_extension -x_dim $x_dim -y_dim $y_dim -z_dim $z_dim -z_shift $z_shift

