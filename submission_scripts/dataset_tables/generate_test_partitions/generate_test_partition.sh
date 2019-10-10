#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 20G
#SBATCH --time 0-00:40
#SBATCH -o slurm_outputs/partition_dataset.slurm.%N.%j.out
#SBAtCH -e slurm_outputs/partition_dataset.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load GCC
module load Anaconda3
echo 'activating virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo '... done.'

export PYTHONPATH=/g/scb2/zaugg/trueba/3d-cnn

TOMOS="190301/003"

export path_to_dataset_table="/struct/mahamid/Irene/yeast/fas_yeast_table.csv"
export global_output_dir="/struct/mahamid/Irene/yeast/ED"
export box_length=128
export overlap=12
export write_on_table=true
for tomo_name in $TOMOS
do
    export output_file=$global_output_dir"/"$tomo_name"/full_tomo_partition.h5"
    export output_dir=$global_output_dir"/"$tomo_name
    echo $output_dir
    mkdir -p $output_dir
    echo 'Generating testing partition for dataset' $tomo_name
    python3 runners/dataset_tables/particle_picking_scripts/1_partition_tomogram.py  -outh5 $output_file -tomo_name $tomo_name -dataset_table $path_to_dataset_table  -box $box_length -write_on_table $write_on_table -overlap $overlap
    echo "... done."
done

