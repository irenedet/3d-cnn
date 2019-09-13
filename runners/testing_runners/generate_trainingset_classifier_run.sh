#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 128G
#SBATCH --time 0-3:00
#SBATCH -o training_set_classifier.slurm.%N.%j.out
#SBAtCH -e training_set_classifier.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"


export dataset_table="/struct/mahamid/Irene/liang_data/multiclass/liang_data_multiclass.csv"
export box_side=64
export semantic_classes='70S,50S'
export output_dir="/scratch/trueba/3Dclassifier/liang_data/training_data/"
export dataset_table="/struct/mahamid/Irene/liang_data/multiclass/liang_data_multiclass.csv"
export tomo_name='200'
export write_on_table=False

echo "starting python script:"
python3 ./runners/testing_runners/generate_trainingset_classifier.py -dataset_table $dataset_table -box_side $box_side -semantic_classes $semantic_classes -output_dir $output_dir -dataset_table $dataset_table -tomo_name $tomo_name -write_on_table $write_on_table
echo "... done."