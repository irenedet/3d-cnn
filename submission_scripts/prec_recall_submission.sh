#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 4G
#SBATCH --time 0-00:30
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de


export QT_QPA_PLATFORM='offscreen'

export path_to_motl="/scratch/trueba/test_pr/3emanf/30_pix_far/redundant/PR/redundant/motl_1958.csv"
export output_dir='/scratch/trueba/test_pr/3emanf/30_pix_far/redundant/PR/redundant'
export path_to_clean="/struct/mahamid/twalther/Processing/190301/005/TM/motl_clean_4b.em"

echo path_to_motl_predicted=$path_to_motl
echo path_to_motl_true=$path_to_clean
export label_name='ribosomes'
export radius=8
export x_shift=0


module load Anaconda3
echo 'starting virtual environment'
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/

echo 'Starting precision-recall analysis'
python3 /g/scb2/zaugg/trueba/3d-cnn/runners/prec_recall_analysis.py -motl $path_to_motl -output $output_dir -clean $path_to_clean -label $label_name -min_peak_distance $radius -x_shift $x_shift
echo '... done.'