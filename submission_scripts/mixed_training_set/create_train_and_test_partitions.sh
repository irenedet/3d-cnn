#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --time 0-10:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

#export path_to_raw="/scratch/trueba/3d-cnn/clean/180426_004/subtomo380-600.hdf"
#export path_to_labeled="/scratch/trueba/3d-cnn/clean/180426_004/clean_mask.hdf"
#export output_dir="/scratch/trueba/3d-cnn/training_data/TEST/mixed_training/004/G_sigma1"
#export shape_x=928
#export shape_y=928
#export shape_z=221

#export path_to_raw="/scratch/trueba/3d-cnn/clean/180426_005/subtomo370-620.hdf"
#export path_to_labeled="/scratch/trueba/3d-cnn/clean/180426_005/4b_ribos_mask.hdf"
#export output_dir="/scratch/trueba/3d-cnn/training_data/TEST/mixed_training/005/G_sigma1"
#export shape_x=928
#export shape_y=928
#export shape_z=251

#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/004/etomo/bin4/004_df_sorted.rec done
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/005/TM/005_bin4.em done
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/021/etomo/bin4/021_df_sorted.rec done
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/024/etomo/bin4/024_df_sorted.rec done
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/003/etomo/bin4/003_df_sorted.rec done
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/etomo/bin4/004_df_sorted.rec done
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/005/etomo/bin4/005_df_sorted.rec done
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/018/etomo/bin4/018_df_sorted.rec done
#/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/etomo/bin4/027_df_sorted.rec done

export raw_em="/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/005/etomo/bin4/005_df_sorted.rec"
export path_to_raw="/struct/mahamid/Irene/yeast/ribosomes/180426_005/005_bin4.hdf"
export path_to_labeled="/struct/mahamid/Irene/yeast/ribosomes/180426_005/particles_mask.hdf"
export output_dir="/struct/mahamid/Irene/yeast/ribosomes/180426_005/G_sigma1/"


export shape_x=928
export shape_y=928
export shape_z=1000

export label_name="ribosomes"
export split=0.8  # percentage of training + validation set
export box_side=128

# For data augmentation:
export number_iter=6
export train_split=0.8  # percentage of training set (within training + validation)
export overlap=12

echo "applying EMAN2 filter to tomogram"
module load EMAN2
e2proc3d.py $raw_em $path_to_raw --process filter.lowpass.gauss:cutoff_abs=.25 --process filter.highpass.gauss:cutoff_pixels=5 --process normalize --process threshold.clampminmax.nsigma:nsigma=3
echo "... done."

module load Anaconda3
echo 'starting virtual environment'
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse

echo "starting python script"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/generate_mixed_training_set/generate_train_and_test_partitions_single_label_file.py -raw $path_to_raw -labeled $path_to_labeled -output $output_dir -label $label_name -box $box_side -shapex $shape_x -shapey $shape_y -shapez $shape_z -number_iter $number_iter -split $split -train_split $train_split -overlap $overlap
