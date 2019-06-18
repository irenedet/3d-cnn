#!/usr/bin/env bash

tomo_name='190301_005'
origin_file="/struct/mahamid/Sara_Goetz/Data/Titan/Processing/190301/005/etomo/bin4/005_sq_df_sorted.rec"
eman2_filtered_file="/struct/mahamid/Irene/yeast/ED/190301_005/eman_filt_eman_filt_005_sq_df_sorted.hdf"
test_partition="/scratch/trueba/3d-cnn/training_data/dice-multi-class/190301_005/NA/train_and_test_partitions/full_partition.h5"
class_0_mask="/struct/mahamid/Irene/yeast/ED/190301_005/clean_masks/class_0/spherical_mask.hdf"
class_1_mask="/struct/mahamid/Irene/yeast/ED/190301_005/clean_masks/class_1/spherical_mask.hdf"
class_2_mask="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/Retrain_retrained_D4_IF8_NA_except_180711_003/double_eman_filt/190301_005/thr_binary_memb.hdf"
test_partition="/struct/mahamid/Irene/yeast/ED/190301_005/eman_filt_eman_filt_tomo_partition.h5"
mult_factor=1 #for eman2 filter (1 if it is not inverted, -1 otherwise)
input_xdim=928
input_ydim=928
input_zdim=500
z_shift=0
x_shift=0
hdf_lamella_file="/struct/mahamid/Irene/yeast/ED/190301_005/005_lamellamask.hdf"
path_to_motl_clean_1="/struct/mahamid/Irene/yeast/ED/190301_005/motl/motl_ED_FAS_4b.csv"
path_to_motl_clean_0="/struct/mahamid/Irene/yeast/ED/190301_005/motl/RR_all_but_003/undetected/motl_2561.csv"