#!/usr/bin/env bash

tomo_name="180711_004"
origin_file="/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/etomo/bin4/004_df_sorted.rec"
eman2_filtered_file="/struct/mahamid/Irene/yeast/vpp/180711_004/004_bin4.hdf"
class_0_mask="/struct/mahamid/Irene/yeast/vpp/180711_004/ribos/binarized_non_sph_ribo_clean_mask.hdf"
class_1_mask="/struct/mahamid/Irene/yeast/vpp/180711_004/fas/clean_mask.hdf"
class_2_mask="/struct/mahamid/Irene/yeast/vpp/180711_004/memb/binarized_lamella_mask.hdf"
mult_factor=-1
test_partition="/struct/mahamid/Irene/yeast/vpp/180711_004/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5"
input_xdim=960
input_ydim=928
input_zdim=500
z_shift=0
x_shift=-16
hdf_lamella_file="/scratch/trueba/3d-cnn/clean/180711_004/004_lamellamask.hdf"
path_to_motl_clean_0="/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/TM/motl_clean_4b.em"
path_to_motl_clean_1="/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180711/004/FAS/TM/motl_clean_fas_4b.em"