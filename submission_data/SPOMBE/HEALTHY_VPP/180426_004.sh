#!/usr/bin/env bash

tomo_name='180426_004'
origin_file="/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/004/NovaCTF/001_bin4.rec"
eman2_filtered_file="/struct/mahamid/Irene/yeast/vpp/180426_004/004_bin4_sq.hdf"
class_0_mask="/struct/mahamid/Irene/yeast/vpp/180426_004/ribos/binarized_non_sph_corrected_mask.hdf"
class_1_mask="/struct/mahamid/Irene/yeast/vpp/180426_004/fas/clean_mask_corrected.hdf"
class_2_mask="/struct/mahamid/Irene/yeast/vpp/180426_004/memb/binarized_lamella_mask.hdf"
test_partition="/struct/mahamid/Irene/yeast/vpp/180426_004/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5"
mult_factor=1 #for eman2 filter (1 if it is not inverted, -1 otherwise)
input_xdim=927
input_ydim=927
input_zdim=1000
z_shift=0
x_shift=-16
hdf_lamella_file="/scratch/trueba/3d-cnn/clean/180426_004/004_lamellamask.hdf"
path_to_motl_clean_0="/struct/mahamid/Irene/yeast/vpp/180426_004/ribos/motl/corrected_motl.csv"
path_to_motl_clean_1="/struct/mahamid/Irene/yeast/vpp/180426_004/fas/motl/corrected_motl.csv"