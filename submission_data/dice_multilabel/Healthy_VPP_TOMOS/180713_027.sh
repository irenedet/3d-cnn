#!/usr/bin/env bash

tomo_name="180713_027"
origin_file="/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/etomo/bin4/027_df_sorted.rec"
eman2_filtered_file="/struct/mahamid/Irene/yeast/vpp/180713_027/027_bin4.hdf"
class_0_mask="/struct/mahamid/Irene/yeast/vpp/180713_027/ribos/binarized_non_sph_ribo_clean_mask.hdf"
class_2_mask="/struct/mahamid/Irene/yeast/vpp/180713_027/memb/binarized_lamella_mask.hdf"
mult_factor=-1
test_partition="/scratch/trueba/3d-cnn/cnn_evaluation/dice-multi-class/D2_IF8_NA/180713_027/tomo_partition.h5"
input_xdim=960
input_ydim=928
input_zdim=500
z_shift=0
x_shift=0
hdf_lamella_file="/scratch/trueba/3d-cnn/clean/180713_027/027_lamellamask.hdf"
path_to_motl_clean_0="/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180713/027/TM/motl_clean_4b.em"