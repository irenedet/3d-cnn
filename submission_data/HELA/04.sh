#!/usr/bin/env bash

tomo_name='04'
origin_file="/struct/mahamid/Irene/NPC/04/04_dose-filt.hdf"
eman2_filtered_file="/struct/mahamid/Irene/NPC/04/04_dose-filt_eman2filter.hdf"
class_npc_mask="/struct/mahamid/Irene/NPC/04/clean/binarized_cylindrical_mask.hdf"
test_partition=""
mult_factor=1 #for eman2 filter (1 if it is not inverted, -1 otherwise)
input_xdim=928
input_ydim=928
input_zdim=400
z_shift=0
x_shift=0