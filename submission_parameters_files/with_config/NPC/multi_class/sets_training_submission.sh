#! /bin/bash

config_file="/struct/mahamid/Irene/3d-cnn/submission_parameters_files/with_config/NPC/multi_class/config_training.yaml"
sets="1 2 3"
for set in $sets
do
  echo "set" $set
  sbatch /struct/mahamid/Irene/3d-cnn/submission_parameters_files/with_config/NPC/multi_class/training/training.sh -yaml_file $config_file -tomos_set $set
done