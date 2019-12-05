from datasets.actions import load_training_dataset_list

training_partition_paths = [
    "/struct/mahamid/Irene/yeast/healthy/180426/004/G_sigma1_non_sph/train_and_test_partitions/full_partition.h5"]
# training_partition_paths = ["/struct/mahamid/Irene/yeast/ED/181126/025/fas_class/train_and_test_partitions/G1.5_E2_R90/fraction_2.h5"]

data_aug_rounds_list = [1]
# data_aug_rounds_list = [3]

segmentation_names = ['ribo', 'fas']
# segmentation_names = ['fas']

split = 0.5

load_training_dataset_list(training_partition_paths,
                           data_aug_rounds_list,
                           segmentation_names,
                           split)
