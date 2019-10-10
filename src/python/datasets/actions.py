import numpy as np
import random

from src.python.filereaders.datasets import load_dataset
from src.python.filewriters.h5 import write_raw_subtomograms_intersecting_mask
from src.python.filewriters.h5 import \
    write_joint_raw_and_labels_subtomograms_dice_multiclass
from src.python.coordinates_toolbox.subtomos import \
    get_particle_coordinates_grid_with_overlap
from src.python.filewriters.h5 import write_subtomograms_from_dataset, \
    write_joint_raw_and_labels_subtomograms, write_strongly_labeled_subtomograms
from src.python.image.filters import preprocess_data
from src.python.filereaders.h5 import read_training_data_dice_multi_class


def _chunkify_list(lst, n):
    """
    Splits a list lst into n chuncks of consecutive elements.
    e.g. chunkify(lst=[0,1,2,3,4,5,6,7,8,9],n=5)
    [array([0, 1]), array([2, 3]), array([4, 5]), array([6, 7]), array([  8, 9])]
    """
    return np.array_split(lst, n)


def _chunks_to_tensor(chunks: list) -> np.array:
    """
    Takes a list of chunks of volumes, and generates a single tensor containing
    all volumes in the chunks:
    :param chunks: [chunk_0, ...., chunk_n] where
    chunk_i = np.array([vol_i_0, vol_i_1, ..., vol_i_k])
    :return: np.array([vol_0_0, ..., vol_i_j, ..., vol_n_k])
    """
    tensor = list()
    for volume in chunks:
        tensor += list(volume)
    return np.array(tensor)


def _define_splitting(split: int or float, n_total: int) -> int:
    """
    Defines splitting integer
    """
    assert (isinstance(split, int) or isinstance(split, float)), \
        "split should be either int or float in (0, 1)"
    if isinstance(split, float):
        assert 0 < split < 1
        split = int(split * n_total)
    print("split = ", split)
    return split


def _chunkify_by_data_augmentation_rounds(raw_data: list, labels: list,
                                          n_chunks: int) -> tuple:
    # Get a list of d number of "chunks" that correspond to each
    # data augmentation round. Thus we get a list of d elements, each of them
    # of size:
    # k x 1 x Dx x Dy x Dz (for raw data) and
    # k x S x Dx x Dy x Dz (for labels data),
    # where k = original_volumes_number (k = len(data)//n_chunks).
    raw_data_chunks = _chunkify_list(lst=raw_data, n=n_chunks)
    labels_data_chunks = _chunkify_list(lst=labels, n=n_chunks)
    assert len(raw_data_chunks) == len(labels_data_chunks), \
        "The labels and raw data were not divided into equal chunks..."
    return raw_data_chunks, labels_data_chunks


def _split_data_augmentation_chunks(raw_data_chunks: list,
                                    labels_data_chunks: list,
                                    split: int or float,
                                    shuffle: bool = True) -> tuple:
    """
    Given raw_data_chunks and labels_data_chunks of shapes:
    k x 1 x Dx x Dy x Dz and k x S x Dx x Dy x Dz
    Create a list of k tuples (where k is the number of original volumes).
    Each tuple is of length 1 + 2*n_chunks (n_chunks = data_aug_rounds + 1):
    (id, raw_id_0, ..., raw_id_n-1, label_id_0, ..., label_id_n-1), for
    id = 0, ..., k-1, where id is the volume index.
    Here:
    raw_id_j is the id-th volume in the j-th data augmentation round
    label_id_j is the id-th volume labels in the j-th data augmentation round.
    each raw_id_j has shape (1, Dx, Dy, Dz) and label_id_j of (S, Dx, Dy, Dz).
    :param raw_data_chunks:
    :param labels_data_chunks:
    :param shuffle:
    :param split:
    :return:
    """
    assert len(labels_data_chunks[0]) == len(raw_data_chunks[0]), \
        "raw and label data chunks are not compatible"
    original_volumes_number = len(raw_data_chunks[0])
    data_order = list(range(original_volumes_number))
    combined_list = [data_order] + raw_data_chunks + labels_data_chunks
    zipped = list(zip(*combined_list))
    if shuffle:
        random.shuffle(zipped)
    else:
        print("Splitting sets without shuffling")

    zipped_train = zipped[:split]
    zipped_val = zipped[split:]
    return zipped_train, zipped_val


def _get_unzipped_data(zipped_chunks: list, n_chunks) -> tuple:
    """
    Unzip and get a list of length 1 + 2*n_chunks:
    [shuffled_data order,
     raw_data_0   , ..., raw_data_n-1,
     labels_data_0, ..., labels_data_n-1]

    each of them is a tuple of size k (where k is the number of original
    volumes), such that:

    data order = (sh(0), ..., sh(k-1))
    raw_data_j = (raw_sh(0)_j, ..., raw_sh(k-1)_j)
    labels_data_j = (label_sh(0)_j, ..., label_sh(k-1)_j)

    where raw_id_j is an array of shape 1 x Dx x Dy x Dz and
    each label_id_j is an array of shape S x Dx x Dy x Dz
    """
    unzipped_chunks = list(zip(*zipped_chunks))
    data_order = unzipped_chunks[0]
    raw_data_chunks = unzipped_chunks[1: n_chunks + 1]
    labels_data_chunks = unzipped_chunks[n_chunks + 1:]
    return data_order, raw_data_chunks, labels_data_chunks


def _unchunkify_data(raw_data_chunks: list, labels_data_chunks: list):
    """
    Given chunks of data augmentation raw and label data, with length n
    (where n is the number of volumes before data augmentatation rounds),
    given by:
    raw_data_chunks = [raw_data_0   , ..., raw_data_n-1]
    and
    labels_data_chunks = [labels_data_0, ..., labels_data_n-1],

    and where the elements of both lists are tuples of length k,
    (where k-1 is the number of rounds of data augmentation) such that:

    raw_data_j = (raw_0_j, ..., raw_k-1_j)
    labels_data_j = (label_0_j, ..., label_k-1_j)

    for all j=0, ..., n-1.

    :param raw_data_chunks: [raw_data_0   , ..., raw_data_n-1]
    :param labels_data_chunks: [labels_data_0, ..., labels_data_n-1]
    :return: raw_data, labels_data, where:
    raw_data is the shuffled list of [raw_sh(i)_j]i,j and
    labels_data is the corresponding shuffled list [label_sh(i)_j]i,j,
    and where sh: [0,...,k-1] -> [0,...,k-1] denotes the permutation function.
    """
    raw_data = _chunks_to_tensor(chunks=raw_data_chunks)
    labels_data = _chunks_to_tensor(chunks=labels_data_chunks)

    # Shuffle to destroy the volume id order
    data_to_shuffle = list(zip(raw_data, labels_data))
    random.shuffle(data_to_shuffle)
    shuffled_data = list(zip(*data_to_shuffle))

    raw_data, labels_data = shuffled_data
    raw_data, labels_data = np.array(raw_data), np.array(labels_data)
    return raw_data, labels_data


def split_dataset(data: np.array, labels: np.array, split: float,
                  data_aug_rounds: int = 0, shuffle=True) -> tuple:
    """
    Splits a h5 partition into training and validation set, TrainSet and
    ValidationSet.
    If data_aug_rounds = d > 0, then the volumes are assumed to be read as:
    -1_0 ... -1_N
     0_0 ...  0_N

     d_0 ...  d_N
    where N is the number of original volumes. In this case, the TrainSet and
    the ValidationSet are such that if j_i in TrainSet, then k_i in TrainSet
    for all k=0...d. That is, we do not mix information of the i-th volume
    between the TrainSet and ValidationSet.
    :param data: array of shape N x 1 x Dx x Dy x Dz
    :param labels: array of shape N x S x Dx x Dy x Dz
    :param split: integer or float in (0, 1)
    :param data_aug_rounds: d
    :param shuffle: if True, the training examples are shuffled before loading
    """
    data = list(data)
    labels = list(labels)

    n_chunks = data_aug_rounds + 1
    original_volumes_number = len(data) // n_chunks
    print("volumes number before data augmentation", original_volumes_number)
    print("volumes number after data augmentation", len(data))

    if len(data) == 1:
        print("No validation set from this list.")
        train_data, train_labels = data, labels
        val_data, val_labels = [], []
        final_data_order = [0]
    else:
        split = _define_splitting(split=split, n_total=original_volumes_number)
        raw_data_chunks, labels_data_chunks = \
            _chunkify_by_data_augmentation_rounds(raw_data=data, labels=labels,
                                                  n_chunks=n_chunks)
        zipped_train, zipped_val = \
            _split_data_augmentation_chunks(raw_data_chunks=raw_data_chunks,
                                            labels_data_chunks=labels_data_chunks,
                                            split=split, shuffle=shuffle)

        data_order_train, shuffled_raw_chunks_train, shuffled_label_chunks_train = \
            _get_unzipped_data(zipped_chunks=zipped_train, n_chunks=n_chunks)

        data_order_val, shuffled_raw_chunks_val, shuffled_label_chunks_val = \
            _get_unzipped_data(zipped_chunks=zipped_val, n_chunks=n_chunks)

        final_data_order = data_order_train + data_order_val

        train_data, train_labels = \
            _unchunkify_data(raw_data_chunks=shuffled_raw_chunks_train,
                             labels_data_chunks=shuffled_label_chunks_train)
        val_data, val_labels = \
            _unchunkify_data(raw_data_chunks=shuffled_raw_chunks_val,
                             labels_data_chunks=shuffled_label_chunks_val)
    return train_data, train_labels, val_data, val_labels, final_data_order


def load_training_dataset_list(training_partition_paths: list,
                               data_aug_rounds_list: list,
                               segmentation_names: list,
                               split: int or float) -> tuple:
    """
    Loads raw and labels volume sets from a list of partition .h5 files. It
    outputs a split between train and validation sets for a neural network,
    arising from equally splitting each training dataset (in the same proportion
    if split is float, or at the same volume number if split is an integer).
    :param training_partition_paths: list of h5 paths
    :param data_aug_rounds_list: list of integers indicating number of data
    augmentation rounds used to generate the corresponding h5 path.
    :param segmentation_names: list of strings, where each string is the name of
    semantic classes for training the neural network.
    :param split: if a float between 0 and 1, defines the proportion of the
    data volumes considered for the training set (and the rest will be for
    validation); if an integer, it determines the volume number where the
    splitting between training and validation set will be done.
    :return:
    """
    train_data = list()
    train_labels = list()
    val_data = list()
    val_labels = list()

    for data_aug_rounds, training_data_path in zip(data_aug_rounds_list,
                                                   training_partition_paths):
        print("\n")
        print("Loading training set from ", training_data_path)
        print("data_aug_rounds", data_aug_rounds)
        raw_data, labels = read_training_data_dice_multi_class(
            training_data_path=training_data_path,
            segmentation_names=segmentation_names)

        print("labels.shape = ", labels.shape)
        print("raw_data.shape = ", raw_data.shape)

        if raw_data.shape[0] == 0:
            print('Empty training set in ', training_data_path)
        else:
            print("Initial unique labels", np.unique(labels))

            # Normalize data
            preprocessed_data = preprocess_data(raw_data)
            preprocessed_data = np.array(preprocessed_data)
            labels = np.array(labels, dtype=np.long)

            train_data_tmp, train_labels_tmp, \
            val_data_tmp, val_labels_tmp, _ = \
                split_dataset(data=preprocessed_data, labels=labels,
                              split=split, data_aug_rounds=data_aug_rounds)

            train_data += list(train_data_tmp)
            train_labels += list(train_labels_tmp)

            not_empty = len(val_labels_tmp) > 0
            if not_empty:
                val_data += list(val_data_tmp)
                val_labels += list(val_labels_tmp)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    val_data = np.array(val_data)
    val_labels = np.array(val_labels)
    return train_data, train_labels, val_data, val_labels


def get_right_padding_lengths(tomo_shape, shape_to_crop_zyx):
    padding = [box_size - (tomo_size % box_size) for tomo_size, box_size
               in zip(tomo_shape, shape_to_crop_zyx)]
    return padding


def pad_dataset(dataset: np.array, cubes_with_border_shape: tuple,
                overlap_thickness: int = 12) -> np.array:
    """
    :param dataset: Orignal unpadded dataset to partition, of shape
    (dim_z, dim_y, dim_x).
    :param cubes_with_border_shape: shape of output subtomograms, in format
    (box_z, box_y, box_x).
    :param overlap_thickness: thickness of overlap (on each side of all
    dimensions), by default is 12 pixels.
    :return: a padded dataset of size
    (2*overlap + nz*p_box_z, 2*overlap + ny*p_box_y, 2*overlap + nx*p_box_x)
    where nz, ny, nx are the minimum integers such that ni*p_box_i > dim_i
    and p_box_i is the side-length non-overlapping part of the subtomograms,
    i.e., p_box_i = box_i - 2*overlap.
    """
    internal_cube_shape = [dim - 2 * overlap_thickness for dim in
                           cubes_with_border_shape]
    right_padding = get_right_padding_lengths(dataset.shape,
                                              internal_cube_shape)
    right_padding = [[overlap_thickness, padding + overlap_thickness] for
                     padding in right_padding]

    padded_dataset = np.pad(array=dataset, pad_width=right_padding,
                            mode="reflect")
    return padded_dataset


def partition_tomogram(dataset, output_h5_file_path: str,
                       subtomo_shape: tuple,
                       overlap: int
                       ):
    padded_dataset = pad_dataset(dataset, subtomo_shape, overlap)
    padded_particles_coordinates = get_particle_coordinates_grid_with_overlap(
        padded_dataset.shape,
        subtomo_shape,
        overlap)
    write_subtomograms_from_dataset(output_h5_file_path, padded_dataset,
                                    padded_particles_coordinates,
                                    subtomo_shape)


def partition_raw_and_labels_tomograms(raw_dataset: np.array,
                                       labels_dataset: np.array,
                                       label_name: str,
                                       output_h5_file_path: str,
                                       subtomo_shape: tuple,
                                       overlap: int
                                       ):
    padded_raw_dataset = pad_dataset(raw_dataset, subtomo_shape, overlap)
    padded_labels_dataset = pad_dataset(labels_dataset, subtomo_shape, overlap)
    assert padded_raw_dataset.shape == padded_labels_dataset.shape

    padded_particles_coordinates = get_particle_coordinates_grid_with_overlap(
        padded_raw_dataset.shape,
        subtomo_shape,
        overlap)
    write_joint_raw_and_labels_subtomograms(
        output_path=output_h5_file_path,
        padded_raw_dataset=padded_raw_dataset,
        padded_labels_dataset=padded_labels_dataset,
        label_name=label_name,
        window_centers=padded_particles_coordinates,
        crop_shape=subtomo_shape)


def partition_raw_intersecting_mask(dataset: np.array,
                                    mask_dataset: np.array,
                                    output_h5_file_path: str,
                                    subtomo_shape: tuple,
                                    overlap: int
                                    ):
    padded_raw_dataset = pad_dataset(dataset, subtomo_shape, overlap)
    padded_mask_dataset = pad_dataset(mask_dataset, subtomo_shape, overlap)

    padded_particles_coordinates = get_particle_coordinates_grid_with_overlap(
        padded_raw_dataset.shape,
        subtomo_shape,
        overlap)

    write_raw_subtomograms_intersecting_mask(
        output_path=output_h5_file_path,
        padded_raw_dataset=padded_raw_dataset,
        padded_mask_dataset=padded_mask_dataset,
        window_centers=padded_particles_coordinates,
        crop_shape=subtomo_shape)


def partition_raw_and_labels_tomograms_dice_multiclass(
        path_to_raw: str,
        labels_dataset_list: list,
        segmentation_names: list,
        output_h5_file_path: str,
        subtomo_shape: tuple,
        overlap: int
):
    raw_dataset = load_dataset(path_to_raw)
    padded_raw_dataset = pad_dataset(raw_dataset, subtomo_shape, overlap)
    padded_particles_coordinates = get_particle_coordinates_grid_with_overlap(
        padded_raw_dataset.shape,
        subtomo_shape,
        overlap)
    padded_labels_dataset_list = []
    for path_to_labeled in labels_dataset_list:
        labels_dataset = load_dataset(path_to_labeled)
        labels_dataset = np.array(labels_dataset)
        print(path_to_labeled, "shape", labels_dataset.shape)
        padded_labels_dataset = pad_dataset(labels_dataset, subtomo_shape,
                                            overlap)
        padded_labels_dataset_list += [padded_labels_dataset]
    datasets_shapes = [padded.shape for padded in padded_labels_dataset_list]
    datasets_shapes += [padded_raw_dataset.shape]
    print("padded_dataset.shapes = ", datasets_shapes)
    write_joint_raw_and_labels_subtomograms_dice_multiclass(
        output_path=output_h5_file_path,
        padded_raw_dataset=padded_raw_dataset,
        padded_labels_list=padded_labels_dataset_list,
        segmentation_names=segmentation_names,
        window_centers=padded_particles_coordinates,
        crop_shape=subtomo_shape)
    return


def generate_strongly_labeled_partition(path_to_raw: str,
                                        labels_dataset_list: list,
                                        segmentation_names: list,
                                        output_h5_file_path: str,
                                        subtomo_shape: tuple,
                                        overlap: int,
                                        min_label_fraction: float = 0):
    raw_dataset = load_dataset(path_to_raw)
    padded_raw_dataset = pad_dataset(raw_dataset, subtomo_shape, overlap)
    padded_particles_coordinates = get_particle_coordinates_grid_with_overlap(
        padded_raw_dataset.shape,
        subtomo_shape,
        overlap)
    padded_labels_dataset_list = []
    for path_to_labeled in labels_dataset_list:
        labels_dataset = load_dataset(path_to_labeled)
        labels_dataset = np.array(labels_dataset)
        print(path_to_labeled, "shape", labels_dataset.shape)
        padded_labels_dataset = pad_dataset(labels_dataset, subtomo_shape,
                                            overlap)
        padded_labels_dataset_list += [padded_labels_dataset]
    datasets_shapes = [padded.shape for padded in padded_labels_dataset_list]
    datasets_shapes += [padded_raw_dataset.shape]
    print("padded_dataset.shapes = ", datasets_shapes)
    write_strongly_labeled_subtomograms(
        output_path=output_h5_file_path,
        padded_raw_dataset=padded_raw_dataset,
        padded_labels_list=padded_labels_dataset_list,
        segmentation_names=segmentation_names,
        window_centers=padded_particles_coordinates,
        crop_shape=subtomo_shape,
        min_label_fraction=min_label_fraction)
    return
