from os.path import join

import h5py
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np


def save_fig(filepath, fig=None):
    '''Save the current image with no whitespace
    Example filepath: "myfig.png" or r"C:\myfig.pdf"
    '''
    import matplotlib.pyplot as plt
    if not fig:
        fig = plt.gcf()

    plt.subplots_adjust(0, 0, 1, 1, 0, 0)
    for ax in fig.axes:
        ax.axis('off')
        ax.margins(0, 0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(filepath, pad_inches=0, bbox_inches='tight')


def get_image_dict(data_path, labels: list = None, iterations=0,
                   vol_numbers: int = None):
    volumes = dict()
    with h5py.File(data_path, 'r') as f:
        print(list(f['volumes']))
        if labels is None:
            labels = list(f['volumes/labels'])
        print(labels)
        names_list = list(f['volumes/raw'])
        print(names_list)
        if vol_numbers is None or vol_numbers > len(names_list):
            vol_numbers = range(len(names_list))
        else:
            vol_numbers = range(vol_numbers)
        print("volumes number =", len(names_list))
        for vol_number in vol_numbers:
            vol_dict = dict()
            vol_name = names_list[vol_number]
            raw_vol = join('volumes/raw', vol_name)
            raw_volume = f[raw_vol][:]
            max_sl = 0
            best_image = 0
            for label in labels:
                label_path = join('volumes/labels', label)
                label_path = join(label_path, vol_name)
                label_volume = f[label_path][:]

                indicator = np.where(label_volume > 0)
                unique_zindicator = np.unique(indicator[0])
                for sl in unique_zindicator:
                    image = label_volume[sl, :, :]
                    img_indicator = np.where(image > 0)
                    # print(len(img_indicator[0]))
                    if best_image < len(img_indicator[0]):
                        best_image = len(img_indicator[0])
                        max_sl = sl
            print("max_sl, best_image", max_sl, best_image)
            vol_dict['raw'] = raw_volume[max_sl, :, :]
            for label in labels:
                label_path = join('volumes/labels', label)
                label_path = join(label_path, vol_name)
                label_volume = f[label_path][:]
                vol_dict[label] = label_volume[max_sl, :, :]
            volumes[vol_name] = vol_dict
    return volumes


import os
from file_actions.readers.tomograms import load_tomogram

ribo_masks_dir = "/struct/mahamid/Irene/scratch/trueba/3d-cnn/cnn_evaluation/ribo_sph_masks_TM_CNNs/64pix_encoder_dropout0_decoder_dropout0_DA_none_BN_False_ribo__D_2_IF_4_set_1"
# tomo_name "/ribo/in_cytosol_mask/motl_*.csv"
tomo_dir = "/struct/mahamid/Irene/yeast"
tomo_names = [
    "ScED_6h/001", "ScED_6h/002", "ScED_6h/003", "ScED_6h/006", "ScED_6h/011", "ScED_6h/015", "ScED_6h/019",
    "ScED_6h/021", "ScED_6h/022", "ScED_6h/023", "ScED_6h/035", "ScED_6h/036", "ScED_6h/037", "ScED_6h/039",
    "ScED_6h/041", "ScED_6h/044", "ScED_6h/047", "ScED_6h/049", "ScED_6h/054", "ScED_6h/055", "ScED_6h/056",
    "ScED_6h/057", "ScED_6h/058", "ScED_6h/059", "ScED_6h/062", "ScED_6h/064", "ScED_6h/065", "ScED_6h/069",
    "ScED_6h/070", ]

pdf = matplotlib.backends.backend_pdf.PdfPages(
        "/struct/mahamid/Irene/3d-cnn/ScED_6h_ribo_clusters_datasets1.pdf")
for index, tomo_name in enumerate(tomo_names[:10]):
    print("tomo_name", tomo_name)
    data_path = join(tomo_dir, tomo_name)
    data_raw = join(data_path, "1xf_tomo.mrc")

    data_label = os.path.join(ribo_masks_dir, tomo_name)
    data_label = join(data_label, "ribo/clusters.hdf")
    dataset_label = load_tomogram(data_label)
    dataset_raw = load_tomogram(data_raw)
    image_label = dataset_label[250, :, :]
    image_raw = dataset_raw[250, :, :]
    matplotlib.use('Agg')
    plt.ioff()
    figsize = (10, 10)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize,
                             num=index, frameon=False)
    axes[0].imshow(image_raw)
    axes[1].imshow(image_label)
    plt.suptitle("tomo " + tomo_name)
    pdf.savefig(fig)
pdf.close()
# /scratch/trueba/3d-cnn/cross-validation/fas_cross_validation/original-training-data//181126/002/strongly_labeled_min0.002_max0.1/no_eman_filter_256pix/full_partition.h5
# import random
#
# pdf = matplotlib.backends.backend_pdf.PdfPages(
#     "training_data_test.pdf")

# for index, tomo_name in enumerate(tomo_names):
#     print("tomo_name", tomo_name)
#
#     data_path = os.path.join()
#
#     labels = ['memb']
#     volumes = get_image_dict(data_path=data_path, labels=labels, iterations=0)
#     names_list = list(volumes.keys())
#     names_list = random.choices(names_list, k=5)
#     vol_numbers = range(len(names_list))
#
#     matplotlib.use('Agg')
#     plt.ioff()
#     print("starting fig for", data_path)
#     figsize = (10, 1.5 * len(names_list))
#     nrows = len(vol_numbers)
#     nlabels = len(labels)
#     fig, axes = plt.subplots(nrows=nrows, ncols=nlabels + 1, figsize=figsize,
#                              num=index, frameon=False)
#     if nrows > 1:
#         for i, vol_number in enumerate(vol_numbers):
#             vol_name = names_list[vol_number]
#             vol_dict = volumes[vol_name]
#             axes[i, 0].imshow(vol_dict['raw'])
#             for ilabel, label in enumerate(labels):
#                 axes[i, ilabel + 1].imshow(vol_dict[label])
#     else:
#         for i, vol_number in enumerate(vol_numbers):
#             vol_name = names_list[vol_number]
#             vol_dict = volumes[vol_name]
#             axes[0].imshow(vol_dict['raw'])
#             for ilabel, label in enumerate(labels):
#                 axes[ilabel + 1].imshow(vol_dict[label])
#     plt.suptitle("tomo " + data_path[-30:-27])
#     pdf.savefig(fig)
# pdf.close()

# for index, tomo_name in enumerate(dataset_names):
#     data_path = join("/scratch/trueba/3d-cnn/SPombe_NPC_SU",
#                      tomo_name)
#     data_path = join(data_path,
#                      "training_data/strongly_labeled_0.002/full_partition.h5")
#     DA_data_path = join("/scratch/trueba/3d-cnn/SPombe_NPC_SU",
#                         tomo_name)
#     DA_data_path = join(DA_data_path,
#                         "training_data/strongly_labeled_0.002")
#     DA_data_path = join(DA_data_path,
#                         "G1.5_E2_R180_DArounds4/full_partition.h5")
#
#     from naming import h5_internal_paths
#
#     with h5py.File(data_path, 'r') as f:
#         n_orig = len(list(f[h5_internal_paths.RAW_SUBTOMOGRAMS]))
#
#     with h5py.File(DA_data_path, 'r') as f:
#         n_DA = len(list(f[h5_internal_paths.RAW_SUBTOMOGRAMS]))
#
#     iterations = n_DA / n_orig
#     print("tomo ", tomo_name, "iterations =", iterations)
