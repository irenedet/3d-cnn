import numpy as np
import matplotlib
from matplotlib import cm
import h5py
from os.path import join
import matplotlib.pyplot as plt
from src.python.filereaders.datasets import load_dataset
import matplotlib.backends.backend_pdf


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
                z_coords, y_coords, x_coords = indicator
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


# dataset_names = [
#     "190218/043",
#     # "190218/044",
#     "190218/048",
#     "190218/049",
#     "190218/050",
#     "190218/051",
#     "190218/052",
#     "190218/054",
#     "190218/056",
#     "190218/059",
#     "190218/060",
#     "190218/061",
#     "190218/062",
#     "190218/063",
#     "190218/064",
#     "190218/065",
#     "190218/066",
#     "190218/067",
#     "190218/068",
#     "190218/069",
#     "190218/070",
#     "190218/071",
#     "190218/072",
#     "190218/073",
#     "190218/075",
#     "190218/076",
#     "190218/077",
#     "190218/078",
#     "190218/081",
#     "190218/082",
#     "190218/083",
#     "190218/084",
#     "190218/085",
#     "190218/086",
#     "190218/087",
#     "190218/088",
#     "190218/089",
#     "190218/090",
#     "190218/091",
#     "190218/092",
#     "190218/093",
#     "190218/094",
#     "190218/095",
#     "190218/096",
#     "190218/097",
#     "190218/098",
#     "190218/099",
#     "190218/100",
#     "190218/101",
#     "190218/102",
#     "190218/103",
#     "190218/104",
#     "190218/105",
#     "190218/106",
#     "190218/108",
#     "190218/110",
#     "190218/111",
#     "190218/112",
#     "190218/113",
#     "190218/114",
#     "190218/115",
#     "190218/116",
#     "190218/117",
#     "190218/118",
#     "190218/119",
#     "190218/120",
#     "190218/121",
#     "190218/122",
#     "190218/123",
#     "190218/124",
#     "190218/125",
#     "190223/129",
#     "190223/130",
#     "190223/131",
#     "190223/132",
#     "190223/133",
#     "190223/134",
#     "190223/135",
#     "190223/136",
#     "190223/139",
#     "190223/140",
#     "190223/141",
#     "190223/142",
#     "190223/143",
#     "190223/144",
#     "190223/145",
#     "190223/146",
#     "190223/148",
#     "190223/149",
#     "190223/151",
#     "190223/152",
#     "190223/153",
#     "190223/154",
#     "190223/155",
#     "190223/156",
#     "190223/157",
#     "190223/159",
#     "190223/160",
#     "190223/162",
#     "190223/163",
#     "190223/165",
#     "190223/166",
#     "190223/168",
#     "190223/169",
#     "190223/171",
#     "190223/172",
#     "190223/173",
#     "190223/174",
#     "190223/175",
#     "190223/176",
#     "190223/177",
#     "190223/178",
#     "190223/179",
#     "190223/180",
#     "190223/181",
#     "190223/182",
#     "190223/183",
#     "190223/184",
#     "190223/185",
#     "190223/186",
#     "190223/187",
#     "190223/188",
#     "190223/189",
#     "190223/190",
#     "190223/191",
#     "190223/192",
#     # "190223/193",
#     "190223/194",
# ]

dataset_names = [
    "190223/132",
    "190223/148",
    "190223/178",
    "190223/183",
    "190223/177",
    "190223/190",
    "190223/191",
    "190223/192",
    "190223/194",
]

# pdf = matplotlib.backends.backend_pdf.PdfPages(
#     "/g/scb2/zaugg/trueba/3d-cnn/NPC_SU_gauss_0.06_0.01_masks_test_datasets.pdf")
# for index, tomo_name in enumerate(dataset_names):
#     print("tomo_name", tomo_name)
#     data_path = join("/struct/mahamid/Irene/NPC/SPombe", tomo_name)
#     data_label = join(data_path, "NPC_SU_mask_gauss_0.06_0.01_bin.hdf")
#     data_raw = join(data_path, "double_eman_filtered_raw_4b.hdf")
#
#     dataset_label = load_dataset(data_label)
#     dataset_raw = load_dataset(data_raw)
#     image_label = dataset_label[175, :, :]
#     image_raw = dataset_raw[175, :, :]
#     matplotlib.use('Agg')
#     plt.ioff()
#     figsize = (10, 10)
#     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize,
#                              num=index, frameon=False)
#     axes[0].imshow(image_raw)
#     axes[1].imshow(image_label)
#     plt.suptitle("tomo " + tomo_name)
#     pdf.savefig(fig)
# pdf.close()

pdf = matplotlib.backends.backend_pdf.PdfPages(
    "/g/scb2/zaugg/trueba/3d-cnn/NPC_SU_strongly_labeled0.02.pdf")
for index, tomo_name in enumerate(dataset_names):
    print("tomo_name", tomo_name)
    data_path = join(
        "/scratch/trueba/3d-cnn/SPombe_NPC_SU/npc_gauss_0.06_0.01_masks",
        tomo_name)
    data_path = join(data_path,
                     "training_data/strongly_labeled_0.02/full_partition.h5")

    iterations = 0
    labels = ['npc']
    volumes = get_image_dict(data_path=data_path, labels=labels, iterations=0,
                             vol_numbers=10)
    names_list = list(volumes.keys())
    vol_numbers = range(len(names_list))
    matplotlib.use('Agg')
    plt.ioff()
    # fig = plt.figure(index, frameon=False)
    figsize = (10, 1.5 * len(vol_numbers))
    nrows = len(vol_numbers)
    nlabels = len(labels)
    fig, axes = plt.subplots(nrows=nrows, ncols=nlabels + 1, figsize=figsize,
                             num=index, frameon=False)
    if nrows > 1:
        for i, vol_number in enumerate(vol_numbers):
            vol_name = names_list[vol_number]
            vol_dict = volumes[vol_name]
            axes[i, 0].imshow(vol_dict['raw'])
            for ilabel, label in enumerate(labels):
                axes[i, ilabel + 1].imshow(vol_dict[label])
    else:
        for i, vol_number in enumerate(vol_numbers):
            vol_name = names_list[vol_number]
            vol_dict = volumes[vol_name]
            axes[0].imshow(vol_dict['raw'])
            for ilabel, label in enumerate(labels):
                axes[ilabel + 1].imshow(vol_dict[label])
    plt.suptitle("tomo " + tomo_name)
    pdf.savefig(fig)
pdf.close()

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
#     from src.python.naming import h5_internal_paths
#
#     with h5py.File(data_path, 'r') as f:
#         n_orig = len(list(f[h5_internal_paths.RAW_SUBTOMOGRAMS]))
#
#     with h5py.File(DA_data_path, 'r') as f:
#         n_DA = len(list(f[h5_internal_paths.RAW_SUBTOMOGRAMS]))
#
#     iterations = n_DA / n_orig
#     print("tomo ", tomo_name, "iterations =", iterations)
