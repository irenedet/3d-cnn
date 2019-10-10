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


def get_image_dict(data_path, label, iterations=0):
    volumes = dict()
    with h5py.File(data_path, 'r') as f:
        print(list(f['volumes']))
        labels = list(f['volumes/labels'])
        print(labels)
        names_list = list(f['volumes/raw'])
        print(names_list)
        vol_numbers = range(len(names_list))
        print("volumes number =", len(names_list))
        for vol_number in range(len(names_list)):
            vol_dict = dict()
            vol_name = names_list[vol_number]
            raw_vol = join('volumes/raw', vol_name)
            raw_volume = f[raw_vol][:]

            label_path = join('volumes/labels', label)
            label_path = join(label_path, vol_name)
            label_volume = f[label_path][:]

            indicator = np.where(label_volume > 0)
            z_coords, y_coords, x_coords = indicator
            unique_zindicator = np.unique(indicator[0])
            max_sl = 0
            best_image = 0
            for sl in unique_zindicator:
                image = label_volume[sl, :, :]
                img_indicator = np.where(image > 0)
                # print(len(img_indicator[0]))
                if best_image < len(img_indicator[0]):
                    best_image = len(img_indicator[0])
                    max_sl = sl
            print("max_sl, best_image", max_sl, best_image)
            vol_dict['raw'] = raw_volume[max_sl, :, :]
            vol_dict[label] = label_volume[max_sl, :, :]

            volumes[vol_name] = vol_dict
    return volumes


dataset_names = [
    "180413/006",
    "180413/007",
    "180426/005",
    "180426/006",
    "180426/008",
    "180426/014",
    "180426/038",
    "180426/027",
    "180426/037",
    "180426/040",
    "180426/043",
    "180711/005",
    "180711/006",
    "180711/007",
    "180711/012",
    "180711/017",
    "180711/022",
    "180713/002",
    "180713/005",
    "180713/007",
    "180713/015",
    "180713/018",
    "180713/020",
    "180713/025",
    "180713/027",
    "180713/031",
    "180713/035",
    "180713/037",
    "180713/039",
    "180713/041",
    "180713/043",
    "180713/050",
]

# pdf = matplotlib.backends.backend_pdf.PdfPages(
#     "/g/scb2/zaugg/trueba/3d-cnn/eman2_christians_datasets.pdf")
# # h5_path = "/scratch/trueba/yeast/mitochondria/2Dmitochondria.h5"
# for index, tomo_name in enumerate(dataset_names):
#     # chris_dir = "/struct/mbeck/zimmerli/SPombe/subtomo/npc_bin4/Tomograms/"
#     # tomo_name_ext = tomo_name + ".rec"
#     # path_to_dataset = join(chris_dir, tomo_name_ext)
#     path_to_dataset = join("/struct/mahamid/Irene/yeast/healthy/", tomo_name)
#     path_to_dataset = join(path_to_dataset, "double_eman_filtered_raw_4b.hdf")
#     dataset = load_dataset(path_to_dataset)
#     dimz, _, _ = dataset.shape
#     sl = dimz // 2
#     slice = dataset[sl, :, :]
#
#     matplotlib.use('Agg')
#     plt.ioff()
#     fig = plt.figure(index, frameon=False)
#     plt.imshow(slice, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
#     plt.title(tomo_name)
#     plt.gcf()
#     pdf.savefig(fig)
# # pdf.close()
pdf = matplotlib.backends.backend_pdf.PdfPages(
    "/g/scb2/zaugg/trueba/3d-cnn/strongly_labeled_training_data_npc_yeast_datasets0.02.pdf")
# h5_path = "/scratch/trueba/yeast/mitochondria/2Dmitochondria.h5"
for index, tomo_name in enumerate(dataset_names):
    print("tomo_name", tomo_name)
    data_path = join("/struct/mahamid/Irene/yeast/healthy/", tomo_name)
    data_path = join(data_path,
                     "npc_class/strongly_labeled0.02/train_and_test_partitions/full_partition.h5")

    iterations = 0
    label = 'npc'
    volumes = get_image_dict(data_path=data_path, label=label, iterations=0)
    names_list = list(volumes.keys())
    vol_numbers = range(len(names_list))
    matplotlib.use('Agg')
    plt.ioff()
    # fig = plt.figure(index, frameon=False)
    figsize = (10, 1.5 * len(vol_numbers))
    nrows = len(vol_numbers)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=figsize, num=index,
                             frameon=False)
    if nrows > 1:
        for i, vol_number in enumerate(vol_numbers):
            vol_name = names_list[vol_number]
            vol_dict = volumes[vol_name]
            axes[i, 0].imshow(vol_dict['raw'])
            axes[i, 1].imshow(vol_dict[label])
    else:
        for i, vol_number in enumerate(vol_numbers):
            vol_name = names_list[vol_number]
            vol_dict = volumes[vol_name]
            axes[0].imshow(vol_dict['raw'])
            axes[1].imshow(vol_dict[label])
    pdf.savefig(fig)
pdf.close()
