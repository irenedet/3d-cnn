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


def get_image_dict(data_path, labels, iterations=0):
    volumes = dict()
    with h5py.File(data_path, 'r') as f:
        print(list(f['volumes']))
        labels = list(f['volumes/labels'])
        print(labels)
        names_list = list(f['volumes/raw'])
        print(names_list)
        vol_numbers = range(len(names_list))
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


dataset_names = [
    "181119/002",
]

pdf = matplotlib.backends.backend_pdf.PdfPages(
    "/g/scb2/zaugg/trueba/3d-cnn/1strongly_labeled0.002_181119_002.pdf")
for index, tomo_name in enumerate(dataset_names):
    print("tomo_name", tomo_name)
    data_path = join("/scratch/trueba/3d-cnn/cross-validation/training-data",
                     tomo_name)
    data_path = join(data_path,
                     "multi_class/strongly_labeled_0.002/full_partition.h5")

    iterations = 0
    labels = ['ribo', 'fas', 'memb']
    volumes = get_image_dict(data_path=data_path, labels=labels, iterations=0)
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
    pdf.savefig(fig)
pdf.close()
