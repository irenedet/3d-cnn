import numpy as np
import matplotlib
from matplotlib import cm
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


datasets_healthy = [
    "180426/004",
    "180426/005",
    "180426/021",
    "180426/024",
    "180711/003",
    "180711/004",
    "180711/005",
    "180711/018",
    "180713/027",
]
datasets_ED = [
    # "190301/003",
    # "190301/005",
    # "190301/009",
    # "190301/012",
    # "190301/016",
    # "190301/022",
    # "190301/028",
    # "190301/031",
    # "190301/032",
    # "190301/033",
    # "190301/035",
    # "190301/037",
    # "190301/043",
    # "190301/045",
    # "190301/001",
    # "190301/002",
    # "190301/004",
    # "190301/006",
    # "190301/007",
    # "190301/010",
    # "190301/011",
    # "190301/013",
    # "190301/014",
    # "190301/015",
    "190301/017",
    # "190301/020",
    # "190301/021",
    "190301/026",
    # "190301/029",
    # "190301/030",
    # "190329/001",
    "190329/004",
    # "190329/005",
    "190329/007",
    # "190329/010",
    # "190329/012",
    # "190329/013",
    # "190329/015",
    # "190329/017",
    # "190329/021",
    # "190329/022",
    # "190329/023",
    # "190329/025",
    # "190329/028",
    # "190329/032",
    # "190329/036",
]

datasets_ED = [
    "190301_017",
    "190301_026",
    "190329_004",
    "190329_007",
]
# pdf = matplotlib.backends.backend_pdf.PdfPages(
#     "/g/scb2/zaugg/trueba/3d-cnn/output.pdf")
# h5_path = "/scratch/trueba/yeast/mitochondria/2Dmitochondria.h5"
for index, tomo_name in enumerate(datasets_ED):
    # path_to_dataset = "/struct/mahamid/Irene/yeast/ED/" + tomo_name + \
    #                   "/eman_filtered_eman_filtered_raw_4b.hdf"
    # pdf = matplotlib.backends.backend_pdf.PdfPages(
    #     "/g/scb2/zaugg/trueba/3d-cnn/output_healthy.pdf")
    # for index, tomo_name in enumerate(datasets_healthy):
    #     path_to_dataset = "/struct/mahamid/Irene/yeast/healthy/" + tomo_name + \
    #                       "/eman_filtered_raw_4b.hdf"
    h5_path = "/scratch/trueba/yeast/mitochondria/2Dmitochondria1.h5"
    import h5py
    from os.path import join

    with h5py.File(h5_path, 'r') as f:
        for volume in datasets_ED:
            internal_path = join('volumes/raw', volume)
            slice = f[internal_path][:]
            slice = slice[:926, :926]

    # dataset = load_dataset(path_to_dataset)
    # print(dataset.shape)
    # sl = dataset.shape[0] // 2
    # print(sl)
    # slice = dataset[sl, :, :]

    matplotlib.use('Agg')
    plt.ioff()

    fig = plt.figure(index, frameon=False)
    plt.imshow(slice, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    plt.gcf()
    # output_folder = "/scratch/trueba/yeast/mitochondria/test/raw/926pix"
    output_folder = "/home/papalotl/Desktop/"
    fig_name = tomo_name[:6] + "_" + tomo_name[7:] + ".png"
    png_name = join(output_folder, fig_name)
    save_fig(png_name, fig=fig)

    # pdf.savefig(fig)
# pdf.close()
