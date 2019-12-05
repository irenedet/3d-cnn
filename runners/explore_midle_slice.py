# import numpy as np
# import matplotlib
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

from filereaders.datasets import load_dataset

datasets = [
    "190301/003",
    "190301/005",
    "190301/009",
    "190301/012",
    "190301/016",
    "190301/022",
    "190301/028",
    "190301/031",
    "190301/032",
    "190301/033",
    "190301/035",
    "190301/037",
    "190301/043",
    "190301/045",
    "190301/001",
    "190301/002",
    "190301/004",
    "190301/006",
    "190301/007",
    "190301/010",
    "190301/011",
    "190301/013",
    "190301/014",
    "190301/015",
    "190301/017",
    "190301/020",
    "190301/021",
    "190301/026",
    "190301/029",
    "190301/030",
    "190329/001",
    "190329/004",
    "190329/005",
    "190329/007",
    "190329/010",
    "190329/012",
    "190329/013",
    "190329/015",
    "190329/017",
    "190329/021",
    "190329/022",
    "190329/023",
    "190329/025",
    "190329/028",
    "190329/032",
    "190329/036",
]

pdf = matplotlib.backends.backend_pdf.PdfPages(
    "/g/scb2/zaugg/trueba/3d-cnn/output2.pdf")
for index, tomo_name in enumerate(datasets):
    path_to_dataset = "/struct/mahamid/Irene/yeast/ED/" + tomo_name + \
                      "/eman_filtered_eman_filtered_raw_4b.hdf"
    dataset = load_dataset(path_to_dataset)
    matplotlib.use('Agg')
    plt.ioff()
    fig = plt.figure(index)
    sl = dataset.shape[0] // 2
    print(sl)
    plt.imshow(dataset[sl, :, :], cmap=plt.cm.Greys_r)
    plt.xlabel("y-axis")
    plt.ylabel("x-axis")
    plt.title(tomo_name)
    plt.gcf()
    # plt.show()
    pdf.savefig(fig)
pdf.close()
