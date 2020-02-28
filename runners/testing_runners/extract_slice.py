import h5py

from file_actions.readers.tomograms import load_tomogram

path_to_dataset = "/home/papalotl/Sara_Goetz/180426/043/NovaCTF/rawtomograms/043_bin4.hdf"
motl = load_tomogram(path_to_dataset)
dimz = motl.shape[0]
with h5py.File(
        "/home/papalotl/Sara_Goetz/180426/043/NovaCTF/rawtomograms/central_slice.h5") as f:
    f['image'] = motl[dimz // 2, :, :]
