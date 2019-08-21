from src.python.filereaders.datasets import load_dataset
import h5py

path_to_dataset = "/home/papalotl/Sara_Goetz/180426/043/NovaCTF/rawtomograms/043_bin4.hdf"
dataset = load_dataset(path_to_dataset)
dimz = dataset.shape[0]
with h5py.File("/home/papalotl/Sara_Goetz/180426/043/NovaCTF/rawtomograms/central_slice.h5") as f:
    f['image'] = dataset[dimz//2, :, :]


