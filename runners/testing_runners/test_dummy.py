import numpy as np
# from src.python.filereaders.datasets import load_dataset
from src.python.filewriters.h5 import write_dataset_hdf

# lamella_file = "/struct/mahamid/Irene/liang_data/lamella.hdf"
# lamella = np.ones((450, 928, 928))
# write_dataset_hdf(lamella_file, lamella)
#
# TOMOS = ["173", "172", "174", "175", "176", "177", "178", "179", "180", "190",
#          "191", "192", "193", "194", "195", "196", "198", "199", "200", "201",
#          ]






model_name = "fjdgbjsdbame"
model_path_pkl = "msahfdh"
log_model = "log_model"
depth = 33
initial_features = 2
n_epochs = 30
training_paths_list = ["a", "b"]
split = 0.7
output_classes = 3
segmentation_names = ["a1", "2w"]
retrain = True
path_to_old_model = "path_to_old_model"
models_notebook = "/home/papalotl/Downloads/dummy.csv"
from src.python.filewriters.csv import write_on_models_notebook
write_on_models_notebook(model_name, model_path_pkl, log_model, depth,
                         initial_features, n_epochs, training_paths_list,
                         split, output_classes, segmentation_names, retrain,
                         path_to_old_model, models_notebook)
