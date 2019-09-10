import numpy as np
# from src.python.filereaders.datasets import load_dataset
from src.python.filewriters.h5 import write_dataset_hdf

fraction_files = [
    "004_fraction_0.h5",
    "004_fraction_1.h5",
    "004_fraction_2.h5",
    "004_fraction_3.h5",
    "004_fraction_4.h5",
    "005_fraction_0.h5",
    "005_fraction_1.h5",
    "005_fraction_2.h5",
    "005_fraction_3.h5",
    "005_fraction_4.h5",
    # "021_fraction_0.h5",
    # "021_fraction_1.h5",
    # "021_fraction_2.h5",
    # "021_fraction_3.h5",
    # "021_fraction_4.h5",
]

input_dir = "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/"

output_files = [
    "fraction_0.h5",
    "fraction_1.h5",
    "fraction_2.h5",
    "fraction_3.h5",
    "fraction_4.h5",
]
a = [
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/004_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/004_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/004_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/004_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/004_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/005_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/005_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/005_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/005_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/005_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/021_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/021_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/021_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/021_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_0/021_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/004_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/004_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/004_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/004_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/004_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/005_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/005_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/005_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/005_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/005_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/021_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/021_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/021_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/021_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_1/021_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/004_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/004_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/004_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/004_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/004_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/005_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/005_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/005_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/005_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/005_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/021_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/021_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/021_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/021_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_2/021_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/004_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/004_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/004_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/004_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/004_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/005_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/005_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/005_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/005_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/005_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/021_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/021_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/021_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/021_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_3/021_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/004_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/004_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/004_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/004_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/004_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/005_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/005_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/005_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/005_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/005_fraction_4.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/021_fraction_0.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/021_fraction_1.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/021_fraction_2.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/021_fraction_3.h5"
    "/scratch/trueba/3d-cnn/training_data/dice-multi-class/cross-validation/training_fraction_4/021_fraction_4.h5"
]
