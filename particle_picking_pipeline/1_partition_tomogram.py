from os.path import join
from os import makedirs
import argparse

from src.python.datasets.actions import partition_tomogram
from src.python.filereaders.hdf import _load_hdf_dataset
from src.python.osactions.filesystem import extract_file_name

parser = argparse.ArgumentParser()
parser.add_argument("-raw", "--path_to_raw",
                    help="path to tomogram to be segmented in hdf format",
                    type=str)
parser.add_argument("-output", "--output_dir",
                    help="directory where the outputs will be stored",
                    type=str)

args = parser.parse_args()
path_to_raw = args.path_to_raw
output_dir = args.output_dir

makedirs(name=output_dir, exist_ok=True)

tomo_name = extract_file_name(path_to_file=path_to_raw)
output_h5_file_name = tomo_name + "_subtomograms.h5"
output_h5_file_path = join(output_dir, output_h5_file_name)
subtomogram_shape = (128, 128, 128)
overlap = 12

raw_dataset = _load_hdf_dataset(hdf_file_path=path_to_raw)
partition_tomogram(dataset=raw_dataset, output_h5_file_path=output_h5_file_path,
                   subtomo_shape=subtomogram_shape,
                   overlap=overlap)
exit(output_h5_file_path)
