import argparse
from distutils.util import strtobool

import pandas as pd

from tomogram_utils.volume_actions.actions import partition_tomogram
from file_actions.readers.hdf import _load_hdf_dataset

parser = argparse.ArgumentParser()

parser.add_argument("-outh5", "--output_h5_path",
                    help="file where the outputs will be stored",
                    type=str)
parser.add_argument("-box", "--box_side",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-overlap", "--overlap",
                    help="name of category to be segmented",
                    type=int)
parser.add_argument("-tomo_name", "--tomo_name",
                    help="name of tomogram in format sessionname/datasetnumber",
                    type=str)
parser.add_argument("-dataset_table", "--dataset_table",
                    help="path to dataset table",
                    type=str)
parser.add_argument("-write_on_table", "--write_on_table",
                    help="write_on_table if True will store the partition path",
                    type=str, default="False")

args = parser.parse_args()
dataset_table = args.dataset_table
tomo_name = args.tomo_name
write_on_table = strtobool(args.write_on_table)
output_h5_file_path = args.output_h5_path
box_side = args.box_side
overlap = args.overlap

df = pd.read_csv(dataset_table)
df['tomo_name'] = df['tomo_name'].astype(str)
tomo_df = df[df['tomo_name'] == tomo_name]
path_to_raw = tomo_df.iloc[0]['eman2_filetered_tomo']

subtomogram_shape = (box_side, box_side, box_side)

raw_dataset = _load_hdf_dataset(hdf_file_path=path_to_raw)
partition_tomogram(dataset=raw_dataset, output_h5_file_path=output_h5_file_path,
                   subtomo_shape=subtomogram_shape, overlap=overlap)

if write_on_table:
    df.loc[df['tomo_name'] == tomo_name, 'test_partition'] = output_h5_file_path
