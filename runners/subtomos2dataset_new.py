import argparse
from distutils.util import strtobool
from os.path import join

from filewriters.h5 import write_clustering_labels_subtomos, \
    write_dataset_from_subtomos_with_overlap_multiclass
from naming import h5_internal_paths
from utils.cast_types import string_to_list

parser = argparse.ArgumentParser()
parser.add_argument("-output_shape", "--output_shape",
                    help="path to tomogram to be segmented in hdf format",
                    type=str)
parser.add_argument("-box_length", "--box_length",
                    help="subtomogram side length (cubic subtomograms assumed)",
                    type=int)
parser.add_argument("-label_name", "--label_name",
                    help="directory where the outputs will be stored",
                    type=str)
parser.add_argument("-overlap", "--overlap",
                    help="Thickness of subtomo overlap in pixels",
                    type=int)
parser.add_argument("-subtomos_path", "--subtomos_path",
                    help="path to subtomograms file in h5 format",
                    type=str)
parser.add_argument("-class_number", "--class_number",
                    help="class number to re-construct",
                    type=int)
parser.add_argument("-output_path", "--output_path",
                    help="directory where the output hdf file will be stored",
                    type=str)
parser.add_argument("-cluster_labels", "--cluster_labels",
                    help="directory where the output hdf file will be stored",
                    type=str, default=False)
parser.add_argument("-reconstruction_type", "--reconstruction_type",
                    help="either prediction or labels to define reconst. set",
                    type=str, default="prediction")

args = parser.parse_args()
label_name = args.label_name
output_shape = args.output_shape
box_length = args.box_length
overlap_thickness = args.overlap
subtomos_path = args.subtomos_path
class_number = args.class_number
output_path = args.output_path
cluster_labels = strtobool(args.cluster_labels)
subtomo_shape = (box_length, box_length, box_length)
output_shape = string_to_list(string=output_shape, separator=',')
# convert to zyx:
output_shape.reverse()
output_shape = tuple(output_shape)
reconstruction_type = args.reconstruction_type

print(reconstruction_type)
assert reconstruction_type in ["prediction", "labels", "raw"]

if reconstruction_type == "prediction":
    subtomos_internal_path = join(
        h5_internal_paths.PREDICTED_SEGMENTATION_SUBTOMOGRAMS,
        label_name)
elif reconstruction_type == "labels":
    subtomos_internal_path = join(
        h5_internal_paths.LABELED_SUBTOMOGRAMS,
        label_name)
else:
    subtomos_internal_path = h5_internal_paths.RAW_SUBTOMOGRAMS

print(subtomos_internal_path)
if not cluster_labels:
    write_dataset_from_subtomos_with_overlap_multiclass(
        output_path,
        subtomos_path,
        output_shape,
        subtomo_shape,
        subtomos_internal_path,
        class_number,
        overlap_thickness,
        reconstruction_type=reconstruction_type)
else:
    subtomos_internal_path = h5_internal_paths.CLUSTERING_LABELS
    write_clustering_labels_subtomos(
        output_path,
        subtomos_path,
        output_shape,
        subtomo_shape,
        subtomos_internal_path,
        label_name,
        class_number,
        overlap_thickness)
