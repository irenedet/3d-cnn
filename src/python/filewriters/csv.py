import csv
import datetime
import re
from os.path import join

from src.python.coordinates_toolbox.utils import rearrange_hdf_coordinates


def motl_writer(path_to_output_folder: str, list_of_peak_scores: list,
                list_of_peak_coords: list):
    """
    Already modified to match em_motl format
    """

    numb_peaks = len(list_of_peak_scores)
    motl_file_name = join(path_to_output_folder,
                          'motl_' + str(numb_peaks) + '.csv')
    with open(motl_file_name, 'w', newline='') as csvfile:
        motlwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        indx = 0
        for val, zyx_coord in reversed(
                sorted(zip(list_of_peak_scores, list_of_peak_coords))):
            indx += 1
            x, y, z = rearrange_hdf_coordinates(zyx_coord)
            motlwriter.writerow([str(val) + ',0,0,' + str(
                indx) + ',0,0,0,' + str(x) + ',' + str(y) + ',' + str(
                z) + ',0,0,0,0,0,0,0,0,0,1'])
    print("The motive list has been writen in", motl_file_name)
    return

def _write_table_header(directory_path: str, param: str,
                        table_writer: csv.writer):
    now = datetime.datetime.now()
    table_writer.writerow([str(now)])
    table_writer.writerow(["From jobs in " + directory_path])
    table_writer.writerow(["CONTENTS"])
    table_writer.writerow(["_job_name"])
    table_writer.writerow(["_K"])
    table_writer.writerow(["_" + param])
    table_writer.writerow(["_classes"])
    table_writer.writerow(["_auPRC"])
    return


def write_jobs_table(directory_path: str, table_name: str, param: str,
                     star_files: list, jobs_statistics_dict: dict):
    table_file_path = join(directory_path, table_name)
    with open(table_file_path, 'w') as csvfile:
        table_writer = csv.writer(csvfile, delimiter=' ',
                                  quotechar='|', quoting=csv.QUOTE_MINIMAL)
        _write_table_header(directory_path, param, table_writer)
        for job_parameters in star_files:
            job_path, k, param_value, classes = job_parameters
            classes = set(classes)
            job_name = re.findall(r"(job\d\d\d)", job_path)[0]
            _, _, _, auPRC, _ = jobs_statistics_dict[job_name]
            row = [job_name, k, param_value, classes, auPRC]
            table_writer.writerow(row)
    return
