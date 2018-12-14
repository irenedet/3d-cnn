import re
from os import listdir
from os.path import isdir, join

import numpy as np

from src.python.calculator.statistics import precision_recall_calculator, pr_auc_score, \
    F1_score_calculator
from src.python.filereaders.star import class3d_data_file_reader


def get_jobs_list(directory_path: str):
    jobs_list = [f for f in listdir(directory_path) if
                 (isdir(join(directory_path, f)) and f[0] != '.')]
    return jobs_list


def _read_job_parameters(job_note_file, param):
    with open(job_note_file, 'r') as note:
        contents = note.read()
        K = re.findall(r"K\s(\d+)", contents)[0][-1:]
        if param == 'diam':
            par = re.findall(r"--particle_diameter\s(\d+)", contents)[0]
        elif param == 'tau':
            par = re.findall(r"--tau2_fudge\s(\d+(\.\d+)?)", contents)[0][0]
        elif param == 'ref':
            par = re.search('--ref(.+?) --', contents).group(1)
            par = par[-16:]
            if par == "run_class001.mrc":
                par = "relion"
            else:
                par = "sph"
        else:
            par = "None"
            raise Warning(
                'Not among valid param values; choose from: diam, tau, ref')
    return K, par


def get_job_parameters(directory_path: str, param: str):
    jobs_parameters = []
    jobs_list = np.sort(get_jobs_list(directory_path))
    for i in range(len(jobs_list)):
        job = jobs_list[i]
        job_note = join(directory_path, job)
        job_note = join(job_note, 'note.txt')
        K, par = _read_job_parameters(job_note, param)
        jobs_parameters += [(join(directory_path, job), K, par)]
    return jobs_parameters


def _add_star_file_path_and_classes(job_parameters: tuple, classes_dict: dict):
    job_name = job_parameters[0][-6:]
    classes = classes_dict[job_name]
    star_file_path = join(job_parameters[0], "run_it025_data.star")
    return star_file_path, job_parameters[1], job_parameters[2], classes


def create_star_files_list(jobs_parameters: list, classes_dict: dict):
    star_files_list = [
        _add_star_file_path_and_classes(job_parameters, classes_dict)
        for job_parameters in jobs_parameters]
    return star_files_list


def _extract_coordinates_per_class(data_list: list, classes: list) -> list:
    coords = [[float(line[1]), float(line[2]), float(line[3])] for line in
                   data_list if (int(line[13]) in classes)]
    return coords


def generate_jobs_statistics_dict(star_files: list, motl_clean_coords,
                                  radius: float) -> dict:
    plots_dict = {}
    for job in star_files:
        star_name, K, par, classes = job
        data_list = class3d_data_file_reader(star_name)

        coords = _extract_coordinates_per_class(data_list, classes=classes)
        precision, recall, detected_clean = precision_recall_calculator(
            coords,
            motl_clean_coords,
            radius=radius)

        auPRC = pr_auc_score(precision=precision, recall=recall)

        F1_score = F1_score_calculator(precision, recall)

        job_name = re.findall(r"job\d\d\d", star_name)[0]
        legend_str = job_name + ' K ' + K + ', param ' + par + ', classes ' + \
                     str(set(classes))

        plots_dict[job_name] = [precision, recall, F1_score, auPRC, legend_str]
    return plots_dict


def get_particle_index_and_class(data_row, particle_regex = r"par_(\d+).mrc"):
    particle_path =  data_row[4]
    particle_index = int(re.findall(particle_regex,particle_path)[0])
    particle_class = int(data_row[13])
    return particle_index, particle_class


def get_particle_indices_in_classes(particles_indices_and_classes: list,
                                    classes: list):
    particle_indices_in_classes = [particle[0] for particle in
                                   particles_indices_and_classes if
                                   particle[1] in classes]
    return particle_indices_in_classes


def get_particles_list(old_star_path:str):
    with open(old_star_path, 'r') as star_file:
        particles_list = [l for l in (line.strip() for line in star_file) if l]
        particles_list = [l.split() for l in particles_list]
    return particles_list[15:]


def get_list_of_indices_and_classes(data_list: list) -> list:
    return [get_particle_index_and_class(data_row) for data_row in data_list]