import matplotlib.pyplot as plt
from tomogram_utils.peak_toolbox.utils import read_motl_coordinates_and_values

from relion_toolbox.utils import get_job_parameters, \
    generate_jobs_statistics_dict, create_star_files_list, \
    get_job_name_from_string

""" Dummy display of how to use relion tools to compare a motl_true_coords list 
to a list of coordinates in certain classes of the relion Class3D algorithm.
"""
path_to_motl_true = "/struct/mahamid/Irene/yeast/healthy/180426/004/ribos/motl/corrected_motl_verified.csv"
directory_path = "/struct/mahamid/Sara_Goetz/Data/Titan/Processing/180426/004/relion/polysomes/Class3D"
param_name = 'tau'
pr_radius = 10
z_shift = -16

# The jobs and classes to compare to true coordinates(these classes were chosen
# randomly):
classes_dict = {
    "job013": [1, 2, 3, 4, 5],
    # "job016": [2, 3, 4, 5],
    # "job017": [1, 2, 4, 5],
    # "job018": [1],
    # "job023": [1],
    # "job025": [1],
    # "job026": [1],
}

jobs_parameters = get_job_parameters(directory_path=directory_path,
                                     param=param_name)
print(jobs_parameters)

true_values, true_coordinates = read_motl_coordinates_and_values(
    path_to_motl=path_to_motl_true)

# Shift coordinates
for coordinate in true_coordinates:
    coordinate[2] = coordinate[2] + z_shift

new_jobs_parameters = []
for job_parameters in jobs_parameters:
    star_file, *rest = job_parameters
    job_name = get_job_name_from_string(string=star_file)
    if job_name in classes_dict.keys():
        new_jobs_parameters.append(job_parameters)
    else:
        print("Ignoring", job_name)

star_files_list = create_star_files_list(jobs_parameters=new_jobs_parameters,
                                         classes_dict=classes_dict)
plots_dict = generate_jobs_statistics_dict(star_files=star_files_list,
                                           motl_clean_coords=true_coordinates,
                                           radius=pr_radius,
                                           param_name=param_name)
for job_name in classes_dict.keys():
    precision, recall, F1_score, auPRC, legend_str = plots_dict[job_name]
    plt.plot(recall, precision, label=legend_str)

plt.legend()
plt.ylim([0, 1.1])
plt.xlim([0, 1.1])
plt.show()
