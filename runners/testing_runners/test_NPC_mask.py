# center = (194, 450, 308)
# dataset = np.zeros((300, 928, 928))
#
# psi = 60
# theta = 15
# sigma = 86
#
# ZXZ_angles = (psi, theta, sigma)
# # ZXZ_angles = (sigma, theta, psi)
#
# dataset = paste_rotated_disk(dataset, center, radius, thickness, ZXZ_angles)
#
#
#
# center = (118, 468, 577)
# psi = 269
# theta = 2
# sigma = 194
#
# ZXZ_angles = (psi, theta, sigma)
# dataset = paste_rotated_disk(dataset, center, radius, thickness, ZXZ_angles)
import csv

import numpy as np

from file_actions.writers.h5 import write_dataset_hdf
from peak_toolbox.utils import paste_rotated_disk

# thickness = 35
# radius = 65
# # center = (194, 450, 308)
# center = (150, 150, 150)
# dataset = np.zeros((300, 928, 928))
#
# psi = 0
# theta = 90
# sigma = 0
#
# ZXZ_angles = (psi, theta, sigma)
#
# dataset = paste_rotated_disk(dataset, center, radius, thickness, ZXZ_angles)
#
# center = (150, 150, 300) #zyx format
#
# psi = 90
# theta = 0
# sigma = 0
#
# ZXZ_angles = (psi, theta, sigma)
#
# dataset = paste_rotated_disk(dataset, center, radius, thickness, ZXZ_angles)
#
# center = (150, 150, 450) #zyx format
#
# psi = 0
# theta = 90
# sigma = 90
motl = np.zeros((300, 928, 928))
motl = []
with open("/scratch/trueba/data/NPC/allmotl_bin4.txt", "r") as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        data = np.array([float(val) for val in row])
        motl += [data]
motl = np.array(motl)

motl = motl[:, 2:9]
thickness = 35
radius = 65
for n in range(motl.shape[0] - 1):
    center = motl[2:5, n]
    center = tuple([int(c) for c in center[::-1]])
    ZXZ_angles = tuple(motl[2:5, n])
    print(center)
    print(ZXZ_angles)
    motl = paste_rotated_disk(motl, center, radius, thickness, ZXZ_angles)

# thickness = 35
# radius = 65
# center = (194, 450, 308)
# dataset = np.zeros((300, 928, 928))
#
# psi = 60
# theta = 15
# sigma = 86
#
# ZXZ_angles = (psi, theta, sigma)
# # ZXZ_angles = (sigma, theta, psi)
#
# dataset = paste_rotated_disk(dataset, center, radius, thickness, ZXZ_angles)
#
#
#
# center = (118, 468, 577)
# psi = 269
# theta = 2
# sigma = 194
#
# ZXZ_angles = (psi, theta, sigma)
# dataset = paste_rotated_disk(dataset, center, radius, thickness, ZXZ_angles)
#
write_dataset_hdf(output_path="/home/papalotl/npc_test002_bis_1.hdf",
                  tomo_data=motl)
