# 3d-cnn

## Description.

3D UNet adapted for cryo-ET segmentation and particle localisaton.

## Submission scripts {**script_name**: description}

### Training the UNet

**submission_scripts/training_submission.sh** : given a training dataset (add description), this script outputs a trained model which is saved under the given path. By default in: models/given_name.pkl

### Using a trained UNet to segment a tomogram
#### STEP 1


**submission_scripts/1_dataset2subtomos_submission.sh** : Given a tomogram, the output is a h5 file where the subtomograms of a given size are stored. This is the preparation step to apply the neural network.

**Input variables**:

  in file: repository_path/3d-cnn/runners/dataset2subtomos.py

 - **path_to_raw ** : path to the tomogram that will be segmented

 - **folder_path = "/scratch/trueba/3d-cnn/TEST/"** TODO

 - **h5_output_file** : name of the h5 file where the output will be stored.

      E.g. h5_output_file = "004_in_subtomos_128side.h5"

**Output**:

- a h5 file where the subtomograms in which the original tomogram was partitioned. Each subtomogram is stored in the internal h5 path:

  volumes/raw/subtomo_name

 where  

```
  subtomo_name = "subtomo_" + "_" + str(subtomogram_center)
```

#### STEP 2


**submission_scripts/2_cnn_subtomo_segmentation_submission.sh** : Given a h5 file that contains a series of subtomograms of a compatible size to a given trained neural network, the neural network is applied for segmentation of each subtomogram.

**Input variables**:

  in file: repository_path/3d-cnn/runners/subtomo_segmentation.py

 - **data_dir** : Path to the directory where the h5 file containing the subtomograms to be segmented are stored.

    E.g. data_dir = "/scratch/trueba/3d-cnn/TEST/"

 - **data_file** : A string defining the name of the h5 file where the subtomograms to be segmented are stored.

  Necessarily, the internal path of the subtomograms should be given by

   volumes/raw/subtomo_name

This file is, in principle, the output of Step 1.

 - ** label_name ** : A string that defines the name of the predicted segmentation. By default this is set as "ribosomes".

 - **model_dir** and **model_name_pkl**: then the variable **model_path = model_dir/model_name_pkl** corresponds to the path of the trained and stored UNet model in .pkl format that will be used to segment each subtomogram.

**Output**

 A set of datasets corresponding to the predicted segmentations of each subtomogram stored in the h5 file **data_dir/data_file**: the corresponding predicted segmentation of each subtomogram:
 ```
 volumes/raw/subtomo_name
```
is stored as
```
  volumes/raw/label_name/subtomo_name
```

#### STEP 2 bis (optional)  


**submission_scripts/2bis_subtomos2dataset_submission.sh** : This is an optional (and relatively expensive step). This script runs a program that assembles prediction sub tomograms in an h5 file, to get the global prediction of the full tomogram.


#### STEP 3
**submission_scripts/3_compute_motl_submission.sh** : A motive list is produced from the predictions obtained in step 2.

**Input variables**:

  in file: repository_path/3d-cnn/runners/compute_peaks_motl.py

 - **peaks_numb**?

**Output**

 - The file with path:
   path_to_output_folder/'motl_' + str(numb_peaks) + '.csv'

   where the a motive list associated to the local maxima of the prediction score (in decreasing order) is produced. Where numb_peaks is the total number of local maxima (or peaks) corresponding in the motive list.

## To run in the cluster

```bash
cd path_to_project_repository
sbatch submission_scripts/script_name.sh
```
# Requirements
 ## 1. conda environment
 ### 1.1 Create a conda environment
 #### 1.1.1 In an EMBL system
 In a folder with large enough capacity (e.g. a /g/scb2/zaugg/zaugg_shared):
```bash
module load Anaconda3
conda env create -f environment.yaml -p /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse --force
ln -sv /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse ~/.conda/envs/mlcourse
```
#### 1.1.2 In a local server
```bash
conda env create -f environment.yaml
```
 ### 1.2 Conda environment usage


 To activate this environment, use:

```
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
```

 or, after symbolic link, now it can be invoked as:

 ``` conda activate /home/trueba/.conda/envs/mlcourse ```

 To deactivate an active environment, use:

 ``` source deactivate ```

## 2. Activate modules of this project
Moreover, for running any script, add to the head of the python script:

```python
import sys

py_src_path = "the/local/path/to/the/src/python"
sys.path.append(py_src_path)
runners_path = "/the/local/path/to/3d-cnn/runners"
sys.path.append(runners_path)

```



