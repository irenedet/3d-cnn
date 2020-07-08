# 3d-cnn

## 1. Description.

3D UNet adapted for cryo-ET segmentation and particle localisaton.

## 2. Usage
### 2.1 Submission scripts 


### 2.2 Run in the cluster

```bash
cd path_to_project_repository
sbatch submission_scripts/script_name.sh
```
## 3. Requirements and conda environment


### 3.0 Package Installation (miniconda and torch)
Install locally the packages (both in the cluster or other):

#### Miniconda
```bash
cd foldertodownload
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

#### pytorch and torchvision for gpus:
```bash
conda install -c pytorch pytorch torchvision
```

#### Switch the global variable to make conda installations in a large folder, 
by adding the paths to the ~/.bashrc or ~/.profile files:

```bash
# If this is the first time you install a venv,
# make sure to locate it in a large capacity folder

ln -s  /struct/mahamid/Irene/envs/.conda /home/trueba
export CONDA_PKGS_DIRS=/struct/mahamid/Irene/envs/.conda
```

### 3.1 Create the conda virtual environment

Now create the virtual environment with all requirements:
(important to put the prefix directory, othwise it will try to install in home
where the memory space may be limited)

```bash
conda env create --file environment.yaml -p /folder/large/capacity/.conda/envs/3d-cnn --force
```
and set the virtual environment path in `~/.bashrc` or `~/.profile`:
```bash
UPICKER_PATH='/path/to/this/repo'
UPICKER_VENV_PATH='/folder/large/capacity/.conda/envs/3d-cnn'
```

### 3.2 Conda environment usage


 To activate this environment, use:

```
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
```

 or, after symbolic link, now it can be invoked as:

 ``` conda activate /home/trueba/.conda/envs/mlcourse ```

 To deactivate an active environment, use:

 ``` source deactivate ```

### 3.3 GPU usage

check gpu current state/availability
```bash
nvidia-smi
```
the output should be smthg like 
```bash
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 440.87       Driver Version: 440.87       CUDA Version: 10.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GRID P40-12Q        On   | 00000000:02:00.0 Off |                  N/A |
| N/A   N/A    P8    N/A /  N/A |  11957MiB / 12288MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     10078      C   ...ttausc/miniconda3/envs/keras/bin/python 11109MiB |
+-----------------------------------------------------------------------------+
```

### 3.4 Activate parallel use of GPUs to handle data

```
import torch

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)
    # DataParallel splits your data automatically and
    # sends job orders to multiple models on several GPUs.
    # After each model finishes their job, DataParallel
    # collects and merges the results before returning it to you.
model.to(device)  # currently the device is in one single node
```

## Appendix. Activate modules of this project
Moreover, for running any script, add to the head of the python script:

```
import sys

py_src_path = "the/local/path/to/the/src/python"
sys.path.append(py_src_path)
runners_path = "/the/local/path/to/3d-cnn/runners"
sys.path.append(runners_path)

```

## 4. Notes for organelle segmentation

### 4.1 Package Installation
#### 1. Preparation step
For the installation, comment all modifications to PYTHONPATH in 
the ~/.bashrc, ~/.bash_profile and ~/.profile files (and source them after that).


#### 2. miniconda 

Install miniconda3 in the cluster node or locally:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
During installation, set the miniconda3 in a large folder, e.g.

```bash
/g/scb/zaugg/zaugg_shared/.venvs/Irene/miniconda3
```
then link to the home folder:

```bash
ln -s /g/scb/zaugg/zaugg_shared/.venvs/Irene/miniconda3 /home/trueba
```
you will see `/home/trueba/miniconda3` in the home folder.

All the venvs created will be then placed in `/g/scb/zaugg/zaugg_shared/.venvs/Irene/miniconda3/envs/`

Also, create a large folder where the packages will be installed by conda
when creating new venvs and link it to the home folder:
```bash
mkdir /g/scb/zaugg/zaugg_shared/.venvs/Irene/.conda
ln -s /g/scb/zaugg/zaugg_shared/.venvs/Irene/.conda /home/trueba
```

Additionally, switch the global variable to make conda installations in that large folder, 
by adding to the ~/.bashrc or ~/.profile files, and:
```bash
export CONDA_PKGS_DIRS=/g/scb/zaugg/zaugg_shared/.venvs/Irene/.conda
```

#### 3. keras (and tensorflow) for gpus

Install the right version of keras-gpu (which also installs tensorflow):
```bash
conda install -c anaconda keras-gpu=2.3.1
#check whether it's better to install keras to match python 3.6??
```

Check the installation by importing tensorflow in a python console.


#### 4. Generate the virtual environment 

The necessary virtual environment for the installation needs only snakemake 5.3 and pandas

```bash
conda create -c conda-forge -c bioconda -n snakemake-pandas snakemake pandas
conda activate snakemake-pandas
```
Check that the version of snakemake is 5.3.0

Later, when running the pipeline, snakemake will generate the keras virtual environment with the rest of 
the necessary packages.

#### 5. Testing the pipeline with --dryrun

After activating the venv snakemake-pandas with:

```bash
conda activate snakemake-pandas
```

check whether snakemake can build the directed graph for the 
pipeline:

in deploy_cluster.sh or deploy_local.sh, add the --dryrun flag:

```bash
srun -t 4:00:00 -c 1 --mem 2G \
    snakemake \
    --snakefile "${srcdir}/snakefile.py" \
    --cluster "sbatch" \
    --config config="${config_file}" \
    --jobscript "${srcdir}/jobscript.sh" \
    --jobs 20 \
    --use-conda \
    --printshellcmds \
    --latency-wait 30 --dryrun #to not execute any rule with dryrun
```

and if it does, then test the pipeline by deleting --dryrun:

```bash
bash deploy_cluster.sh config.yaml
```

The pipeline will generate automatically in the running directory a 
hidden folder .snakemake, where the venv and the snakemake locks will
be stored.


### 4.4 Activate/deactivate modes

There are three modes:
- Training-evaluation
- Training-Production
- Non Training-Predict

In the file `config.yaml` just set the `active` parameters as corresponding (True or False).

### 4.5 Notes on formats

- Label files are piece-wise constant maps in .mrc format. The id value - semantic 
class correspondence is predetermined in the file `labels.csv`.
- The `train_metadata.csv` and `yeast_sara_short.csv` have the information for 
the training.

### About permissions:

to change and allow reading:

```bash
chmod 774 /path/to/dir
```


```python
# import sys
# print(sys.path)
# [
# '', 
# '/struct/mahamid/Irene/3d-cnn', 
# '/struct/mahamid/Irene/3d-cnn/src/python', 
# '/struct/mahamid/Irene/organelle_detection_local/new_test/.snakemake/conda/b7da5f31/lib/python36.zip', 
# '/struct/mahamid/Irene/organelle_detection_local/new_test/.snakemake/conda/b7da5f31/lib/python3.6', 
# '/struct/mahamid/Irene/organelle_detection_local/new_test/.snakemake/conda/b7da5f31/lib/python3.6/lib-dynload', 
# '/home/trueba/.local/lib/python3.6/site-packages', 
# '/struct/mahamid/Irene/organelle_detection_local/new_test/.snakemake/conda/b7da5f31/lib/python3.6/site-packages', 
# '/struct/mahamid/Irene/organelle_detection_local/new_test/.snakemake/conda/b7da5f31/lib/python3.6/site-packages/pysftp-0.2.9-py3.6.egg', 
# '/struct/mahamid/Irene/organelle_detection_local/new_test/.snakemake/conda/b7da5f31/lib/python3.6/site-packages/rsa-3.1.4-py3.6.egg']
# 
# ['/struct/mahamid/Irene/organelle_detection_local/new_test/.snakemake/conda/b7da5f31/lib/python36.zip','/struct/mahamid/Irene/organelle_detection_local/new_test/.snakemake/conda/b7da5f31/lib/python3.6', '/struct/mahamid/Irene/organelle_detection_local/new_test/.snakemake/conda/b7da5f31/lib/python3.6/lib-dynload', '/struct/mahamid/Irene/organelle_detection_local/new_test/.snakemake/conda/b7da5f31/lib/python3.6/site-packages', '/struct/mahamid/Irene/organelle_detection_local/new_test/.snakemake/conda/b7da5f31/lib/python3.6/site-packages/pysftp-0.2.9-py3.6.egg', '/struct/mahamid/Irene/organelle_detection_local/new_test/.snakemake/conda/b7da5f31/lib/python3.6/site-packages/rsa-3.1.4-py3.6.egg'
```

