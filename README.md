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
Install locally the following packages in the cluster or locally:

4.1.1 miniconda 
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Switch the glogal variable to make conda installations in a large folder, by adding
to the ~/.bashrc or ~/.profile files:
```bash
export CONDA_PKGS_DIRS=/struct/mahamid/Irene/envs/.conda
```

4.1.2 keras and tensorflow for gpus

ToDo
```bash
conda install -c anaconda keras-gpu
```

### 4.2 Generate the virtual environment with only snakemake and pandas

```bash
conda create -c conda-forge -c bioconda -n snakemake-demo python=3.6
conda activate snakemake-demo
conda install pandas
conda install -c bioconda snakemake
```

Later, when running the pipeline, a virtual environment with the rest of the packages will 
created.

### 4.3 Dryrun

After activating snakemake-demo do:

First check whether snakemake can build the directed graph:

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

and it is ran by:

```bash
bash deploy_cluster.sh config.yaml
```

Obviously, to run the code, just delete the --dryrun flag.

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




