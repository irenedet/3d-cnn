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



