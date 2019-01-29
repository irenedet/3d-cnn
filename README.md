# 3d-cnn
## Descripton.
UNet adapted for cryo-ET segmentation.

## Submission scripts {**script_name**: description}

### Training the UNet
  **submission_scripts/training_submission.sh** : given a training dataset (add description), this script outputs a trained model which is saved under the given path. By default in: models/given_name.pkl

### Using a trained UNet to segment a tomogram
  STEP 1
  **submission_scripts/1_dataset2subtomos_submission.sh** : Given a tomogram, the output is a h5 file where the subtomograms of a given size are stored. This is the preparation step to apply the neural network.

 STEP 2
 **submission_scripts/2_cnn_subtomo_segmentation_submission.sh** : Given a h5 file that contains a series of subtomograms of a compatible size to a given trained neural network, the neural network is applied for segmentation of each subtomogram.
  The predictions are stored in the same file, with the route:
  subtomos/predictions/segmentation_name. By default, segmentation_name = "ribosomes".

  STEP 2 bis (optional)
  **submission_scripts/2bis_subtomos2dataset_submission.sh** : This is an optional (and relatively expensive step). This script runs a program that assembles prediction sub tomograms in an h5 file, to get the global prediction of the full tomogram.


   
   **submission_scripts/3_compute_motl_submission.sh** : A motive list is produced from the predictions obtained in step 2.
