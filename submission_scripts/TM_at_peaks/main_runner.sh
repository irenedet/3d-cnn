#!/usr/bin/env bash


####### Energy depleted tomograms:

TOMO_NAMES=( "181119_002"
             "181119_030"
             "181126_002"
             "181126_012"
             "181126_025"
             "190301_005" )

DATASETS=( "/struct/mahamid/Irene/yeast/ED/181119_002/eman_filt_eman_filt_002_sq_df_sorted.hdf"
           "/struct/mahamid/Irene/yeast/ED/181119_030/eman_filt_eman_filt_030_sq_df_sorted.hdf"
           "/struct/mahamid/Irene/yeast/ED/181126_002/eman_filt_eman_filt_002_sq_df_sorted.hdf"
           "/struct/mahamid/Irene/yeast/ED/181126_012/eman_filt_eman_filt_012_sq_df_sorted.hdf"
           "/struct/mahamid/Irene/yeast/ED/181126_025/eman_filt_eman_filt_025_sq_df_sorted.hdf"
           "/struct/mahamid/Irene/yeast/ED/190301_005/eman_filt_eman_filt_005_sq_df_sorted.hdf" )

particle='fas' # fas or ribo


if [ "$particle" == 'ribo' ]
then
    echo 'Analyzing ribo motls in energy depleted tomograms.'
    ############## For RIBOSOMES:
    export MOTLS=(  "/struct/mahamid/Irene/yeast/ED/181119_002/motl/RR_ribo_motl_pred.csv"
                    "/struct/mahamid/Irene/yeast/ED/181119_030/motl/RR_ribo_motl_pred.csv"
                    "/struct/mahamid/Irene/yeast/ED/181126_002/motl/RR_ribo_motl_pred.csv"
                    "/struct/mahamid/Irene/yeast/ED/181126_012/motl/RR_ribo_motl_pred.csv"
                    "/struct/mahamid/Irene/yeast/ED/181126_025/motl/RR_ribo_motl_pred.csv"
                    "/struct/mahamid/Irene/yeast/ED/190301_005/motl/RR_ribo_motl_pred.csv" )

    export catalogue_path="/struct/mahamid/Irene/yeast/catalogues/4b_ribo_catalogue.h5"
    export reference_rotation_angles_file="/g/easybuild/x86_64/CentOS/7/haswell/software/PyTom/0.971-foss-2016b-Python-2.7.12/pytom/angles/angleLists/angles_19.95_1944.em"
    export angles_in_degrees='False'
    export output_name='RR_ribo_motl_pred_with_angles.csv'
    export path_to_mask="/struct/mahamid/Irene/yeast/templates/mask_1b_32.em"

elif [ "$particle" == 'fas' ]
then
    echo 'Analyzing fas motls in energy depleted tomograms.'
    ############## For FAS:
    export MOTLS=(  "/struct/mahamid/Irene/yeast/ED/181119_002/motl/motl_ED_FAS_4b.csv"
                    "/struct/mahamid/Irene/yeast/ED/181119_030/motl/motl_ED_FAS_4b.csv"
                    "/struct/mahamid/Irene/yeast/ED/181126_002/motl/motl_ED_FAS_4b.csv"
                    "/struct/mahamid/Irene/yeast/ED/181126_012/motl/motl_ED_FAS_4b.csv"
                    "/struct/mahamid/Irene/yeast/ED/181126_025/motl/motl_ED_FAS_4b.csv"
                    "/struct/mahamid/Irene/yeast/ED/190301_005/motl/motl_ED_FAS_4b.csv" )


    export catalogue_path="/struct/mahamid/Irene/yeast/catalogues/4b_fas_catalogue.h5"
    export reference_rotation_angles_file="/g/easybuild/x86_64/CentOS/7/haswell/software/PyTom/0.971-foss-2016b-Python-2.7.12/pytom/angles/angleLists/angles_19.95_1944.em"
    export angles_in_degrees='False'
    export output_name='motl_ED_FAS_4b_with_angles.csv'
    export path_to_mask="/struct/mahamid/Irene/yeast/templates/mask_FAS_4b_thr.hdf"

else
    echo "Non valid particle."
fi


for ((i=0;i<${#TOMO_NAMES[@]};++i)); do
    export tomo_name="${TOMO_NAMES[i]}"
    export path_to_motl="${MOTLS[i]}"
    export path_to_dataset="${DATASETS[i]}"
    export output_dir=$(dirname "${path_to_motl}")
    export path_to_output_csv=$output_dir"/"$output_name

    echo "Submitting job for $path_to_motl"
    sbatch /g/scb2/zaugg/trueba/3d-cnn/submission_scripts/TM_at_peaks/TM_submission.sh -tomo_name $tomo_name -path_to_motl $path_to_motl -path_to_mask $path_to_mask -path_to_dataset $path_to_dataset -path_to_output_csv $path_to_output_csv -catalogue_path $catalogue_path -ref_angles $reference_rotation_angles_file -angles_in_degrees $angles_in_degrees
done

