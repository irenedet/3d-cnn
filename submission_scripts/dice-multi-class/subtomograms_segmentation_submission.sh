#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 32G
#SBATCH --time 0-10:00
#SBATCH -o slurm.%N.%j.out
#SBAtCH -e slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de



module load Anaconda3
echo "activating virtual environment"
source activate /g/scb2/zaugg/zaugg_shared/Programs/Anaconda/envs/irene/.conda/envs/mlcourse
echo "... done"

export PYTHONPATH=$PYTHONPATH:/g/scb2/zaugg/trueba/3d-cnn


echo "running python 3 script"
python3 /g/scb2/zaugg/trueba/3d-cnn/pipelines/dice_multi-class/evaluation/cnn_subtomo_segmentation.py

#extract_parameter_from_label(label, 'frac')
#
#plot_df = pd.DataFrame({})
#indx_shuffle_no_shuffle = 10
#for label in stats_df.keys():
#    if label not in ['tomo_name']:
#        print(label)
#        frac = extract_parameter_from_label(label, 'frac')
#        D = extract_parameter_from_label(label, 'D')
#        IF = extract_parameter_from_label(label, 'IF')
#        assert len(stats_df.loc[stats_df[label].notnull()][label].values) <= 2
#        print(stats_df.loc[stats_df[label].notnull()][label].values)
#        for inedx, tomo in enumerate(['004', '005']):
#            tomo_name = tomo
#            miniplot_df=pd.DataFrame({})
#            miniplot_df['tomogram'] = [tomo]
#            if len(stats_df.loc[stats_df[label].notnull()][label].values) >= inedx + 1:
#                miniplot_df['auPRC'] = [stats_df.loc[stats_df[label].notnull()][label].values[inedx]]
#            else:
#                miniplot_df['auPRC'] = np.nan
#            miniplot_df['D'] = int(D)
#            miniplot_df['IF'] = int(IF)
#            miniplot_df['frac'] = int(frac)
#            if label in labels:
#                miniplot_df['shuffle'] ='no_shuffle' + ", D" + str(D)
#            else:
#                miniplot_df['shuffle'] ='shuffle' + ", D" + str(D)
#            #mini_plot_df = pd.DataFrame(miniplot_df)
#            plot_df = plot_df.append(miniplot_df, sort=False)