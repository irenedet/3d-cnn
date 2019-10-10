#! /bin/bash

#SBATCH -A mahamid
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 60G
#SBATCH --time 0-00:20
#SBATCH -o get_shapes.slurm.%N.%j.out
#SBAtCH -e get_shapes.slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irene.de.teresa@embl.de

module load Anaconda3
echo "activating virtual environment"
source activate /struct/mahamid/Processing/envs/.conda/3d-cnn/
echo "... done"


echo "starting python script:"
python3 runners/dataset_tables/get_shapes.py
echo "... done."





#particle_file='/struct/mahamid/Irene/NPC/binary_thin_cylindrical_mask_bin4.em';
#particle = tom_emread(particle_file); particle=particle.Value;
#disp(['Particle of size ', num2str(size(particle))])
#reference = particle;
#motl_file = '/scratch/trueba/data/NPC/allmotl_bin4.txt';
#full_motl = read_motl(motl_file);
#disp(['Motif list of size ', num2str(size(full_motl))])
#
#for tomo_number = linspace(1,35,35)
#    str_tomo_number = num2str(tomo_number, '%02.f');
#    dimxyz = tomos_dimensions(str_tomo_number);
#    disp(['Generating mask for tomo ', num2str(tomo_number),...
#        ' of dimensions ', num2str(dimxyz)]);
#    class = 1;
#    bin = 0;
#    output_mask_file = ['/struct/mahamid/Irene/NPC/',...
#        str_tomo_number, '/clean/thin_cylindrical_mask.em'];
#
#    tom_motl = extract_particles_in_tomo(full_motl, tomo_number);
#    particle_mask = my_tom_classmask(tom_motl, reference, dimxyz, class, bin);
#    tom_emwrite(output_mask_file, particle_mask);
#    disp(['Mask of tomo ', str_tomo_number, ' saved in ',...
#        output_mask_file])
#end
#
#disp('The job has been finished.')


#tomos_motls = containers.Map;
#tomos_motls('1') = '/struct/mahamid/Irene/yeast/healthy/180413/006/clean_motls/npc_motl_clean.csv';
#tomos_motls('2') = '/struct/mahamid/Irene/yeast/healthy/180413/007/clean_motls/npc_motl_clean.csv';
#tomos_motls('3') = '/struct/mahamid/Irene/yeast/healthy/180426/005/clean_motls/npc_motl_clean.csv';
#tomos_motls('4') = '/struct/mahamid/Irene/yeast/healthy/180426/006/clean_motls/npc_motl_clean.csv';
#tomos_motls('5') = '/struct/mahamid/Irene/yeast/healthy/180426/008/clean_motls/npc_motl_clean.csv';
#tomos_motls('6') = '/struct/mahamid/Irene/yeast/healthy/180426/014/clean_motls/npc_motl_clean.csv';
#tomos_motls('7') = '/struct/mahamid/Irene/yeast/healthy/180426/018/clean_motls/npc_motl_clean.csv';
#tomos_motls('8') = '/struct/mahamid/Irene/yeast/healthy/180426/027/clean_motls/npc_motl_clean.csv';
#tomos_motls('9') = '/struct/mahamid/Irene/yeast/healthy/180426/037/clean_motls/npc_motl_clean.csv';
#tomos_motls('10') = '/struct/mahamid/Irene/yeast/healthy/180426/038/clean_motls/npc_motl_clean.csv';
#tomos_motls('11') = '/struct/mahamid/Irene/yeast/healthy/180426/040/clean_motls/npc_motl_clean.csv';
#tomos_motls('12') = '/struct/mahamid/Irene/yeast/healthy/180426/043/clean_motls/npc_motl_clean.csv';
#tomos_motls('13') = '/struct/mahamid/Irene/yeast/healthy/180711/004/clean_motls/npc_motl_clean.csv';
#tomos_motls('14') = '/struct/mahamid/Irene/yeast/healthy/180711/005/clean_motls/npc_motl_clean.csv';
#tomos_motls('15') = '/struct/mahamid/Irene/yeast/healthy/180711/006/clean_motls/npc_motl_clean.csv';
#tomos_motls('16') = '/struct/mahamid/Irene/yeast/healthy/180711/007/clean_motls/npc_motl_clean.csv';
#tomos_motls('17') = '/struct/mahamid/Irene/yeast/healthy/180711/012/clean_motls/npc_motl_clean.csv';
#tomos_motls('18') = '/struct/mahamid/Irene/yeast/healthy/180711/017/clean_motls/npc_motl_clean.csv';
#tomos_motls('19') = '/struct/mahamid/Irene/yeast/healthy/180711/020/clean_motls/npc_motl_clean.csv';
#tomos_motls('20') = '/struct/mahamid/Irene/yeast/healthy/180711/022/clean_motls/npc_motl_clean.csv';
#tomos_motls('21') = '/struct/mahamid/Irene/yeast/healthy/180713/002/clean_motls/npc_motl_clean.csv';
#tomos_motls('22') = '/struct/mahamid/Irene/yeast/healthy/180713/005/clean_motls/npc_motl_clean.csv';
#tomos_motls('23') = '/struct/mahamid/Irene/yeast/healthy/180713/007/clean_motls/npc_motl_clean.csv';
#tomos_motls('24') = '/struct/mahamid/Irene/yeast/healthy/180713/015/clean_motls/npc_motl_clean.csv';
#tomos_motls('25') = '/struct/mahamid/Irene/yeast/healthy/180713/018/clean_motls/npc_motl_clean.csv';
#tomos_motls('26') = '/struct/mahamid/Irene/yeast/healthy/180713/020/clean_motls/npc_motl_clean.csv';
#tomos_motls('27') = '/struct/mahamid/Irene/yeast/healthy/180713/025/clean_motls/npc_motl_clean.csv';
#tomos_motls('28') = '/struct/mahamid/Irene/yeast/healthy/180713/027/clean_motls/npc_motl_clean.csv';
#tomos_motls('29') = '/struct/mahamid/Irene/yeast/healthy/180713/031/clean_motls/npc_motl_clean.csv';
#tomos_motls('30') = '/struct/mahamid/Irene/yeast/healthy/180713/035/clean_motls/npc_motl_clean.csv';
#tomos_motls('31') = '/struct/mahamid/Irene/yeast/healthy/180713/037/clean_motls/npc_motl_clean.csv';
#tomos_motls('32') = '/struct/mahamid/Irene/yeast/healthy/180713/039/clean_motls/npc_motl_clean.csv';
#tomos_motls('33') = '/struct/mahamid/Irene/yeast/healthy/180713/041/clean_motls/npc_motl_clean.csv';
#tomos_motls('34') = '/struct/mahamid/Irene/yeast/healthy/180713/043/clean_motls/npc_motl_clean.csv';
#tomos_motls('35') = '/struct/mahamid/Irene/yeast/healthy/180713/050/clean_motls/npc_motl_clean.csv';
#
#
#tomos_names = containers.Map;
#tomos_names('1') =  '180413/006';
#tomos_names('2') =  '180413/007';
#tomos_names('3') =  '180426/005';
#tomos_names('4') =  '180426/006';
#tomos_names('5') =  '180426/008';
#tomos_names('6') =  '180426/014';
#tomos_names('7') =  '180426/018';
#tomos_names('8') =  '180426/027';
#tomos_names('9') =  '180426/037';
#tomos_names('10') = '180426/038';
#tomos_names('11') = '180426/040';
#tomos_names('12') = '180426/043';
#tomos_names('13') = '180711/004';
#tomos_names('14') = '180711/005';
#tomos_names('15') = '180711/006';
#tomos_names('16') = '180711/007';
#tomos_names('17') = '180711/012';
#tomos_names('18') = '180711/017';
#tomos_names('19') = '180711/020';
#tomos_names('20') = '180711/022';
#tomos_names('21') = '180713/002';
#tomos_names('22') = '180713/005';
#tomos_names('23') = '180713/007';
#tomos_names('24') = '180713/015';
#tomos_names('25') = '180713/018';
#tomos_names('26') = '180713/020';
#tomos_names('27') = '180713/025';
#tomos_names('28') = '180713/027';
#tomos_names('29') = '180713/031';
#tomos_names('30') = '180713/035';
#tomos_names('31') = '180713/037';
#tomos_names('32') = '180713/039';
#tomos_names('33') = '180713/041';
#tomos_names('34') = '180713/043';
#tomos_names('35') = '180713/050';
#
#
#particle_file='/struct/mahamid/Irene/NPC/binary_thin_cylindrical_mask_bin4.em';
#particle = tom_emread(particle_file); particle=particle.Value;
#disp(['Particle of size ', num2str(size(particle))])
#reference = particle;
#% motl_file = '/scratch/trueba/data/NPC/allmotl_bin4.txt';
#% disp(['Motif list of size ', num2str(size(full_motl))])
#dimxyz = [960 928 500];
#for tomo_number = [1] %linspace(1,35,35)
#    str_tomo_number = num2str(tomo_number);
#    motl_file=tomos_motls(str_tomo_number);
#    motl = read_motl(motl_file)';
#    disp(['Motif list of size ', num2str(size(motl))])
#
#    tomo_name = tomos_names(str_tomo_number);
#    disp(['Generating mask for tomo ', tomo_name,...
#        ' of dimensions ', num2str(dimxyz)]);
#    class = 1;
#    bin = 0;
#    output_folder = ['/struct/mahamid/Irene/yeast/healthy/',...
#        tomo_name, '/clean_masks']
#    output_mask_file = [output_folder, 'npc_cylindrical_mask.em'];
#    mkdir output_folder
#
#    particle_mask = my_tom_classmask(motl, reference, dimxyz, class, bin);
#    tom_emwrite(output_mask_file, particle_mask);
#    disp(['Mask of tomo ', str_tomo_number, ' saved in ',...
#        output_mask_file])
#end
#
#disp('The job has been finished.')


#/struct/mahamid/Irene/yeast/healthy/180413/006/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180413/007/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180426/005/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180426/006/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180426/008/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180426/014/clean_masksnpc_cylindrical_mask.em <- check!
#/struct/mahamid/Irene/yeast/healthy/180426/018/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180426/027/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180426/037/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180426/038/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180426/040/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180426/043/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180711/004/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180711/005/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180711/006/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180711/007/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180711/012/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180711/017/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180711/020/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180711/022/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/002/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/005/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/007/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/015/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/018/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/020/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/025/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/027/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/031/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/035/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/037/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/039/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/041/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/043/clean_masksnpc_cylindrical_mask.em
#/struct/mahamid/Irene/yeast/healthy/180713/050/clean_masksnpc_cylindrical_mask.em


