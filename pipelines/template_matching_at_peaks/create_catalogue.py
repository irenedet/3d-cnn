from src.python.template_matching.tools import create_template_catalogue

angles_file = "/g/easybuild/x86_64/CentOS/7/haswell/software/PyTom/0.971-foss-2016b-Python-2.7.12/pytom/angles/angleLists/angles_19.95_1944.em"

ribo_catalogue_path = "/struct/mahamid/Irene/yeast/catalogues/4b_ribo_catalogue.h5"
path_to_ribo_reference = "/struct/mahamid/Irene/yeast/templates/yeast_ribo_LSU_4b.hdf"

create_template_catalogue(output_path=ribo_catalogue_path,
                          reference_file=path_to_ribo_reference,
                          angles_file=angles_file, in_degrees=False)

fas_catalogue_path = "/struct/mahamid/Irene/yeast/catalogues/4b_fas_catalogue.h5"
path_to_fas_reference = "/struct/mahamid/Irene/yeast/templates/ED_FAS_iniavg_4b_b32_density_neg.em"
create_template_catalogue(output_path=fas_catalogue_path,
                          reference_file=path_to_fas_reference,
                          angles_file=angles_file, in_degrees=False)
