class DatasetTableHeader(object):
    def __init__(self, semantic_classes: list = None,
                 clean_motl: list = None, filtering_mask: str = "lamella_file",
                 partition_name='train_partition',
                 processing_tomo='eman2_filtered_tomo',
                 image_acquisition_parameter='vpp'):

        if clean_motl is None:
            self.clean_motl = ["path_to_motl_clean"]
        else:
            self.clean_motl = clean_motl

        self.processing_tomo = processing_tomo
        self.original_tomo = 'original_tomo'
        self.tomo_name = 'tomo_name'
        self.semantic_classes = semantic_classes
        self.sample_type = 'sample_type'
        self.binning = 'binning'
        self.image_acquisition_parameter = image_acquisition_parameter
        self.filtering_mask = filtering_mask
        self.species = 'species'
        self.partition_name = partition_name
        self.test_partition = 'test_partition'
        self.x_dim = 'x_dim'
        self.y_dim = 'y_dim'
        self.z_dim = 'z_dim'
        self.x_shift = 'x_shift'
        self.y_shift = 'y_shift'
        self.z_shift = 'z_shift'
        if semantic_classes is not None:
            self.masks_names = [name + "_mask" for name in
                                self.semantic_classes]
        return


class ModelsTableHeader(object):
    def __init__(self):
        self.batch_normalization = "BN"
        self.date = "date"
        self.depth = "depth"
        self.dropout = "dropout"
        self.initial_features = 'if'
        self.logging_path = 'log_path'
        self.model_name = "model_name"
        self.model_path = "model_path"
        self.epochs = "n_epochs"
        self.old_model = "old_model"
        self.output_classes = "output_classes"
        self.retrain = "retrain"
        self.semantic_classes = "segmentation_names"
        self.train_split = "train_split"
        self.training_set = "training_set"
        self.encoder_dropout = "encoder_dropout"
        self.decoder_dropout = "decoder_dropout"
        return
