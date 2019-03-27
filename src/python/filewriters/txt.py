def write_model_description(file_path: str,
                            training_data_path: str,
                            label_name: str,
                            split: int,
                            model_name_pkl: str,
                            conf: dict,
                            data_order: list):
    with open(file_path, 'w') as txt_file:
        txt_file.write("training_data_path = " + training_data_path)
        txt_file.write("\n")
        txt_file.write("label_name=" + label_name)
        txt_file.write("\n")
        txt_file.write("split = " + str(split))
        txt_file.write("\n")
        txt_file.write("model_name = " + model_name_pkl)
        txt_file.write("\n")
        txt_file.write("conf = " + str(conf))
        txt_file.write("\n")
        txt_file.write("data_order = ")
        txt_file.write(str(data_order))
        txt_file.write("\n")
    return
