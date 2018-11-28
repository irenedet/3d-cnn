import numpy as np


def normalize_training_data(data: np.array):
    data_mean = data.mean()
    data_std = data.std()
    print("The data mean value is", data_mean)
    print("The data std value is", data_std)

    data -= data_mean
    data /= data_std
    # check again to double check
    print("After normalization the data has mean value", data.mean())
    print("After normalization the data has standard deviation", data.std())
    return data
