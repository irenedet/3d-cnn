import numpy as np


def split_dataset(data: np.array, labels: np.array, split: int) -> tuple:
    train_data, train_labels = data[:split], labels[:split]
    val_data, val_labels = data[split:], labels[split:]
    print("Shape of training data:", train_data.shape)
    print("Shape of validation data", val_data.shape)
    return train_data, train_labels, val_data, val_labels
