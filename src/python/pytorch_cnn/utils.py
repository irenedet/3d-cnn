import numpy as np
import torch
import torch.optim as optim
from src.python.pytorch_cnn.classes.unet import UNet
from src.python.pytorch_cnn.io import get_device
from src.python.filereaders.h5 import read_training_data
from src.python.image.filters import preprocess_data
from src.python.datasets.actions import split_dataset


def load_unet_model(path_to_model: str, confs: dict, mode="eval"):
    model = UNet(**confs)
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = get_device()
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    if "eval" == mode:
        model.eval()
        return model, optimizer, epoch, loss
    elif "train" == mode:
        model.train()
        # optimizer = optim.Adam(model.parameters())
        # optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        return model, optimizer, epoch, loss
    else:
        print("The loading mode requested is not supported.")


def save_unet_model(path_to_model: str, epoch, net, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path_to_model)


def get_testing_and_training_sets_from_partition(training_data_path: str,
                                                 label_name: str,
                                                 split=0.8) -> tuple:
    print("The training data path is ", training_data_path)
    raw_data, labels = read_training_data(training_data_path,
                                          label_name=label_name)
    print("Initial unique labels", np.unique(labels))

    # Normalize data
    preprocessed_data = preprocess_data(raw_data)

    # add a channel dimension
    preprocessed_data = np.array(preprocessed_data)[:, None]
    labels = np.array(labels)[:, None]

    train_data, train_labels, val_data, val_labels, data_order = \
        split_dataset(preprocessed_data, labels, split)
    return train_data, train_labels, val_data, val_labels, data_order
