import torch
import torch.optim as optim
from src.python.pytorch_cnn.classes.unet import UNet


def load_unet_model(path_to_model: str, confs: dict, mode="eval"):
    model = UNet(*confs)
    optimizer = optim.Adam(model.parameters())

    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    if "eval" == mode:
        model.eval()
        return model, optimizer, epoch, loss
    elif "train" == mode:
        model.train()
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
