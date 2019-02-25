""" Taken from the web:
https://discuss.pytorch.org/t/
multi-class-cross-entropy-loss-function-implementation-in-pytorch/19077
modified to fit 3d image classification
"""
import torch


def multi_class_cross_entropy_loss_torch(predictions, labels):
    """
    Calculate multi-class cross entropy loss for every pixel in an image,
    for every image in a batch.

    In the implementation,
    - the first sum is over all classes,
    - the second sum is over all rows of the slice,
    - the third sum is over all columns of the slice
    - the fourth sum is over all slices of the image
    - the last mean is over the batch of images.

    :param predictions: Output prediction of the neural network.
    :param labels: Correct labels.
    :return: Computed multi-class cross entropy loss.
    """

    loss = -torch.mean(
        torch.sum(
            torch.sum(
                torch.sum(
                    torch.sum(labels * torch.log(predictions), dim=1), dim=1),
                dim=1), dim=1))
    return loss
