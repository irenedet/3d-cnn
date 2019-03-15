import torch
import torch.nn as nn
import torch.nn.functional as F

from src.python.tensors.actions import crop_tensor


class BCELoss(nn.Module):
    def __init__(self, weight_function=None):
        # in torch, loss functions must inherit from `torch.nn.Module`
        # and call the super constructor to be compatible with the
        # automatic differentiation capabilities
        super().__init__()

        # the weighting function is optional and will not
        # be used if it is `None`
        # if a weighting function is given, it must
        # take the target tensor as input and return
        # a weight tensor with the same shape
        self.weight_function = weight_function

    # to implement a loss function, we only need to
    # overload the forward pass.
    # the backward pass will be performed by torch automatically
    def forward(self, input, target):
        ishape = input.shape
        tshape = target.shape

        # make sure that the batches and channels target and input agree
        assert ishape[:2] == tshape[:2]
        assert ishape[1] == 1, "Only supports a single channel for now"

        # crop the target to fit the input
        target = crop_tensor(target, ishape)

        # check if we have a weighting function and if so apply it
        if self.weight_function is not None:
            # apply the weight function
            weight = self.weight_function(target)
            # compute the loss WITHOUT reduction, which means that
            # the los will have the same shape as input and target
            loss = F.binary_cross_entropy(input, target, reduction='none')

            # multiply the loss by the weight and
            # reduce it via element-wise mean
            assert weight.shape == loss.shape, "Loss and weight must have the same shape"
            loss = torch.mean(loss * weight)

        # if we don't have a weighting function, just apply the loss
        else:
            loss = F.binary_cross_entropy(input, target)
        return loss


class Multi_class_BCELoss(nn.Module):
    def __init__(self, in_channels, weight_function=None):
        # in torch, loss functions must inherit from `torch.nn.Module`
        # and call the super constructor to be compatible with the
        # automatic differentiation capabilities
        super().__init__()
        self.in_channels = in_channels
        # the weighting function is optional and will not
        # be used if it is `None`
        # if a weighting function is given, it must
        # take the target tensor as input and return
        # a weight tensor with the same shape
        self.weight_function = weight_function

    # to implement a loss function, we only need to
    # overload the forward pass.
    # the backward pass will be performed by torch automatically
    def forward(self, input, target):
        ishape = input.shape
        tshape = target.shape

        # make sure that the batches and channels target and input agree
        assert ishape[:2] == tshape[:2]
        assert ishape[1] == self.in_channels, "check number of input channels"

        # crop the target to fit the input
        target = crop_tensor(target, ishape)

        # check if we have a weighting function and if so apply it
        if self.weight_function is not None:
            # apply the weight function
            weight = self.weight_function(target)
            # compute the loss WITHOUT reduction, which means that
            # the los will have the same shape as input and target
            loss = F.binary_cross_entropy(input, target, reduction='none')

            # multiply the loss by the weight and
            # reduce it via element-wise mean
            assert weight.shape == loss.shape, "Loss and weight must have the same shape"
            loss = torch.mean(loss * weight)

        # if we don't have a weighting function, just apply the loss
        else:
            loss = F.binary_cross_entropy(input, target)
        return loss


# sorensen dice coefficient implemented in torch
# the coefficient takes values in [0, 1], where 0 is
# the worst score, 1 is the best score
class DiceCoefficient(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b ca be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        denominator = (prediction * prediction).sum() + (target * target).sum()
        return 2 * intersection / denominator.clamp(min=self.eps)


class DiceCoefficientLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    # the dice coefficient of two sets represented as vectors a, b ca be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        prediction.float()
        target.float()
        intersection = (prediction * target).sum()
        denominator = (prediction * prediction).sum() + (target * target).sum()
        return 1 - (2 * intersection / denominator.clamp(min=self.eps))


# class NLLLoss(nn.Module):
#     def __init__(self, class_weights=None):
#         super().__init__()
#         self.loss = nn.NLLLoss(weight=class_weights)
#
#     def forward(self, prediction, target):
# #        prediction.float()
        # target.long()
        # assert prediction.shape[0] == target.shape[0]
        # print("len(target.shape)", len(target.shape))
        # print("len(prediction.shape)", len(prediction.shape))
        # assert len(target.shape) == len(prediction.shape) - 1
#        # target = crop_tensor(target, prediction.shape)
#        # target = target[:, None]
        # return self.loss(prediction, target.long())
