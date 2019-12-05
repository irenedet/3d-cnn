import torch.nn as nn

from networks.unet_new import UNet3D

path_to_model = "/home/papalotl/finetuning/best_checkpoint.pytorch"
confs = {'in_channels': 1,
         "out_channels": 1,
         "depth": 8,
         "initial_features": 2,
         "skip_connections": False,
         # "final_activation": nn.Sigmoid(),
         # "elu": False
         }

# unet, optimizer, epoch, loss = load_unet_model(path_to_model=path_to_model,
#                                                confs=confs,
#                                                mode="eval", net=UNet3D)
unet3d = UNet3D(in_channels=1, out_channels=1, depth=4, initial_features=2,
                final_activation=nn.Sigmoid(), skip_connections=False,
                elu=False)
print(unet3d)

# unet = UNet(in_channels=1, out_channels=1, depth=4, initial_features=2,
#             final_activation=nn.Sigmoid(), elu=False)
#
# print(unet)
