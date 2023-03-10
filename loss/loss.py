from turtle import forward
import torch.nn as nn
import torch


class BalloonLoss(nn.Module):
    def __init__(self):
        super(BalloonLoss, self).__init__()

    def forward(self, output, target):
        lat_out = output[0]
        lon_out = output[1]
        alt_out = output[2]

        lat_tar = target[0]
        lon_tar = target[1]
        alt_tar = target[2]