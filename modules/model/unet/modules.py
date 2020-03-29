import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channel, out_channel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up, self).__init__()

        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # padding
        diffh = x2.size(2) - x1.size(2)
        diffw = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [
                diffw // 2,
                diffw - diffw // 2,
                diffh // 2,
                diffh - diffh // 2,
            ],
        )

        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)

        return x


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 1)

    def forward(self, x):
        return self.conv(x)
