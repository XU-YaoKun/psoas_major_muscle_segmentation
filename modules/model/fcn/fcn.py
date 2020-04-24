import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class FCN(nn.Module):
    def __init__(self, n_channels, n_class):
        super(FCN, self).__init__()

        # conv1
        self.conv1_1 = ConvBlock(n_channels, 64)
        self.conv1_2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv2
        self.conv2_1 = ConvBlock(64, 128)
        self.conv2_2 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv3
        self.conv3_1 = ConvBlock(128, 256)
        self.conv3_2 = ConvBlock(256, 256)
        self.conv3_3 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # conv4
        self.conv4_1 = ConvBlock(256, 512)
        self.conv4_2 = ConvBlock(512, 512)
        self.conv4_3 = ConvBlock(512, 512)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
 
        # fcn6
        self.fc6 = nn.Conv2d(512, 1024, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fcn7
        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(1024, n_class, 1)
        self.score_pool3 = nn.Conv2d(128, n_class, 1)
        self.score_pool4 = nn.Conv2d(256, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 2, stride=2, bias=False
        )
        self.upscore4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=4, bias=False
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 2, stride=2, bias=False
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        h = x
        h = self.conv1_1(h)
        h = self.conv1_2(h)
        h = self.pool1(h)

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = self.pool2(h)
        pool2 = h

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)
        h = self.pool4(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore2(h)
        upscore2 = h  # 1/8

        h = self.score_pool4(pool3)
        score_pool4c = h  # 1/8
        h = upscore2 + score_pool4c  # 1/8

        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/4

        h = self.score_pool3(pool2)
        score_pool3c = h  # 1/4
        h = upscore_pool4 + score_pool3c  # 1/4

        h = self.upscore4(h)

        return h
