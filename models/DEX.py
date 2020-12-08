import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def vgg_block(in_channels, out_channels, more=False):
    blocklist = [
        ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)),
        ('relu1', nn.ReLU(inplace=True)),
        ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
        ('relu2', nn.ReLU(inplace=True)),
    ]
    if more:
        blocklist.extend([
            ('conv3', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)),
            ('relu3', nn.ReLU(inplace=True)),
        ])
    blocklist.append(('maxpool', nn.MaxPool2d(kernel_size=2, stride=2)))
    block = nn.Sequential(OrderedDict(blocklist))
    return block


# VGG16 architecture
class VGG(nn.Module):
    def __init__(self, classes=1000, channels=3, which_features="conv"):
        if which_features not in ["conv", "fc1", "fc2", "cls"]:
            raise NotImplementedError
        self.which_features = which_features
        super().__init__()
        self.conv = nn.Sequential(
            vgg_block(channels, 64),
            vgg_block(64, 128),
            vgg_block(128, 256, True),
            vgg_block(256, 512, True),
            vgg_block(512, 512, True),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5, inplace=True),
        )
        self.cls = nn.Linear(4096, classes)

    def forward(self, x):
        in_size = x.shape[0]
        x = self.conv(x)
        x = x.view(in_size, -1)
        if self.which_features == "conv":
            return x
        x = self.fc1(x)
        if self.which_features == "fc1":
            return x.view(in_size, -1)
        x = self.fc2(x)
        if self.which_features == "fc2":
            return x.view(in_size, -1)
        x = self.cls(x)
        if self.which_features == "cls":
            return x.view(in_size, -1)
        x = F.softmax(x, dim=1)
        return x


class Age(VGG):
    def __init__(self, classes=101, channels=3, which_features="conv"):
        super().__init__(which_features=which_features)
        self.cls = nn.Linear(4096, classes)


if __name__ == '__main__':
    net = Age()
    print(net)

