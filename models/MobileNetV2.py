import torchvision
import torch
from torch import nn


class MobileNetV2(nn.Module):
    def __init__(self, which_features="MobileNet_layer1"):
        super(MobileNetV2, self).__init__()
        self.which_features = which_features

        pretrained_model = torchvision.models.mobilenet_v2(pretrained=True)

        self.layer1 = pretrained_model._modules['features'][0:2] #112x16
        self.layer2 = pretrained_model._modules['features'][2] #56x24
        self.layer3 = pretrained_model._modules['features'][3:5] #28x32
        self.layer4 = pretrained_model._modules['features'][5:8] #14x64
        self.layer5 = pretrained_model._modules['features'][8:17] #7x160
        self.layer6 = pretrained_model._modules['features'][17:19]  # 7x1280
        # clear memory
        del pretrained_model

    def forward(self, x):
        # normalize to imagenet var and mean
        mean = torch.tensor([0.485, 0.456, 0.406], device='cpu')
        var = torch.tensor([0.229, 0.224, 0.225], device='cpu')
        in_size = x.shape[0]
        x = torch.div(torch.sub(x, mean[None, :, None, None]), var[None, :, None, None])
        x = self.layer1(x)
        if self.which_features == "MobileNet_layer1":
            return x.view(in_size, -1)
        x = self.layer2(x)
        if self.which_features == "MobileNet_layer2":
            return x.view(in_size, -1)
        x = self.layer3(x)
        if self.which_features == "MobileNet_layer3":
            return x.view(in_size, -1)
        x = self.layer4(x)
        if self.which_features == "MobileNet_layer4":
            return x.view(in_size, -1)
        x = self.layer5(x)
        if self.which_features == "MobileNet_layer5":
            return x.view(in_size, -1)
        x = self.layer6(x)
        if self.which_features == "MobileNet_layer6":
            return x.view(in_size, -1)
        return x