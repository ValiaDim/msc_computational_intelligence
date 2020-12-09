import torchvision
import torch
from torch import nn


class VGG19_batchnorm(nn.Module):
    def __init__(self, which_features="vgg_layer1"):
        super(VGG19_batchnorm, self).__init__()
        self.which_features = which_features
        pretrained_model = torchvision.models.vgg19_bn(pretrained=True, progress=True)

        self.layer1 = pretrained_model._modules['features'][0:6]
        self.layer2 = pretrained_model._modules['features'][6:13]
        self.layer3 = pretrained_model._modules['features'][13:26]
        self.layer4 = pretrained_model._modules['features'][26:39]
        self.layer5 = pretrained_model._modules['features'][39:52]
        # clear memory
        del pretrained_model

    def forward(self, x):
        # normalize to imagenet var and mean
        mean = torch.tensor([0.485, 0.456, 0.406], device='cpu')
        var = torch.tensor([0.229, 0.224, 0.225], device='cpu')
        in_size = x.shape[0]
        x = torch.div(torch.sub(x, mean[None, :, None, None]), var[None, :, None, None])
        x = self.layer1(x)
        if self.which_features == "vgg_layer1":
            return x.view(in_size, -1)
        x = self.layer2(x)
        if self.which_features == "vgg_layer2":
            return x.view(in_size, -1)
        x = self.layer3(x)
        if self.which_features == "vgg_layer3":
            return x.view(in_size, -1)
        x = self.layer4(x)
        if self.which_features == "vgg_layer4":
            return x.view(in_size, -1)
        x = self.layer5(x)
        if self.which_features == "vgg_layer5":
            return x.view(in_size, -1)
        return x