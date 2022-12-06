from torch import nn
import numpy as np
import torch
from torchvision.models import resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()
        base = resnet18(pretrained=pretrained)
        self.base = nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.drop = nn.Dropout()
        self.final = nn.Linear(in_features,num_classes)
    
    def forward(self,x):
        x = self.base(x)
        x = self.drop(x.view(-1,self.final.in_features))
        return self.final(x)