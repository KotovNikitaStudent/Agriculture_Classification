from abstract_model import AbstractModel

from torchvision.models import mobilenet_v2
import torch.nn as nn
import torch


class MobilenetV2(AbstractModel):
    def __init__(self, pretrained=True):
        self.__pratrained = pretrained
        self.__model = mobilenet_v2(pretrained=self.__pratrained)
    
    def get_model(self, n_out=10, weight=None):
        self.__model._modules['classifier'][-1] = nn.Linear(self.__model._modules['classifier'][-1].in_features, n_out, bias=True)
        
        if weight is not None:
            self.__model.load_state_dict(torch.load(weight))
            self.__model = nn.DataParallel(self.__model)
        
        return self.__model