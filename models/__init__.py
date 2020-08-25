
from .continuous import ContinuousRepresentation
from supervision import quaternion,losses

import torchvision
import torch

import sys
from typing import Union, Sequence

try:
    from kaolin.rep import TriangleMesh
except ImportError:
    __KAOLIN_LOADED__ = False
else:
    __KAOLIN_LOADED__ = True

class Model(torch.nn.Module):
    def __init__(self,
        model: torch.nn.Module,
        head: torch.nn.Module,
    ):
        super(Model,self).__init__()
        self.model = model
        self.head = head

    def forward(self, img: torch.Tensor) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        b , c , h , w = img.size()
        latent = self.model(img)
        return self.head(latent , b)


def get_model(name: str="vgg16", head_name: str="continuous",  vertices: torch.Tensor=None , faces : torch.Tensor = None, pretrained: bool=True, with_bn: bool=False):
    
    model, head = None, None
    
    if name == "vgg16":
        model = torchvision.models.vgg16(pretrained, 2048)
    
    elif name == "resnet34":
        model = torchvision.models.resnet34(pretrained)
        model.fc = torch.nn.Linear(512, 2048)
    
    else:
        print("Invalid model name {}".format(name))
    
    if head_name == "spherical":
        head = SphericalRegression(

        )
    elif head_name == "continuous":
        head = ContinuousRepresentation(vertices , faces)
    else:
        print("Invalid model name {}".format(name))
    
    if model is not None and head is not None:
        return Model(model, head)
    else:
        raise ValueError("model and/or head arguments are invalid (model:{}, head:{})".format(model, head))
    
def convert_relu_to_elu(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
                setattr(model, child_name, nn.ELU())
        else:
            convert_relu_to_elu(child)

