import torch
import torch.nn as nn
import torch.nn.functional as F
from supervision import quaternion , losses

try:
    from kaolin.graphics import DIBRenderer as Renderer
except ImportError:
    __KAOLIN_LOADED__ = False
else:
    __KAOLIN_LOADED__ = True

from math import radians
import numpy as np
from utils import geometry

class ContinuousRepresentation(nn.Module):
    def __init__(self , vertices = None , faces = None):
        #6D representation of rotation;
        super(ContinuousRepresentation,self).__init__()

        self.vertices = vertices
        self.faces = faces
        
        self.linear_1 = nn.Linear(2048,1024)
        self.linear_2 = nn.Linear(1024,512)
        self.linear_3 = nn.Linear(512,128)
        self.fc_position = nn.Linear(128,3,bias = True)
        self.fc_rotation = nn.Linear(128,6,bias=False)

    def forward(self, x ,b):

        
        x = self.linear_1(x)
        x = F.elu(x)
        x = self.linear_2(x)
        x = F.elu(x)
        x = self.linear_3(x)
        x = F.elu(x)
        
        position = self.fc_position(x)
        
        ortho6d = self.fc_rotation(x)
        
        
        out_rotation_matrix = quaternion.torch_ortho6d_to_rotation_matrix(ortho6d) #batch*3*3
        
        return out_rotation_matrix, position
