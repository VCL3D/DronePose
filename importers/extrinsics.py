import numpy as np
import torch
import pandas as pd
import os
from utils import geometry
from supervision.quaternion import rotation_matrix_to_quaternion


def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def load_extrinsics(filename, data_type=torch.float32):
    data = np.loadtxt(filename)
    pose = np.zeros([4, 4])
    pose[3, 3] = 1
    pose[:3, :3] = data[:3, :3]
    pose[:3, 3] = data[3, :]
    extrinsics = torch.tensor(pose, dtype=data_type)
    return extrinsics , extrinsics.inverse()

def load_pose(key,trajectories_dir,frame):
    building , row , date = key.split("_")
    i = int(row)
    csv_file = os.listdir(os.path.join(trajectories_dir,building,date))
    for file in csv_file:
        if "blender" in file:
            csv_name = file
    
    csv = pd.read_csv(os.path.join(trajectories_dir,building,date,csv_name),sep='\t',decimal='.')

    if frame == 0:
        Rd = np.array([csv['Orientation(x)_t1'][i],csv['Orientation(y)_t1'][i],csv['Orientation(z)_t1'][i],csv['Orientation(w)_t1'][i]])
        Td = np.array([csv['Position(x)_t1'][i],csv['Position(y)_t1'][i],csv['Position(z)_t1'][i]])
    else:
        Rd = np.array([csv['Orientation(x)_t2'][i],csv['Orientation(y)_t2'][i],csv['Orientation(z)_t2'][i],csv['Orientation(w)_t2'][i]])
        Td = np.array([csv['Position(x)_t2'][i],csv['Position(y)_t2'][i],csv['Position(z)_t2'][i]])
    #read camera's pose
    Rc = np.array([csv['PilotRot(x)'][i],csv['PilotRot(y)'][i],csv['PilotRot(z)'][i],csv['PilotRot(w)'][i]])
    Tc = np.array([ csv['Pilot(x)'][i],csv['Pilot(y)'][i],csv['Pilot(z)'][i]])

    #get drone matrix in world space
    Pdw = geometry.objectMatrix(Rd,Td)
    #get camera matrix in world space
    Pcw = geometry.objectMatrix(Rc,Tc)
    
    PdroneToCamera = geometry.getRelativeMatrix(Pdw[:3,:3],Pcw[:3,:3],Pdw[:3,3],Pcw[:3,3])

    
    return PdroneToCamera , PdroneToCamera.inverse()

def load_egoPose(key , trajectories_dir ,target_,source_):
    building , row , date = key.split("_")
    i = int(row)
    csv_file = os.listdir(os.path.join(trajectories_dir,building,date))
    for file in csv_file:
        if "blender" in file:
            csv_name = file
    
    csv = pd.read_csv(os.path.join(trajectories_dir,building,date,csv_name),sep='\t',decimal='.')
     #read source pose
    if target_ == 1:
        Rsource = np.array([csv['Orientation(x)_t1'][i],csv['Orientation(y)_t1'][i],csv['Orientation(z)_t1'][i],csv['Orientation(w)_t1'][i]])
        Tsource = np.array([csv['Position(x)_t1'][i],csv['Position(y)_t1'][i],csv['Position(z)_t1'][i]])
        #read target pose
        Rtarget = np.array([csv['Orientation(x)_t2'][i],csv['Orientation(y)_t2'][i],csv['Orientation(z)_t2'][i],csv['Orientation(w)_t2'][i]])
        Ttarget = np.array([csv['Position(x)_t2'][i],csv['Position(y)_t2'][i],csv['Position(z)_t2'][i]])
    else:
        Rtarget = np.array([csv['Orientation(x)_t1'][i],csv['Orientation(y)_t1'][i],csv['Orientation(z)_t1'][i],csv['Orientation(w)_t1'][i]])
        Ttarget = np.array([csv['Position(x)_t1'][i],csv['Position(y)_t1'][i],csv['Position(z)_t1'][i]])
        #read source pose
        Tsource = np.array([csv['Position(x)_t2'][i],csv['Position(y)_t2'][i],csv['Position(z)_t2'][i]])
        Rsource = np.array([csv['Orientation(x)_t2'][i],csv['Orientation(y)_t2'][i],csv['Orientation(z)_t2'][i],csv['Orientation(w)_t2'][i]])

    #get source pose matrix in world space
    source_extrinsics = geometry.objectMatrix(Rsource,Tsource)
    #get target pose matrix in world space
    target_extrinsics = geometry.objectMatrix(Rtarget,Ttarget)

    source_to_target_pose = target_extrinsics.inverse() @ source_extrinsics

    return source_to_target_pose