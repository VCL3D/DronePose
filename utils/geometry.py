import numpy as np
import torch
import cv2
try:
    from kaolin.rep import TriangleMesh
except ImportError:
    #print("Cannot load Kaolin.")
    __KAOLIN_LOADED__ = False
else:
    __KAOLIN_LOADED__ = True

def loadobj(filename):
    
    mesh = TriangleMesh.from_obj(filename)

    vertices = mesh.vertices
    faces = mesh.faces.int()

    vertices = vertices.unsqueeze(0)

    return vertices,faces


def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32')

    assert points_3D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    _, R_exp, t = cv2.solvePnP(points_3D,
                              np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)),
                              cameraMatrix,
                              distCoeffs)                            

    R, _ = cv2.Rodrigues(R_exp)
    return R, t


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    #TODO: Fix the normalisation;
    #norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    #norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    norm_quat = quat
    #print("norm quat ", norm_quat)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def perspectiveprojectionnp(fovy, ratio=1.0, near=0.3, far=750.0):
    
    tanfov = np.tan(fovy / 2.0)

    return np.array([[1.0 / (ratio * tanfov)], [1.0 / tanfov], [-1]], dtype=np.float32)


def objectMatrix(Q,T):
    """Returns a bx4x4 matrix;
    Q:Quaternion in global space;
    T:Translation in global space;
    """
    if(type(Q).__module__ == np.__name__):
        Q_ = torch.from_numpy(Q)
        Q_ = Q_.view(-1,4)
        Q_ = torch.stack(
            [
                Q_[:,3],
                Q_[:,0],
                Q_[:,1],
                Q_[:,2],
            ]
        )
        Rot_ = quat2mat(Q_.transpose(0,1))
        Pw = torch.zeros((4,4),dtype=torch.float32)
        Pw[:3,:3] = Rot_
        Pw[:3,3] = torch.from_numpy(T.transpose())
        Pw[3,3] = 1
    else:
        #is Tensor
        Q_ = Q
        b , _ = T.size()
        Rot_ = quat2mat(Q_)
        Pw = torch.zeros((b,4,4),dtype=torch.float32)
        Pw[:,:3,:3] = Rot_
        Pw[:,:3,3] = T
        Pw[:,3,3] = 1
    return Pw

def getRelativeMatrix(R1,R2,T1,T2):
    """Returns the relative position matrix between 1 and 2"""
    #TODO: Retrieve relative matrix from quaternion multiplication directly
    P_3x3 = torch.matmul(R2.transpose(1,0),R1)
    P_3x1 = torch.matmul(R2.transpose(1,0),(T1-T2))

    P12 = torch.zeros((4,4),dtype=torch.float32)

    P12[:3,:3] = P_3x3
    P12[:3,3] = P_3x1
    P12[3,3] = 1
    return P12

def get_3D_corners(vertices):
    
    vertices = vertices.transpose()

    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])
    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    corners = np.concatenate((np.transpose(corners), np.ones((1,8)) ), axis=0)
    return corners

def compute_2d_bb(pts):
    pts = pts.cpu().numpy().transpose(1,0)
    min_x = np.min(pts[0,:])
    max_x = np.max(pts[0,:])
    min_y = np.min(pts[1,:])
    max_y = np.max(pts[1,:])
    w  = max_x - min_x
    h  = max_y - min_y
    cx = (max_x + min_x) / 2.0
    cy = (max_y + min_y) / 2.0
    new_box = [cx, cy, w, h]
    return w , h