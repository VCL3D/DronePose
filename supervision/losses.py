import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d
from supervision import quaternion
import numpy as np
from utils.geometry import perspectiveprojectionnp
from math import radians
from typing import Tuple


def compute_geodesic_distance_from_two_matrices(m1, m2):
    """Input : 
    m1 gt rotation matrix
    m2 predicted rotation matrix
    """
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device))*-1 )
    
    
    theta = torch.acos(cos)
    
    return theta

         
def silhouette_error(Sgt,Sest):
    """
    Calculates the error between the ground truth and the rednered shilhouettes.
    Sest (b x c x w x h)
    Returns : bx1
    """

    #Sest = (Sest / 255 ).type(torch.int32)
    #error  = torch.norm(Sgt * Sest, p  = 1 ) / torch.norm(Sgt + Sest - Sgt * Sest, p =1)
    error = torch.norm(Sgt - Sest , dim = 1).mean()

    return error

def IOU(mask1,mask2):
    #expects binary mask and only zeros and ones
    b , c , h , w = mask1.size()
    eps = 1e-7
    mul = (mask1 * mask2).reshape(b,-1).sum(1)
    add = (mask1 + mask2).reshape(b,-1).sum(1)

    iou = mul / (add - mul + eps)
    return iou

def GIoU(mask1,mask2):
    #mask size : b x 3 x height x width 
    I = intersection(mask1,mask2)
    U = union(mask1,mask2)
    b , c , h , w = mask1.size()
    C1 = smallestReact(mask1.cpu())
    C1 = torch.from_numpy(smallestReact(mask1.cpu()))
    C2 = torch.from_numpy(smallestReact(mask2.cpu()))
    C_ = np.zeros((b,c,h,w))
    for i in range(b):
        #cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()) )
        #cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda())*-1 )
        min_y = torch.min(C1[i,3],C2[i,3]).int()
        min_x = torch.min(C1[i,2],C2[i,2]).int()
        max_y = torch.max(C1[i,1],C2[i,1]).int()
        max_x = torch.max(C1[i,0],C2[i,0]).int()
        C_[i,:,min_y:max_y,min_x:max_x] = 1
    #react = np.zeros((240,320,3))
    C_ = torch.from_numpy(C_).to(mask1.device)
    C =  2* C_.view(b,-1).sum(1)

    eps = 1e-7
    iou_term = (I  / U + eps)
    giou_term =(C - U ) / (C + eps)
    for i in range(b):
        if(U[i] != 0):
            iou_term[i] = (I[i] / U[i])
        else:
            iou_term[i] = torch.tensor(0.0)
        if C[i] != 0:
            giou_term[i] =(C[i] - U[i]) / C[i]
        else:
            giou_term[i] = torch.tensor(0.0,requires_grad=True)
        if (giou_term[i] == 1):
            giou_term[i] = torch.tensor(-1.0,requires_grad=True)
    return iou_term - giou_term


def union(mask1 , mask2):
    b , _ , _ , _ = mask1.size()
    I = intersection(mask1,mask2)
    U = (mask1 + mask2).reshape(b,-1).sum(1) - 2 * I
    return U.clone().detach().requires_grad_(True)

def intersection(mask1 , mask2):
    b , _ , _ , _ = mask1.size()
    mul = (mask1 * mask2).reshape(b,-1).sum(1)
    return  mul.clone().detach().requires_grad_(True)

def smallestReact(im):
    """Calculates the smallest reactangle that cotains both masks"""
    b , c , h , w = im.size()
    m = np.count_nonzero(im == 1 , axis = 1)
    C = np.zeros((b,4))
    for i in range(b):
        if m.sum(axis = 1)[i].sum() == 0:
            #there is no mask predicted
            max_x = w
            min_x = 0
            max_y = h
            min_y = 0
        else:

            max_x = np.nonzero(m.sum(axis = 1)[i])[0].max()
            max_y = np.nonzero(m.sum(axis = 2)[i])[0].max()
            min_x = np.nonzero(m.sum(axis = 1)[i])[0].min()
            min_y = np.nonzero(m.sum(axis = 2)[i])[0].min()
        
        C[i,0] = max_x
       
        C[i,1] = max_y
       
        C[i,2] = min_x
        
        C[i,3] = min_y
    return C

def compute_geodesic_loss(gt_r_matrix, out_r_matrix):
    theta = compute_geodesic_distance_from_two_matrices(gt_r_matrix, out_r_matrix)
    error = theta.mean()
    return error

def compute_shilhouette_loss(Sgt,Sest):
    #NOTE: test this
    Serror = silhouette_error(Sgt,Sest)
    #error = Serror.mean()
    return Serror

def giou(Sgt,Sest):
    giou = GIoU(Sgt,Sest)
    return 1 - giou.mean()

def iou(Sgt,Sest):
    iou_ = IOU(Sgt,Sest)
    return 1 - torch.mean(iou_)

def render_mask(renderer, vertices,faces,PdroneToCamera,b):
    """Set up renderer for visualising drone based on the predicted pose
        PdroneToCamera: relative pose predicted by network or GT
    """
    vertices = vertices.expand(b,vertices.size(1),vertices.size(2))

    device = vertices.get_device()
    
    #setup color
    vert_min = torch.min(vertices, dim=1, keepdims=True)[0]
    vert_max = torch.max(vertices, dim=1, keepdims=True)[0]

    colors = torch.zeros(vertices.size()).to(device)

    colors[:,:,:3] = 1
    
    #get homogeneous coordinates for vertices
    vertices = torch.torch.nn.functional.pad(vertices[:,:,],[0,1],"constant",1.0)
    vertices = vertices.transpose(2,1)

    vertices = torch.matmul(PdroneToCamera,vertices)

    vertices = vertices.transpose(2,1)

    b , _ ,_ = PdroneToCamera.size()

    #set camera parameters
    cameras = []
    camera_rot_bx3x3 = torch.zeros((b,3,3),dtype=torch.float32).to(device)
    camera_rot_bx3x3[:,0,0] = 1
    camera_rot_bx3x3[:,1,1] = 1
    camera_rot_bx3x3[:,2,2] = 1
    
    cameras.append(camera_rot_bx3x3)
    camera_pos_bx3 = torch.zeros((b,3),dtype=torch.float32).to(device)
    cameras.append(camera_pos_bx3)

    camera_proj_3x1 = torch.zeros((3,1),dtype=torch.float32).to(device)
    camera_proj_3x1[:,:] = torch.from_numpy(perspectiveprojectionnp(radians(50.8), ratio= 4./3., near=0.3, far=750.0))
    cameras.append(camera_proj_3x1)

    renderer.set_camera_parameters(cameras)

    #convert points from homogeneous
    z_vec =  vertices[..., -1:]
    scale = torch.tensor(1.) / torch.clamp(z_vec, 0.000000001)

    vertices =  vertices[..., :-1]

    #forward pass
    predictions , mask , _ = renderer(points=[vertices,faces.long()],colors_bxpx3=colors)

    return predictions #, mask

def render_silhouette(renderer, vertices,faces,PdroneToCamera,b):
    """Set up renderer for visualising drone based on the predicted pose
        PdroneToCamera: relative pose predicted by network or GT
    """
    vertices = vertices.expand(b,vertices.size(1),vertices.size(2))

    device = vertices.get_device()
    
    #setup color
    vert_min = torch.min(vertices, dim=1, keepdims=True)[0]
    vert_max = torch.max(vertices, dim=1, keepdims=True)[0]

    colors = torch.zeros(vertices.size()).to(device)

    colors[:,:,:3] = 1
    
    #get homogeneous coordinates for vertices
    vertices = torch.torch.nn.functional.pad(vertices[:,:,],[0,1],"constant",1.0)
    vertices = vertices.transpose(2,1)

    vertices = torch.matmul(PdroneToCamera,vertices)

    vertices = vertices.transpose(2,1)

    b , _ ,_ = PdroneToCamera.size()

    #set camera parameters
    cameras = []
    camera_rot_bx3x3 = torch.zeros((b,3,3),dtype=torch.float32).to(device)
    camera_rot_bx3x3[:,0,0] = 1
    camera_rot_bx3x3[:,1,1] = 1
    camera_rot_bx3x3[:,2,2] = 1
    
    cameras.append(camera_rot_bx3x3)
    camera_pos_bx3 = torch.zeros((b,3),dtype=torch.float32).to(device)
    cameras.append(camera_pos_bx3)

    camera_proj_3x1 = torch.zeros((3,1),dtype=torch.float32).to(device)
    camera_proj_3x1[:,:] = torch.from_numpy(perspectiveprojectionnp(radians(50.8), ratio= 4./3., near=0.3, far=750.0))
    cameras.append(camera_proj_3x1)

    renderer.set_camera_parameters(cameras)

    #convert points from homogeneous
    z_vec =  vertices[..., -1:]
    scale = torch.tensor(1.) / torch.clamp(z_vec, 0.000000001)

    vertices =  vertices[..., :-1]

    #forward pass
    predictions , mask , _ = renderer(points=[vertices,faces.long()],colors_bxpx3=colors)

    return mask #, mask

def smooth_silhouete_loss(s1, s2, kernel_size=49):    
    pool2d = torch.nn.AvgPool2d(kernel_size, 1, kernel_size // 2, count_include_pad=False)
    ss1 = pool2d(s1)
    iss1 = pool2d(1.0 - s1)
    ss2 = pool2d(s2)
    iss2 = pool2d(1.0 - s2)
    loss = s1 * (iss2 - ss2) + s2 * (iss1 - ss1)
    return loss.mean()

def gaussian_silhouete_loss(s1, s2, kernel_size=49):    
    
    ss1 = gaussian_blur(s1, kernel_size = (kernel_size,kernel_size) ,sigma = (1.5,1.5))
    
    iss1 = gaussian_blur(1.0 - s1,kernel_size = (kernel_size,kernel_size),sigma = (1.5,1.5))
    
    ss2 = gaussian_blur(s2, kernel_size = (kernel_size,kernel_size),sigma = (1.5,1.5))
    
    iss2 = gaussian_blur(1.0 - s2,kernel_size = (kernel_size,kernel_size),sigma = (1.5,1.5))
    
    loss = s1 * (iss2 - ss2) + s2 * (iss1 - ss1)
    
    return loss.mean()


def to_dq(pose):
    b, _ = pose.size()
    r = pose[:, :4]
    t = torch.cat([torch.zeros(b, 1), pose[:, 4:]], dim=1)
    return torch.cat([r, 0.5 * qmul(t, r)], dim=1)

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

def qconj(q):
    c = q * -1.0
    c[:, 0] *= -1.0
    return c

def qangle(q, eps=1e-12):
    norm = torch.norm(q[:, 1:])
    return 2.0 * torch.atan2(norm, q[:, 0] + eps)

def dq_rot(dq):
    return dq[:, :4]

def dq_trans(dq):
    return dq[:, 4:]

def dqconj(dq):
    r = dq[:, :4]
    t = dq[:, 4:]
    return torch.cat([qconj(r), qconj(t)], dim=1)

def dot(lhs, rhs):
    return torch.sum(lhs * rhs, dim=1)

def dqmul(lhs, rhs):
    r = qmul(dq_rot(lhs), dq_rot(rhs))
    t = qmul(dq_rot(lhs), dq_trans(rhs)) + qmul(dq_trans(lhs), dq_rot(rhs))
    return torch.cat([r, t], dim=1)

def dq_identity(like):
    b, _ = like.size()
    return torch.cat([
        torch.ones(b, 1),
        torch.zeros(b, 7)
    ], dim=1)

def to_dq(rot, trans):
    b  = rot.shape[0]
    device = rot.get_device()
    t = torch.cat([torch.zeros(b, 1).to(device), trans], dim=1)
    return torch.cat([rot, 0.5 * qmul(t, rot)], dim=1)

def dq_loss(pred_dq,gt_dq):
    device = pred_dq.get_device()
    diff_dq = dqmul(pred_dq, dqconj(gt_dq))
    loss = torch.nn.L1Loss()(diff_dq, dq_identity(diff_dq).to(device))
    return loss



# def dq_loss(pred_r, pred_t, gt_r, gt_t):
#     pred_dq = to_dq(pred_r, pred_t)
#     gt_dq = to_dq(gt_r, gt_t)
#     device = pred_dq.get_device()
#     diff_dq = dqmul(pred_dq, dqconj(gt_dq))
#     loss = torch.nn.L1Loss()(diff_dq, dq_identity(diff_dq).to(device))
#     return loss


###### Kornia implementation of gaussian filter ######

def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2)**2 / float(2 * sigma**2)
    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.

    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples::

        >>> kornia.image.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> kornia.image.get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("kernel_size must be an odd positive integer. "
                        "Got {}".format(kernel_size))
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d



def get_gaussian_kernel2d(kernel_size: Tuple[int, int],
                          sigma: Tuple[float, float]) -> torch.Tensor:
    """Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.

    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError("kernel_size must be a tuple of length two. Got {}"
                        .format(kernel_size))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d




class GaussianBlur(nn.Module):
    r"""Creates an operator that blurs a tensor using a Gaussian filter.

    The operator smooths the given tensor with a gaussian kernel by convolving
    it to each channel. It suports batched operation.

    Arguments:
        kernel_size (Tuple[int, int]): the size of the kernel.
        sigma (Tuple[float, float]): the standard deviation of the kernel.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples::

        >>> input = torch.rand(2, 4, 5, 5)
        >>> gauss = kornia.filters.GaussianBlur((3, 3), (1.5, 1.5))
        >>> output = gauss(input)  # 2x4x5x5
    """

    def __init__(self, kernel_size: Tuple[int, int],
                 sigma: Tuple[float, float]) -> None:
        super(GaussianBlur, self).__init__()
        self.kernel_size: Tuple[int, int] = kernel_size
        self.sigma: Tuple[float, float] = sigma
        self._padding: Tuple[int, int] = self.compute_zero_padding(kernel_size)
        self.kernel: torch.Tensor = get_gaussian_kernel2d(kernel_size, sigma)

    @staticmethod
    def compute_zero_padding(kernel_size: Tuple[int, int]) -> Tuple[int, int]:
        """Computes zero padding tuple."""
        computed = [(k - 1) // 2 for k in kernel_size]
        return computed[0], computed[1]

    def forward(self, x: torch.Tensor):  # type: ignore
        if not torch.is_tensor(x):
            raise TypeError("Input x type is not a torch.Tensor. Got {}"
                            .format(type(x)))
        if not len(x.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(x.shape))
        # prepare kernel
        b, c, h, w = x.shape
        tmp_kernel: torch.Tensor = self.kernel.to(x.device).to(x.dtype)
        kernel: torch.Tensor = tmp_kernel.repeat(c, 1, 1, 1)

        # TODO: explore solution when using jit.trace since it raises a warning
        # because the shape is converted to a tensor instead to a int.
        # convolve tensor with gaussian kernel
        return conv2d(x, kernel, padding=self._padding, stride=1, groups=c)


######################
# functional interface
######################
def gaussian_blur(input: torch.Tensor,
                  kernel_size: Tuple[int,
                                     int],
                  sigma: Tuple[float,
                               float]) -> torch.Tensor:
    r"""Function that blurs a tensor using a Gaussian filter.

    See :class:`~kornia.filters.GaussianBlur` for details.
    """
    return GaussianBlur(kernel_size, sigma)(input)
