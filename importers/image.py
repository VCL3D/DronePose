import cv2
import torch
import numpy

def load_image(filename, data_type=torch.float32, width = 320, height = 240 , convert=True):
    color_img = numpy.array(cv2.resize(cv2.imread(filename, cv2.IMREAD_ANYCOLOR),(width,height),interpolation=cv2.INTER_AREA))
    h, w, c = color_img.shape
    #convert image to rgb
    if convert:
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    color_data = color_img.astype(numpy.float32).transpose(2, 0, 1)
    return torch.from_numpy(
        color_data.reshape(1, c, h, w)        
    ).type(data_type) / 255.0


def load_depth(filename, data_type=torch.float32, scale=1.0):
    depth_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYDEPTH))
    h, w = depth_img.shape
    depth_data = depth_img.astype(numpy.float32) * scale
    return torch.from_numpy(
        depth_data.reshape(1, 1, h, w)        
    ).type(data_type)

def crop_depth(filename, data_type=torch.float32, scale=0.001):
    depth_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYDEPTH))
    center_cropped_depth_img = depth_img[60:420, 0:640]
    h, w = center_cropped_depth_img.shape
    depth_data = center_cropped_depth_img.astype(numpy.float32) * scale
    return torch.from_numpy(
        depth_data.reshape(1, 1, h, w)        
    ).type(data_type)