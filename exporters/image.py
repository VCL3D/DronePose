import colour
from PIL import Image
import numpy
import torch
import os
import cv2


def blend_prediction(image,mask,c,path,name,alpha = 0.4):
    #convert image to PIL
    device = mask.get_device()
    img_ = Image.fromarray(numpy.uint8(image.squeeze(0).permute(1,2,0).cpu() * 255))
    img_ = img_.convert('RGBA')
    #get desired colour to blend image
    color = list(map(colour.web2rgb, [c]))[0]
    mask[:,0,:,:] = torch.where(mask[:,0,:,:] != 0,torch.tensor(int(color[0] * 255)).to(device),torch.tensor(0).to(device))
    mask[:,1,:,:] = torch.where(mask[:,1,:,:] != 0,torch.tensor(int(color[1] * 255)).to(device),torch.tensor(0).to(device))
    mask[:,2,:,:] = torch.where(mask[:,2,:,:] != 0,torch.tensor(int(color[2] * 255)).to(device),torch.tensor(0).to(device))
    mask_ = Image.fromarray(numpy.uint8(mask.squeeze(0).permute(1,2,0).cpu()))
    mask_ = mask_.convert('RGBA')
    blended_img = Image.blend(img_, mask_, alpha=alpha)
    #save image
    blended_img.save(os.path.join(path,name))