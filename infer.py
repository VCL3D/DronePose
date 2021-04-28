import argparse
import os
import sys
import torch
import utils
import importers
import exporters
import models
from supervision.losses import render_silhouette,render_mask
from kaolin.graphics import DIBRenderer as Renderer
import glob
import cv2

def parse_arguments(args):
    usage_text = (
        "DronePose Inference."
        "Usage:  python infer.py [options],"
        "   with [options]: (as described below)"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    # paths
    parser.add_argument("--input_path", type = str, help = "Path to the root folder containing all the images")
    parser.add_argument("--output_path", type = str, help = "Path for saving the blended images")                  
    # model
    parser.add_argument('--model', type=str, default="resnet34", help='Model name')
    parser.add_argument('--head', type=str, default="continuous", help='Head name')
    parser.add_argument('--weights', type=str, help='Path to the trained weights file.')    
    parser.add_argument("--exocentric_w", type=float, default=0.1, help = " Exocentric silhouette supervision loss regulariser.")
    parser.add_argument('--colour',type=str,default='red',help = "Colour to be used for the final blended image")
    # hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')  
    return parser.parse_known_args(args)


if __name__ == "__main__":
    #parse arguments
    args, unknown = parse_arguments(sys.argv)
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    #set up device
    device = torch.device("cuda:{}" .format(gpus[0])\
        if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0\
        else "cpu")
    # load 3d model
    vertices , faces = utils.geometry.loadobj("./data/DJI.obj")
    vertices = vertices.to(device)
    faces = faces.to(device)
    #set up renderer (width , height)
    width, height = 320, 240
    renderer = Renderer(width,height)
    if args.exocentric_w > 0.0:
        model = models.get_model(args.model, args.head , vertices , faces )
    else:
        model = models.get_model(args.model, args.head)
    model = model.to(device)
    if (len(gpus) > 0): #workaround beacuse the models were saved with nn.Parallel        
        model = torch.nn.parallel.DataParallel(model, gpus)
    utils.init.load_weights(model,args.weights) 
    model.eval()
    #get all images
    if not os.path.exists(args.input_path):
        print("Input image path does not exist (%s)." % args.input_path)
        exit(-1)
    images = sorted(glob.glob(args.input_path + "/*.png"),key=os.path.getmtime)
    with torch.no_grad():
        for img_ in images:
            img = importers.load_image(img_)
            pred_rot_matrix , pred_translation = model(img)
            Pdw = torch.zeros((1,4,4)).to(device)
            Pdw[:,:3,:3] = pred_rot_matrix.to(device)
            Pdw[:,:3,3] = pred_translation.to(device)
            Pdw[:,3,3] = 1
            #calculate silhouette
            mask = render_mask(renderer,vertices,faces,Pdw,1)
            #get image name
            name = img_.split("/")[-1]
            #save image
            exporters.image.blend_prediction(img,mask,args.colour,args.output_path,name)
    print("Inference script has finished.")
