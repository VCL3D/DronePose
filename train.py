import argparse
import os
import sys
import time
import torch
import torchvision
import utils
from utils import metrics
import dataset
import models
from supervision import *
import numpy as np
from kaolin.graphics import DIBRenderer as Renderer
import tqdm

def parse_arguments(args):
    usage_text = (
        "ExocentricEgocentricDronePose3D."
        "Usage:  python train.py [options],"
        "   with [options]: (as described below)"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    # durations
    parser.add_argument('-e',"--epochs", default=20 , type = int, help = "Train for a total number of <epochs> epochs.")
    parser.add_argument('-b',"--batch_size", default=8, type = int, help = "Train with a <batch_size> number of samples each train iteration.")
    parser.add_argument("--test_batch_size", default= 8, type = int, help = "Test with a <batch_size> number of samples each test iteration.")    
    parser.add_argument('-d','--disp_iters', type=int, default=50, help='Log training progress (i.e. loss etc.) on console every <disp_iters> iterations.')
    parser.add_argument('-w','--workers', type=int, default=4, help='Test model every <test_iters> iterations.')
    # paths
    parser.add_argument("--root_path", type = str, help = "Path to the root folder containing all the files")
    parser.add_argument("--trajectory_path", type = str, help = "Path containing the ground_truth poses")
    parser.add_argument("--saved_models_path", type = str, help = "Path where models are saved")
    parser.add_argument("--load_model", type = str, help = "Path where models are saved")                
    #splits
    parser.add_argument("--data_splits" , type = str , help = "Flag for splitting the data")
    # model
    parser.add_argument('--model', type=str, default="resnet34", help='Model name')
    parser.add_argument('--head', type=str, default="continuous", help='Head name')
    parser.add_argument('--weight_init', type=str, default="none", help='Weight initialization method, or path to weights file (for fine-tuning or continuing training)')
    # optimization
    parser.add_argument('-o','--optimizer', type=str, default="adam", help='The optimizer that will be used during training.')
    parser.add_argument("--opt_state", type = str, help = "Path to stored optimizer state file (for continuing training)")
    parser.add_argument('-l','--lr', type=float, default=0.001, help='Optimization Learning Rate.')
    parser.add_argument('-m','--momentum', type=float, default=0.0, help='Optimization Momentum.')
    parser.add_argument('--momentum2', type=float, default=0.0, help='Optimization Second Momentum (optional, only used by some optimizers).')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Optimization Epsilon (optional, only used by some optimizers).')
    parser.add_argument('--weight_decay', type=float, default=0, help='Optimization Weight Decay.')    
    # hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    # other
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')
    parser.add_argument("--view_list",nargs="*", type=str, default = ["egocentric","exocentric"], help = "List of views to be loaded")
    parser.add_argument("--drone_list",nargs="*", type=str, default = ["M2ED"], help = "List of drone models to be loaded")
    parser.add_argument("--visdom", type=str, nargs='?', default="localhost", const="127.0.0.1", help = "Visdom server IP (port defaults to 8097)")
    parser.add_argument("--visdom_iters", type=int, default=400, help = "Iteration interval that results will be reported at the visdom server for visualization.")
    parser.add_argument("--seed", type=int, default=1337, help="Fixed manual seed, zero means no seeding.")
    parser.add_argument("--frame_list",nargs="*", type=int, default = [0,1], help = "List of frames to be included")
    parser.add_argument("--types_list",nargs="*", type=str, default = ["colour", "depth","silhouette"], help = "List of different modalities to be loaded")
    # network specific params
    parser.add_argument("--regression_w", type=float, default=0.9, help = "Pose regression loss weight.")
    parser.add_argument("--six_d_ratio", type=float, default=0.05, help = "Ratio between the position and the rotation loss.")
    parser.add_argument("--exocentric_w", type=float, default=0.1, help = " Exocentric silhouette supervision loss regulariser.")
    return parser.parse_known_args(args)


def train(args,train_set,test_set,sm_loss):
    """
    train script
    """
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    device, visualizer = utils.initialize(args)
    
    # load 3d model
    vertices , faces = utils.geometry.loadobj("./data/DJI.obj")
    vertices = vertices.to(device)
    faces = faces.to(device)
    #set up renderer (width , height)
    renderer = Renderer(320,240)
    if args.exocentric_w > 0.0:
        model = models.get_model(args.model, args.head , vertices , faces )
    else:
        model = models.get_model(args.model, args.head)
    
    model = model.to(device)
    if (len(gpus) > 1):        
        model = torch.nn.parallel.DataParallel(model, gpus)
    
    optimizer = torch.optim.Adam(model.parameters(),lr = args.lr)
    
    if args.load_model is not None:
        utils.init.initialize_weights(model, optimizer, args.load_model)
    
    #define losses
    score_position = torch.nn.MSELoss()
    criterion = torch.nn.BCEWithLogitsLoss()

    model.train()
    iteration_counter = 0
    
    #set up logger
    logger = utils.logger.Logger(os.path.join(args.saved_models_path, args.name +  '_log.txt'))
    logger.set_names([
                    'Epoch', "Learning Rate","Model" , "Head" , "Batch Size" , "iteration_counter",
                    "Six_d_ratio" , "Regression Weight" , "Exocentric Weight" ,
                    'Val Normalised Position Error', 'Val Angular Distance' , 'Val ESA Pose Error Total',
                    "2 cm 2 deg acc" , "5 cm 5 deg acc" , "10 cm 10 deg acc",
                    "ADD 0.02d" , "ADD 0.05d" , "ADD 0.10 d"
                    ])
    for epoch in range(args.epochs):
        print("Training | Epoch: {}".format(epoch))
        for batch_id, batch in enumerate(train_set):           
            b, c, h, w = batch['exocentric'][0]["colour"].size()
            model_input = batch['exocentric'][0]["colour"].to(device)
            if b < args.batch_size:
                continue
            # loss init
            active_loss = torch.tensor(0.0).to(device)
            mask_loss = 0.0
            optimizer.zero_grad()
            predicted_rot_matrix , translation = model(model_input)
            Pdw = torch.zeros((b,4,4)).to(device)
            Pdw[:,:3,:3] = predicted_rot_matrix
            Pdw[:,:3,3] = translation
            Pdw[:,3,3] = 1
            #calculate silhouette
            mask = render_silhouette(renderer,vertices,faces,Pdw,b)
            mask = mask.transpose(3,1)
            mask = mask.transpose(3,2)

            rotation_mat_gt = batch['exocentric'][0]["pose"][:,:3,:3].to(device)
            translation_gt = batch['exocentric'][0]["pose"][:,:3,3].to(device)

            if args.regression_w > 0.0:
                if args.head == "continuous":
                    rotation_loss = compute_geodesic_loss(rotation_mat_gt, predicted_rot_matrix)
                    position_loss = score_position(translation,translation_gt)
                    regression_loss = compute_geodesic_loss(rotation_mat_gt, predicted_rot_matrix) + score_position(translation,translation_gt)
                    active_loss = args.regression_w * ( args.six_d_ratio * compute_geodesic_loss(rotation_mat_gt, predicted_rot_matrix) + (1 - args.six_d_ratio) * score_position(translation,translation_gt)) 
            if args.exocentric_w > 0.0:
                #add exocentric supervision
                if sm_loss == "gauss":
                    exocentric_loss = gaussian_silhouete_loss(
                    batch['exocentric'][0]["silhouette"].to(device)[:, 0, :, :].unsqueeze(1),
                    mask , kernel_size=69
                )
                elif sm_loss == "box_filter":
                    exocentric_loss = smooth_silhouete_loss(
                    batch['exocentric'][0]["silhouette"].to(device)[:, 0, :, :].unsqueeze(1),
                    mask,kernel_size = 49
                )
                elif sm_loss == "giou":
                    exocentric_loss = giou(batch['exocentric'][0]["silhouette"].to(device),mask).to(device)
                
                else:
                    exocentric_loss = iou(batch['exocentric'][0]["silhouette"].to(device),mask).to(device)
                
                active_loss += args.exocentric_w * exocentric_loss
            
            active_loss.backward()
            optimizer.step()
            iteration_counter += b
            if (iteration_counter + 1) % args.disp_iters <= args.batch_size:
                #loss plots
                visualizer.append_loss(epoch + 1,iteration_counter,active_loss,"active loss")
                if args.exocentric_w > 0.0:
                    visualizer.append_loss(epoch + 1, iteration_counter, exocentric_loss, "exocentric loss")
                if args.regression_w > 0.0:
                    visualizer.append_loss(epoch + 1, iteration_counter, regression_loss, "regression loss")
                    visualizer.append_loss(epoch + 1, iteration_counter, rotation_loss, "rotation loss")
                    visualizer.append_loss(epoch + 1, iteration_counter, position_loss, "position loss")

            # %visualisation
            if args.visdom_iters > 0 and (iteration_counter + 1) % args.visdom_iters <= args.batch_size:
                visualizer.show_images(mask.clamp(min=0.0, max=1.0),'Mask predicted via train')
                if args.exocentric_w > 0.0:
                    visualizer.show_images(batch['exocentric'][0]["silhouette"],'Mask exocentric')
                    #test difference between gt and predicted
                    diff = torch.abs(mask - batch['exocentric'][0]["silhouette"].to(device))
                    visualizer.show_images(diff.clamp(min = 0.0 , max = 1.0),'Train Difference')
        
        #TODO: change logic and save the best model in terms of validation loss
        utils.checkpoint.save_network_state(model, optimizer,epoch,\
            args.name + "_model_state_epoch_" + str(epoch), args.saved_models_path)
        print("Checkpoint")
        #test model
        test(args,test_set,epoch,model,logger,device,vertices)
        model.train()        
    #close logger file when train has finished
    logger.close()

def test(args,test_set,epoch,model,logger,device,vertices):
    """
    Evaluation script and logger script
    """
    model.eval()
    counter = 0
    #store the total error of the test set
    total_score = []
    total_orientation = []
    total_position = []
    total_acc002 = []
    total_acc005 = []
    total_acc010 = []
    #add metric
    total_add002 = []
    total_add005 = []
    total_add010 = []
    #get lr
    lr = args.lr
    with torch.no_grad():
        for test_batch_id , test_batch in enumerate(test_set):
            b, c, h, w = test_batch['exocentric'][0]["colour"].size()

            pred_rot_matrix , pred_translation   = model(test_batch['exocentric'][0]["colour"].to(device))
                
            translation_gt = test_batch['exocentric'][0]["pose"][:,:3,3].to(device)    
            #transformation matrix for calculating metrics
            Pdw = torch.zeros((b,4,4))
            Pdw[:,:3,:3] = pred_rot_matrix
            Pdw[:,:3,3] = pred_translation
            Pdw[:,3,3] = 1
            Pdw = Pdw.to(device)

            #relative angle -- Metrics from ESA challenge
            rotation_mat_gt = test_batch['exocentric'][0]["pose"][:,:3,:3].to(device)
            position_score = metrics.calcNormalisedPositionDistance(translation_gt.cpu(),pred_translation.cpu())
            orientation_score = metrics.calcAngularDistance(rotation_mat_gt.cpu(),pred_rot_matrix.cpu())
            #append the mean error per batch size
            total_orientation.append(orientation_score.mean())
            total_position.append(position_score.mean())
            total_score.append((position_score + orientation_score).mean())
                
            #calculate n◦, n cm
            acc002 , acc005 , acc010 = metrics.evaluate_pose_add(rotation_mat_gt.cpu(),pred_rot_matrix.cpu(),translation_gt.cpu(),pred_translation.cpu())
            total_acc002.append(acc002)
            total_acc005.append(acc005)
            total_acc010.append(acc010)
                
            #calculate ADD metric
            add002 , add005 , add010 = metrics.add(vertices,test_batch['exocentric'][0]["pose"].to(device),Pdw)
            total_add002.append(add002)
            total_add005.append(add005)
            total_add010.append(add010)
                
            counter += b
        #append values to the logger
        # append logger file
        logger.append([
                epoch + 1,lr,args.model,args.head, args.batch_size, counter,
                args.six_d_ratio,args.regression_w,args.exocentric_w ,
                np.mean(total_position) ,np.mean(total_orientation),np.mean(total_score),
                np.mean(total_acc002), np.mean(total_acc005), np.mean(total_acc010),
                np.mean(total_add002), np.mean(total_add005), np.mean(total_add010)
                ])
        print("Testing | Epoch: {} , iteration {} , position_loss {} , orientation score {} , total add 10 error {}".format(epoch, counter,np.mean(total_position), np.mean(total_orientation),np.mean(total_add010)))


if __name__ == "__main__":
    #parse arguments
    args, unknown = parse_arguments(sys.argv)
    # training data loader
    train_data_params = dataset.dataloader.DataLoaderParams(\
        root_path=args.root_path, trajectories_dir = args.trajectory_path,data_split = "train", drone_list = args.drone_list,view_list=args.view_list,\
        frame_list = args.frame_list, types_list = args.types_list, transform = None) 
    train_data_iterator = dataset.dataloader.DataLoad(train_data_params)
    train_set = torch.utils.data.DataLoader(train_data_iterator,\
        batch_size = args.batch_size, shuffle=True,\
        num_workers = args.workers, pin_memory=False)
    # validation data loader
    test_data_params = dataset.dataloader.DataLoaderParams(\
        root_path=args.root_path, trajectories_dir = args.trajectory_path,data_split = "test", drone_list = args.drone_list, view_list=args.view_list,\
        frame_list = args.frame_list, types_list = args.types_list, transform = None
        ) 
    test_data_iterator = dataset.dataloader.DataLoad(test_data_params)
    test_set = torch.utils.data.DataLoader(test_data_iterator,\
        batch_size = args.test_batch_size, shuffle=True,\
        num_workers = args.workers, pin_memory=False)
    print("Train size : {0} | Test size : {1}".format(train_data_iterator.__len__(),test_data_iterator.__len__()))

    train(args,train_set,test_set,"gauss")
