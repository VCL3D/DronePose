import os
import sys
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import importers
import warnings
from PIL import Image
import utils

class DataLoaderParams:
    def __init__(self,
        root_path,
        trajectories_dir,
        data_split,
        drone_list,
        view_list,
        frame_list,
        types_list,
        transform):
        
        self.root_path = root_path
        self.trajectories_dir = trajectories_dir
        self.data_split = data_split
        self.drone_list = drone_list
        self.view_list = view_list
        self.frame_list = frame_list
        self.types_list = types_list
        self.transform = transform

class DataLoad(Dataset):
    def __init__(self,params):
        super(DataLoad,self).__init__()
        self.params = params
        root_path = self.params.root_path
        trajectories_dir = self.params.trajectories_dir
        data_split  = self.params.data_split
        
        self.data = {}

        buildings = self.get_splits(data_split)

        for drone in os.listdir(root_path):
            #ignore not included drone models
            if drone not in self.params.drone_list:
                continue
            root_path_ = os.path.join(root_path,drone)
            for view in os.listdir(root_path_):
                if view not in self.params.view_list:
                    continue
                view_path = os.path.join(root_path_,view)
                #get included types
                for type_ in os.listdir(view_path):
                    if type_ not in self.params.types_list:
                            continue
                    #get images for each building
                    for building in buildings:
                        img_path = os.path.join(view_path,type_,building)
                        for img in os.listdir(img_path):
                            if (len(img.split("_")) <= 6):
                                row, date , frame , __ , view , _ = img.split("_")
                            else:
                                row, date , frame , __ ,_, view , _ = img.split("_")
                            #get only included view
                            if view not in self.params.view_list:
                                continue
                            #ignore not listed frame
                            if int(frame) not in self.params.frame_list:
                                continue
                            #NOTE:set the path to the extrinsics file
                            full_img_name = os.path.join(img_path,img)
                            #set a unique name for each row
                            unique_name = building + "_" + str(row) + "_" + str(date)
                            if unique_name not in self.data:
                                self.data[unique_name] = {}

                            if view not in self.data[unique_name]:
                                self.data[unique_name][view] = {}
                            
                            if frame not in self.data[unique_name][view]:
                                self.data[unique_name][view][frame] = {}
                            
                            self.data[unique_name][view][frame][type_] = full_img_name
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        #get an entry
        key = list(self.data.keys())[idx]
        datum = self.data[key]
        datum_out = {}
        for view in self.params.view_list:
            datum_view = {}
            for frame in self.params.frame_list:
                frame_dict = {}
                for type_ in self.params.types_list:
                    if type_ == "colour":
                        value = importers.image.load_image(datum[view][str(frame)][type_])
                        #apply transform
                        if self.params.transform is not None:
                            value = self.params.transform(value)
                    elif type_ == "silhouette" and view == "exocentric":
                        value = importers.image.load_image(datum[view][str(frame)][type_])
                    elif type_ == "colour" or type_ == "silhouette":
                          continue
                    elif type_ == "depth":
                        value = importers.image.load_depth(datum[view][str(frame)][type_])
                    else:
                        value = importers.image.load_image(datum[view][str(frame)][type_])
                    frame_dict.update({str(type_) : value.squeeze(0)})
                    #load pose for each frame
                    if view == "exocentric":
                        pose , pose_inv  = importers.load_pose(key,self.params.trajectories_dir,frame)
                        frame_dict.update({"pose":pose,"pose_inv" : pose_inv, 'key':key})
                    else:
                        #NOTE:get source to target pose (t0,t1)
                        if int(frame) == 1:
                            #set source to 0
                            source_ = 0
                        else:
                            #source is 1
                            source_ = 1
                        ego_pose = importers.load_egoPose(key,self.params.trajectories_dir,source_,int(frame))
                        frame_dict.update({'key':key , 'source_to_target' : ego_pose})
                datum_view.update({
                    frame:frame_dict
                })
            datum_out.update({
                view:datum_view
            })
        return datum_out
    
    def get_data(self):
        return self.data

    def get_splits(self,split):
        
        if split == "train":
            txt_path = os.path.join(os.getcwd(),"data splits","scenes_train.txt")
        
        elif split == "test":
            txt_path = os.path.join(os.getcwd(),"data splits","scenes_test.txt")
        
        elif split == "val":
            txt_path = os.path.join(os.getcwd(),"data splits","scenes_val.txt")
        
        with open(txt_path,'r') as txt_file:
            splits = [line.strip() for line in txt_file]
        
        return splits
