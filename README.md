# DronePose: Photorealistic UAV-Assistant Dataset Synthesis for 3D Pose Estimation via a Smooth Silhouette Loss

[![Paper](http://img.shields.io/badge/DronePose-arxiv.2008.08823-critical.svg?style=plastic)](https://arxiv.org/pdf/2008.08823.pdf)
[![Conference](http://img.shields.io/badge/ECCV-2020-blue.svg?style=plastic)](https://eccv2020.eu/)
[![Workshop](http://img.shields.io/badge/R6D-2020-darkblue.svg?style=plastic)](http://cmp.felk.cvut.cz/sixd/workshop_2020/)
[![Project Page](http://img.shields.io/badge/Project-Page-blueviolet.svg?style=plastic)](https://vcl3d.github.io/DronePose/)

**TODO**:
- [x] Train scripts
- [x] Evaluation scripts
- [x] Pre-trained model
- [x] Smooth silhoutte loss code
- [x] Inference code


# Data
The exocentric data used to train our single shot pose estimation model, are available [here](https://vcl3d.github.io/UAVA/) and are part of a larger dataset that contains rendered color images, silhouette masks , depth , normal maps, and optical flow for each viewpoint (e.g. user and UAV).
NOTE: The data should follow the same organisation structure.

# Requirements
The code is based on PyTorch and has been tested with Python 3.6 and CUDA 10.1.
We recommend setting up a virtual environment (follow the `virtualenv` documentation) for installing PyTorch and the other necessary Python packages.

**Note**

For running the inference code, Kaolin(0.1.0) is need.

## Train scripts
You can train your models by running python train.py with the following arguments:

- `--root_path`: Specifies the root path of the data.
- `--trajectory_path`: Specifies the trajectory path. 
-  `--drone_list`: The drone model from which data will be used.
-  `--view_list`: The camera view (i.e. UAV or user) from which data will be loaded.
-  `--frame_list`: The frames (i.e. 0 or 1) that will be loaded.
-  `--types_list`: The different modalities (e.g. colour,depth,silhouette) that will be loaded from the dataset.
-  `--saved_models_path`: Path where models are saved.


## Pre-trained Models
Our PyTorch pre-trained models (corresponding to those reported in the paper) are available at our [releases](https://github.com/VCL3D/DronePose/releases) and contain these model variants:

* [__Direct__ @ epoch 20](https://github.com/VCL3D/DronePose/releases/download/DIRECT/Direct)
* [__I0.1__ @ epoch 20](https://github.com/VCL3D/DronePose/releases/download/I0.1/I0.1)
* [__S0.1__ @ epoch 20](https://github.com/VCL3D/DronePose/releases/download/S0.1/S0.1)
* [__Gauss0.1__ @ epoch 20](https://github.com/VCL3D/DronePose/releases/download/Gauss0.1/Gauss0.1)

## Inference
You can try any of the above models by using our infer.py script and by setting the below arguments:
- `--input_path`: Path to the root folder containing the images.
- `--output_path`: Path for saving the final result.
- `--weights`: Path to the trained weights file.

## In-the-wild (YouTube videos) Results
![](data/Outdoor_1.gif)
