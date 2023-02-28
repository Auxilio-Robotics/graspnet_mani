#!/usr/bin/env python

""" Runs inference on the subscribed images and publishes on the grasp_pose topic
    Author: chenxi-wang, Abhinav Gupta
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import cv2

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path', required=False,default='/home/teamf/graspnet_ws/src/perception/grasp_detector/scripts/checkpoint-rs.tar', help='Model checkpoint path')
# parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
# parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
# parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
# parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
# cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    # checkpoint = torch.load(cfgs.checkpoint_path)
    checkpoint = torch.load('/home/teamf/graspnet_ws/src/perception/grasp_detector/scripts/checkpoint-rs.tar')

    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    # print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))

    print("-> loaded checkpoint %s (epoch: %d)"%('grasp_detector/scripts/checkpoint-rs.tar', start_epoch))

    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir=None,color=None,depth=None,mask=None):
    # load data

    color = color.astype(np.float64)/255.0
    color = cv2.rotate(color,cv2.ROTATE_90_CLOCKWISE)
    depth = cv2.rotate(depth,cv2.ROTATE_90_CLOCKWISE)

    # workspace_mask = np.zeros((720,1280),dtype=np.uint8)

    workspace_mask = np.zeros((480,848),dtype=np.uint8)

    workspace_mask = cv2.rotate(workspace_mask,cv2.ROTATE_90_CLOCKWISE)
    idx= depth > 1000

    #pad the bbox
    inflate=30

    workspace_mask[mask[1]-inflate:mask[3]-inflate,mask[0]-inflate:mask[2]+inflate] = 255
    workspace_mask[idx] = 0
    
    # meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    # meta = scio.loadmat(os.path.join('//home/teamf/graspnet/graspnet-baseline/doc/example_data/', 'meta.mat'))
    # intrinsic = meta['intrinsic_matrix']
   

    #image_dim= (1280,720)
    # intrinsic = np.array([[632.3783569335938, 0.0, 641.1390991210938, ],[ 0.0, 632.3783569335938, 356.175537109375,], [0.0, 0.0, 1.0]])
    
    #image_dim= (484,240) 
    # intrinsic = np.array([[301.21, 0.0,212.72],[ 0.0, 300.49, 121.4,], [0.0, 0.0, 1.0]])
    
    #image_dim = (640,480)
    # intrinsic = np.array([[602.4285278320312, 0.0, 321.4526062011719], [0.0, 600.9822998046875, 242.80712890625], [0.0, 0.0, 1.0]])    
    
    #image_dim = (848,480)
    intrinsic  =np.array([[602.428466796875, 0.0, 425.4526062011719],[ 0.0, 600.9822998046875, 242.80712890625], [0.0, 0.0, 1.0]])
    
    factor_depth = 1000

    # generate cloud
    camera = CameraInfo(480, 848, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    # camera = CameraInfo(720, 1280, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    print("camera=",(camera.height,camera.width))
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True) #shape=(480,640,3)
    mask = (workspace_mask & (depth > 0)) #shape =(480,640)

    cloud_masked = cloud[mask>0,:]
    color_masked = color[mask>0,:]

    # sample points
    if len(cloud_masked) >= 20000:
        idxs = np.random.choice(len(cloud_masked), 20000, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), 20000-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
        
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    #get best 10 grasps
    gg = gg[:10]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir=None,color_img=None,depth_img =None,mask=None,vis_flag=False):

    net = get_net()
    end_points, cloud = get_and_process_data(color=color_img,depth=depth_img,mask=mask)
    gg = get_grasps(net, end_points)

    if 0.01> 0:
        gg = collision_detection(gg, np.array(cloud.points))

    if(vis_flag):
        vis_grasps(gg, cloud)
    return gg

