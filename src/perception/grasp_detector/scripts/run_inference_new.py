#!/usr/bin/env python3

""" Runs inference on the subscribed images and returns the grasp poses
    Author: chenxi-wang, Abhinav Gupta
"""

import os
import sys
import numpy as np
import open3d as o3d
from open3d import *
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import cv2
import math
import itertools

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

def preprocess_pcd(cloud):

    #preprocess the point cloud. Removes noise/outliers and applies min/max bounds

    # bounds = [[-math.inf, math.inf], [-math.inf, math.inf], [1, 2]]  # set the bounds
    # bounding_box_points = list(itertools.product(*bounds))  # create limit points
    # bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
    #     o3d.utility.Vector3dVector(bounding_box_points))  # create bounding box object
    # cloud_cropped = cloud.crop(bounding_box)
    # pcd_stat, ind_stat = cloud_cropped.remove_statistical_outlier(nb_neighbors=30,std_ratio=5)

    cloud_cropped=cloud
   
    return cloud_cropped


def get_and_process_data(data_dir=None,color=None,depth=None,intrinsic=None,mask=None):
    # load data

    color = color.astype(np.float64)/255.0
    color = cv2.rotate(color,cv2.ROTATE_90_CLOCKWISE)
    depth = cv2.rotate(depth,cv2.ROTATE_90_CLOCKWISE)

    workspace_mask = np.zeros((720,1280),dtype=np.uint8)

    # workspace_mask = np.zeros((480,848),dtype=np.uint8)

    workspace_mask = cv2.rotate(workspace_mask,cv2.ROTATE_90_CLOCKWISE)
    min_depth=1000

    #pad the bbox
    inflate=30

    # workspace_mask[mask[1]-inflate:mask[3]-inflate,mask[0]-inflate:mask[2]+inflate] = 255

    workspace_mask[:,:] = 255

    # workspace_mask[idx] = 0
    
    # meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    # meta = scio.loadmat(os.path.join('//home/teamf/graspnet/graspnet-baseline/doc/example_data/', 'meta.mat'))
    # intrinsic = meta['intrinsic_matrix']

    factor_depth = 1000

    # generate cloud
    # camera = CameraInfo(480, 848, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    camera = CameraInfo(720, 1280, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)

    print("camera=",(camera.height,camera.width))
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True) #shape=(480,640,3)
    mask = (workspace_mask & (depth > min_depth)) #shape =(480,640)

    cloud_masked = cloud[mask>0,:]
    color_masked = color[mask>0,:]

    #storing pt clouds

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    #preprocess the point clouds. Remove noise (outliers), max/min bounds, plane surface of table.
    # Create bounding box:

    # cloud_filtered = preprocess_pcd(cloud)

    # cloud_pts = np.asarray(cloud_filtered.points)
    # color_pts = np.asarray(cloud_filtered.colors)

    # sample points
    if len(cloud_masked) >= 20000:
        idxs = np.random.choice(len(cloud_masked), 20000, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), 20000-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)

    # sample points
    # if len(cloud_pts) >= 20000:
    #     idxs = np.random.choice(len(cloud_pts), 20000, replace=False)
    # else:
    #     idxs1 = np.arange(len(cloud_pts))
    #     idxs2 = np.random.choice(len(cloud_pts), 20000-len(cloud_pts), replace=True)
    #     idxs = np.concatenate([idxs1, idxs2], axis=0)
        
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # cloud_sampled = cloud_pts[idxs]
    # color_sampled = color_pts[idxs]

    # convert data
    
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

def demo(data_dir=None,color_img=None,depth_img =None,mask=None,intrinsics=None,vis_flag=False):

    net = get_net()
    end_points, cloud = get_and_process_data(color=color_img,depth=depth_img,intrinsic=intrinsics,mask=mask)
    gg = get_grasps(net, end_points)

    if 0.01> 0:
        gg = collision_detection(gg, np.array(cloud.points))

    if(vis_flag):
        vis_grasps(gg, cloud)
    return gg

