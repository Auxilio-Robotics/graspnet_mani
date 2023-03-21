#!/usr/bin/env python3

""" Runs inference on the subscribed images and returns the grasp poses
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
from typing import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

class GraspArgs:
    def __init__(self,checkpoint_path=None,num_point=20000,num_view = 300,collision_thresh=0.01,voxel_size=0.01,max_depth=1200):
        
        if(checkpoint_path is None):
            self.checkpoint_path = os.path.join(os.getcwd(),"checkpoint-rs.tar")
        else:
            self.checkpoint_path = checkpoint_path

        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh=collision_thresh
        self.voxel_size = voxel_size
        self.max_depth=max_depth

class GraspGenerator:

    def __init__(self,grasp_args: GraspArgs, color:np.ndarray,depth:np.ndarray,bbox:np.ndarray,intrinsics:np.ndarray):
        #initialize grasp arguments
        self.grasp_args = grasp_args

        #Initialize parameters
        self.color = color
        self.depth = depth
        self.bbox=bbox
        self.workspace_mask = np.zeros((self.color.shape[0],self.color.shape[1]),dtype=np.uint8)
        self.net = self.get_net()

        #initialize intrinsics
        self.intrinsic = intrinsics

        # print(self.intrinsic)

    def get_net(self):

        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        # Load checkpoint
        checkpoint = torch.load(self.grasp_args.checkpoint_path)

        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.grasp_args.checkpoint_path, start_epoch))

        # set model to eval mode
        net.eval()

        return net
    
    def _print_shapes(self,):
        print("Color image shape=",self.color.shape)
        print("Depth image shape=",self.depth.shape)
        print("Workspace mask shape=",self.workspace_mask.shape)

    def get_and_process_data(self):
        
        #preprocess the images
        self.color = self.color.astype(np.float64)/255.0

        inflate=20
        # idx= self.depth > self.grasp_args.max_depth

        self.workspace_mask[self.bbox[1]-inflate:self.bbox[3]-inflate,self.bbox[0]-inflate:self.bbox[2]+inflate] = 255

        # self.workspace_mask[self.bbox[0]-inflate:self.bbox[2]-inflate,self.bbox[1]-inflate:self.bbox[3]+inflate] = 255

        
        # print(np.count_nonzero(self.workspace_mask))
        # self.workspace_mask[:,:] = 255

        # cv2.imshow('Image',self.color*np.reshape(self.workspace_mask,(720,1280,1))/255)
        # cv2.waitKey(0)
        # self.workspace_mask[idx] = 0
        # self.workspace_mask[100:600,600:900] = 255
        
        factor_depth = 1000
        camera = CameraInfo(self.color.shape[1], self.color.shape[0], self.intrinsic[0][0], self.intrinsic[1][1], self.intrinsic[0][2], self.intrinsic[1][2], factor_depth)

        # print("camera=",(camera.height,camera.width))
        cloud = create_point_cloud_from_depth_image(self.depth, camera, organized=True) #shape=(480,640,3)
        mask = (self.workspace_mask & (self.depth > 0)) #shape =(720,1280)
      
        cloud_masked = cloud[mask>0,:]
        color_masked = self.color[mask>0,:]

        # print("Centroid=",cloud_masked.mean(axis=0))

        # print("max=",cloud_masked.max(axis=0))
        # print("min=",cloud_masked.min(axis=0))
        # print("num_pts=",cloud_masked.shape)

        # sample points
        if len(cloud_masked) >= 20000:
            idxs = np.random.choice(len(cloud_masked), 20000, replace=False)
        else:
            print("Cloud_masked shape",)
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

    def get_grasps(self,net, end_points):
        # Forward pass
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg
    
    def get_centroid(self,end_points:dict):

        pts = end_points["point_clouds"].to("cpu").detach().numpy()
        pts=pts.squeeze(axis=0)
        #calculate median for centroid to avoid outliers
        return np.median(pts,axis=0)


    def collision_detection(self,gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
        gg = gg[~collision_mask]
        return gg
    
    def sort_grasps(self,gg):
        gg.nms()
        gg.sort_by_score()
        print("Sorting grasps - -- - - - - - - -")
        

    def vis_grasps(self,gg, cloud):
        # gg = self.sort_grasps(gg)
        gg.nms()
        gg.sort_by_score()
        gg = gg[:10]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

def demo(color=None,depth=None,bbox=None,intrinsics=None,vis_flag=False):

    #runs inference on images and return grasp poses
    grasp_args = GraspArgs()
    grasp_obj = GraspGenerator(grasp_args,color=color,depth=depth,bbox=bbox,intrinsics=intrinsics)

    end_points, cloud = grasp_obj.get_and_process_data()
    gg = grasp_obj.get_grasps(grasp_obj.net, end_points)

    if grasp_args.collision_thresh> 0:
        gg = grasp_obj.collision_detection(gg, np.array(cloud.points))

    if(vis_flag):
        grasp_obj.vis_grasps(gg, cloud)

    else:
        grasp_obj.sort_grasps(gg)
        
    return gg

def centroid_grasp(color=None,depth=None,bbox=None,intrinsics=None,vis_flag=False):

    #runs inference on images and return grasp poses
    grasp_args = GraspArgs()
    grasp_obj = GraspGenerator(grasp_args,color=color,depth=depth,bbox=bbox,intrinsics=intrinsics)

    end_points, cloud = grasp_obj.get_and_process_data()
    centroid = grasp_obj.get_centroid(end_points)
    print("centroid=",centroid)

    return centroid