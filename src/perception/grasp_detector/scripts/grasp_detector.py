#!/usr/bin/env python3
# Capturing image data and running graspnet. Publishing on grasp_msg topic

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
import json

from rospy.numpy_msg import numpy_msg
from grasp_msgs.msg import grasp,grasp_pose
from run_inference_stretch import demo,centroid_grasp

import tf
from tf import TransformerROS
from tf import TransformListener, TransformBroadcaster
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

class GraspDetector():

  def __init__(self):

    # self.pub = rospy.Publisher('grasp_msg', String, queue_size=20)

    self.pub = rospy.Publisher('grasp_msg', grasp, queue_size=20)
    self.grasp_pose_pub = rospy.Publisher('grasp_pose_xyzt', grasp_pose, queue_size=20)
    self.bridge = CvBridge()
    self.color_sub = message_filters.Subscriber("camera/color/image_raw", Image)
    self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
    self.box_sub = rospy.Subscriber("object_bounding_boxes", String,self.get_bboxes)
    self.cameraInfoSub = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo,self.get_cam_info)

    self.image_scale=3 #rescaling bbox detections on low res image to high res

    self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1, allow_headerless=True)
    self.tf_ros = TransformerROS()
    self.tf_listener = TransformListener()
    self.tf_publisher = TransformBroadcaster()

  def register_callback_and_publish(self,):
    self.ts.registerCallback(self.inference)


  def get_cam_info(self,msg):
    self.intrinsics = np.array(msg.K).reshape(3,3)

  def get_bboxes(self,msg):
    self.bboxes = msg

  def get_transform(self,target_frame:str,source_frame:str):

    print(self.tf_listener.allFramesAsString())

    # if self.tf_listener.frameExists(target_frame) and self.tf_listener.frameExists(source_frame):
    t = self.tf_listener.getLatestCommonTime(target_frame, source_frame)
    position, quaternion = self.tf_listener.lookupTransform(target_frame, source_frame, t)

    #if you face error that transform not found. launch stretch_driver before running realsense node.
    #realsense node publishes o tf_static and it overrides all tf variables for some reason.
    #the distance between link head pan to base link is 1.35 in gazebo but 1.32 through tf transforms.
    # print ((position, quaternion))

    tf_matrix = self.tf_ros.fromTranslationRotation(position,quaternion)

    # print("base2cam=",tf_matrix)

    return tf_matrix

  def publish_transform(self,tf_matrix: np.ndarray):
    quat = tf.transformations.quaternion_from_matrix(tf_matrix)
    self.tf_publisher.sendTransform((tf_matrix[0][3],tf_matrix[1][3],tf_matrix[2][3])
                                    ,quat
                                    ,rospy.Time.now()
                                    ,"/grasp"
                                    ,"/base_link")
    
    angles = tf.transformations.euler_from_quaternion(quat)

    angles=np.array(angles)
    angles[0]=0
    angles[1]=0

    roll,pitch,yaw = angles[0],angles[1],angles[2]

    #ignores pitch and roll.
    quat_yaw_only = tf.transformations.quaternion_from_euler(roll,pitch,yaw)

    #publishes the grasp pose with only yaw angle
    self.tf_publisher.sendTransform((tf_matrix[0][3],tf_matrix[1][3],tf_matrix[2][3])
                                    ,quat_yaw_only
                                    ,rospy.Time.now()
                                    ,"/grasp_yaw"
                                    ,"/base_link")



  def publish_grasp_pose(self,tf_matrix: np.ndarray):
    _,_,yaw = tf.transformations.euler_from_matrix(tf_matrix)

    g_pose = grasp_pose()
    g_pose.x = tf_matrix[0][3]
    g_pose.y = tf_matrix[1][3]
    g_pose.z = tf_matrix[2][3]
    g_pose.yaw = yaw

    self.grasp_pose_pub.publish(g_pose)


  def inference(self,color,depth):

    rospy.loginfo(rospy.get_caller_id() + "Images received!!!")
    cv_img_color = self.bridge.imgmsg_to_cv2(color,"bgr8")
    cv_img_depth = self.bridge.imgmsg_to_cv2(depth)
    box_dict = json.loads(self.bboxes.data)

    # define source and target frame
    source_frame= "/camera_aligned_depth_to_color_frame"
    target_frame= "/base_link"

    tf_mat_base2cam = self.get_transform(target_frame,source_frame)

    print(tf_mat_base2cam)

    r = R.from_matrix(tf_mat_base2cam[:3,:3])

    print("RPY=")
    print(r.as_euler('xyz', degrees=True))
   
    mask = None
    for idx,label in enumerate(box_dict["box_classes"]):
      #print object id label
      print(label)
      if(int(label)==39):
       mask = np.array(box_dict["boxes"][idx]).astype(np.int32)*self.image_scale

    if mask is None:
      print("Warning: Object is not in field of view of the Robot!!")
      return

    gg=None
    gg_c=None

    #uncomment this to run graspnet
    gg = demo(color =cv_img_color,depth = cv_img_depth,bbox=mask,
              intrinsics=self.intrinsics,vis_flag=False)
 
    gg_c = centroid_grasp(color =cv_img_color,depth = cv_img_depth,bbox=mask,intrinsics=self.intrinsics,vis_flag=False)


    if(len(gg)==0):
      gg_c = centroid_grasp(color =cv_img_color,depth = cv_img_depth
                            ,bbox=mask,intrinsics=self.intrinsics,vis_flag=False)
      print("Initiating centroid grasp........")
      print("Centroid=",gg_c)
      gg_c = np.vstack((gg_c.reshape(3,1),np.array([[1]])))

      tf_base2obj = np.matmul(tf_mat_base2cam,gg_c)
      g=grasp_pose()
      g.x = gg_c[0]
      g.y = gg_c[1] 
      g.z = gg_c[2] 
      g.yaw = 0

      rate = rospy.Rate(10) # 10hz
      rospy.loginfo("Publishing centroid grasp poses to grasp_msg topic")
      self.pub.publish(json.dumps({'xyzw' : [g.x, g.y, g.z, g.yaw]}))
      rate.sleep()

    else:
      print("Best Grasp Pose: ",gg[0])
      #storing data in grasp msg
      g = grasp()
      g.object_id = gg[0].object_id
      g.score = gg[0].score
      g.width = gg[0].width
      g.height = gg[0].height
      g.depth = gg[0].depth
      g.translation = gg[0].translation.astype(np.float32)
      g.rotation = np.ravel(gg[0].rotation_matrix.astype(np.float32))

      rot_mat_grasp = gg[0].rotation_matrix.astype(np.float32)
      t_grasp = gg[0].translation.astype(np.float32)

      #Adjusting for x offset
      t_grasp[0] = t_grasp[0]
      
      tf_mat_cam2obj = np.hstack((rot_mat_grasp,t_grasp.reshape((3,1))))
      tf_mat_cam2obj = np.vstack((tf_mat_cam2obj,np.array([[0,0,0,1]])))

      r1 = R.from_euler('x', 180, degrees=True)
      r2 = R.from_euler('y', 90, degrees=True)

      # tf_correct = np.matmul(r1.as_matrix(),r2.as_matrix())

      # tf_correct = np.hstack((tf_correct,np.zeros((3,1))))
      # tf_correct = np.vstack((tf_correct,np.array([[0,0,0,1]])))

      # tf_base2obj = np.matmul(np.matmul(tf_mat_base2cam,tf_correct),tf_mat_cam2obj)

      tf_base2obj = np.matmul(tf_mat_base2cam,tf_mat_cam2obj)

      # print("cam2obj=",tf_mat_cam2obj)
      # print('trans', t_grasp.reshape((3,1)))

      #publish transform T_base2obj and grasp_pose
      self.publish_transform(tf_base2obj)
      self.publish_grasp_pose(tf_base2obj)

      print("tf_base2obj",tf_base2obj)
  
      rate = rospy.Rate(10) # 10hz
      rospy.loginfo("Publishing GraspNet grasp poses to grasp_msg topic")
      self.pub.publish(g)
      rate.sleep()
    
def main():
  rospy.init_node('image_converter', anonymous=True)
  try:
    grasp_gen = GraspDetector()
    grasp_gen.register_callback_and_publish()
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
  main()