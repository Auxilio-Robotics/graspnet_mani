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
from grasp_msgs.msg import grasp
from run_inference_2 import demo

class GraspDetector():

  def __init__(self):

    self.pub = rospy.Publisher('grasp_msg', grasp, queue_size=20)
    self.bridge = CvBridge()
    self.color_sub = message_filters.Subscriber("camera/color/image_raw", Image)
    self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
    self.box_sub = rospy.Subscriber("object_bounding_boxes", String,self.get_bboxes)
    self.cameraInfoSub = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo,self.get_cam_info)

    self.image_scale=3 #rescaling bbox detections on low res image to high res

    self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1, allow_headerless=True)

  def register_callback_and_publish(self,):
    self.ts.registerCallback(self.inference)


  def get_cam_info(self,msg):
    self.intrinsics = np.array(msg.K).reshape(3,3)

  def get_bboxes(self,msg):
    self.bboxes = msg

  def inference(self,color,depth):

    rospy.loginfo(rospy.get_caller_id() + "Images received!!!")
    cv_img_color = self.bridge.imgmsg_to_cv2(color,"bgr8")
    cv_img_depth = self.bridge.imgmsg_to_cv2(depth)
    box_dict = json.loads(self.bboxes.data)
   
    mask = None
    for idx,label in enumerate(box_dict["box_classes"]):
      #print object id label
      print(label)
      if(int(label)==39):
       mask = np.array(box_dict["boxes"][idx]).astype(np.int32)*self.image_scale

    if mask is None:
      print("Warning: Object is not in field of view of the Robot!!")
      exit()

    print(mask)

    gg = demo(color =cv_img_color,depth = cv_img_depth,bbox=mask,intrinsics=self.intrinsics,vis_flag=False)
    
    #print best grasp pose
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
  
    rate = rospy.Rate(10) # 10hz
    rospy.loginfo("Publishing grasp poses to grasp_msg topic")
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