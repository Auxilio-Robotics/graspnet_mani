#!/usr/bin/env python3
# Capturing image data and running graspnet. Publishing on grasp_msg topic

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import message_filters
from std_msgs.msg import String
import json

from rospy.numpy_msg import numpy_msg
from grasp_msgs.msg import grasp
from run_inference import demo

#define publisher
global pub
pub = rospy.Publisher('grasp_msg', grasp, queue_size=20)
bridge = CvBridge()

def callback(color,depth,boxes_msg):
    global index,pub
 
    rospy.loginfo(rospy.get_caller_id() + "I see depth image")
    cv_img_color = bridge.imgmsg_to_cv2(color,"bgr8")
    cv_img_depth = bridge.imgmsg_to_cv2(depth)

    box_dict = json.loads(boxes_msg.data)

    for idx,label in enumerate(box_dict["box_classes"]):
      #print object id label
      print(label)
      if(int(label)==39):
       mask = np.array(box_dict["boxes"][idx]).astype(np.int32)

    gg = demo(color_img =cv_img_color,depth_img = cv_img_depth,mask=mask,vis_flag=True)
    
    #print best grasp pose
    print(gg[0])

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
    pub.publish(g)
    rate.sleep()

  
def listener():
 
    rospy.init_node('listener', anonymous=True)
    color_sub = message_filters.Subscriber("camera/color/image_raw", Image)
    depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image)
    box_sub = message_filters.Subscriber("object_bounding_boxes", String)

    ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub,box_sub], 10, 0.1, allow_headerless=True)
    ts.registerCallback(callback)
    rospy.spin()

def main():
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
  listener()
  main()