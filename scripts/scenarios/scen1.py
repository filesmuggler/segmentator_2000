#!/usr/bin/env python3

from segmentator_2000.srv import *
import sys
import copy
import rospy
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
import geometry_msgs
import message_filters
import tf

import tf2_ros
import tf2_geometry_msgs
from tf.transformations import quaternion_from_euler

import numpy as np
import rosservice

lssn_detect_prox = rospy.ServiceProxy("/lssn_detect", LssnDetect)
lssn_rgb_prox = rospy.ServiceProxy('/lssn_rgb_seg', LssnRgbSeg)
lssn_pcl_prox = rospy.ServiceProxy('/lssn_pcl_seg', LssnPclSeg)


wait_delay = 0.2


def main():
    rospy.init_node('scenario1')
    detected = False
    while not detected:
        resp = lssn_detect_prox(["cup"])
        detected = resp.detected
        print("Bottle ", detected)
    print("Bottle ", detected)
    resp = LssnRgbSegResponse()
    resp = lssn_rgb_prox(["cup"])
    obj_poses = lssn_pcl_prox(resp.cam_info,
                              resp.rgb_image,
                              resp.depth_image,
                              resp.class_list,
                              resp.mask_list)
    bottle_pose = obj_poses.obj_poses
    print(bottle_pose)

    try:
        tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0))  # tf buffer length
        tf_listener = tf2_ros.TransformListener(tf_buffer)
        tf_transform = tf_buffer.lookup_transform("odom",
                                                  bottle_pose.header.frame_id,  # source frame
                                                  rospy.Time(0),  # get the tf at first available time
                                                  rospy.Duration(4.0))
        pose_transformed = tf2_geometry_msgs.do_transform_pose(bottle_pose, tf_transform)
        br = tf.TransformBroadcaster()

        roll = 0
        pitch = 3.14
        yaw = 0
        q = quaternion_from_euler(roll, pitch, yaw)
        pose_transformed.pose.orientation.x = q[0]
        pose_transformed.pose.orientation.y = q[1]
        pose_transformed.pose.orientation.z = q[2]
        pose_transformed.pose.orientation.w = q[3]
        # print(pose_transformed)
        while True:
            br.sendTransform((pose_transformed.pose.position.x,
                              pose_transformed.pose.position.y,
                              pose_transformed.pose.position.z),
                             (pose_transformed.pose.orientation.x,
                              pose_transformed.pose.orientation.y,
                              pose_transformed.pose.orientation.z,
                              pose_transformed.pose.orientation.w), rospy.Time.now(), "cup", "odom")

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        print("tf not working")

    # self.posestamped_pub.publish(pose_transformed)


if __name__ == "__main__":
    main()

# rosservice call /goto_object_service "object_coordinates:
#   position:
#     x: 0.20
#     y: 0.69
#     z: 0.23
#   orientation:
#     x: 0.0
#     y: 0.9999997
#     z: 0.0
#     w: 0.0007963"