#!/usr/bin/env python2
"""
publishes tf
"""
__author__ = 'Krzysztof Stezala <krzysztof.stezala at student.put.poznan.pl>'
__version__ = '0.1'
__license__ = 'MIT'

## System
import sys, time

import rospy
import tf
import tf2_ros
import tf2_geometry_msgs

from segmentator_2000.srv import LssnTf, LssnTfResponse


class LssnTfServer:
    def __init__(self):
        rospy.init_node('lssn_tf_server')
        rospy.Service('lssn_tf',LssnTf,self.handle_data)
        rospy.spin()

    def handle_data(self,req):
        try:
            tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0))  # tf buffer length
            tf_listener = tf2_ros.TransformListener(tf_buffer)
            tf_transform = tf_buffer.lookup_transform("base_link",
                                                      "xtion_depth_frame",  # source frame
                                                      rospy.Time(0),  # get the tf at first available time
                                                      rospy.Duration(4.0))
            pose_transformed = tf2_geometry_msgs.do_transform_pose(req.obj_pose, tf_transform)
            br = tf.TransformBroadcaster()

            # roll=0
            # pitch=3.14
            # yaw=0
            # q = quaternion_from_euler(roll,pitch,yaw)
            # pose_transformed.pose.orientation.x = q[0]
            # pose_transformed.pose.orientation.y = q[1]
            # pose_transformed.pose.orientation.z = q[2]
            # pose_transformed.pose.orientation.w = q[3]
            pose_transformed.pose.orientation.x = req.obj_pose.pose.orientation.x
            pose_transformed.pose.orientation.y = req.obj_pose.pose.orientation.y
            pose_transformed.pose.orientation.z = req.obj_pose.pose.orientation.z
            pose_transformed.pose.orientation.w = req.obj_pose.pose.orientation.w
            print(pose_transformed)
            br.sendTransform((pose_transformed.pose.position.x,
                              pose_transformed.pose.position.y,
                              pose_transformed.pose.position.z),
                             (pose_transformed.pose.orientation.x,
                              pose_transformed.pose.orientation.y,
                              pose_transformed.pose.orientation.z,
                              pose_transformed.pose.orientation.w), rospy.Time.now(), "cup", "base_link")
            return LssnTfResponse(pose_transformed)
        except(tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("tf not working")




if __name__ == "__main__":
    lssn_tf_server = LssnTfServer()
