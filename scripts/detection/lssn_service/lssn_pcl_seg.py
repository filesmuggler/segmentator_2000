#!/usr/bin/env python2
"""LssnPclSeg class"""
__docformat__="google"
from segmentator_2000.srv import LssnPclSeg, LssnPclSegResponse

import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import ros_numpy
import numpy as np
import tf
from math import atan2, sqrt, sin, pi
import tf2_ros

class LssnPclSegServer:
    def __init__(self):
        '''Initialize ros service'''
        rospy.init_node('lssn_pcl_seg_server')
        rospy.Service('lssn_pcl_seg', LssnPclSeg, self.handle_data)
        rospy.loginfo("Lssn Pcl Seg ready to handle data.")
        rospy.spin()

    def handle_data(self, req):
        objects = []
        for i,label in enumerate(req.labels):
            if label == "cup":
                depth = ros_numpy.numpify(req.depth_image)
                #masked_depth_f = masked_depth[(masked_depth > 0)]

                z = depth[req.y_s[i]][req.x_s[i]]
                print("z: ",z)
                cx, cy = req.cam_info.K[2], req.cam_info.K[5]
                fx, fy = req.cam_info.K[0], req.cam_info.K[4]
                u, v = req.x_s[i],req.y_s[i]
                # z = masked_depth[u,v]

                # print("u:",u," v:",v)
                # print("new u:",new_u,"new v:",new_v)
                #z = z / 1000.0
                y = z * (v - cy) / fy
                x = -z * (u - cx) / fx

                print("x:",x," y:",y," z:",z)

                pitch_degrees = 0
                # orientation
                try:
                    # tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0))  # tf buffer length
                    # tf_listener = tf2_ros.TransformListener(tf_buffer)
                    # #TODO: parametrize frames in launch file
                    # tf_transform = tf_buffer.lookup_transform("base_link",
                    #                                           "xtion_depth_frame",  # source frame
                    #                                           rospy.Time(0),  # get the tf at first available time
                    #                                           rospy.Duration(4.0))
                    # # pose_transformed = tf2_geometry_msgs.do_transform_pose(bottle_pose, tf_transform)
                    # # print(pose_transformed)
                    # orientation_q = tf_transform.transform.rotation
                    # orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                    # (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
                    # # print(tf_transform)
                    # # print(roll,pitch,yaw)
                    # pitch_degrees = pitch * 180.0 / 3.14
                    # if (pitch_degrees < 0):
                    #     pitch_degrees += 360.0
                    # #print("camera orientation: ", pitch_degrees)
                    # rows = rows - np.mean(rows)
                    # cols = cols - np.mean(cols)
                    # coords = np.vstack([rows, cols])
                    # cov = np.cov(coords)
                    # evals, evecs = np.linalg.eig(cov)
                    #
                    # sort_indices = np.argsort(evals)[::-1]
                    # x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
                    # x_v2, y_v2 = evecs[:, sort_indices[1]]
                    #
                    # scale = 50
                    # if y_v1 < 0:
                    #     y_v1 = -y_v1
                    #
                    # #TODO: refactor bottle into more generic name
                    #
                    # bottle_angle = atan2(y_v1, x_v1)
                    # # bottle_angle_degrees = bottle_angle * 180.0/3.14
                    # # if( bottle_angle_degrees < 0 ):
                    # #     bottle_angle_degrees += 360.0
                    # #print("y_v1: ", y_v1)
                    # #print("object picture angle: ", bottle_angle)
                    #
                    # d = sqrt((y_v1 ** 2 + x_v1 ** 2)) * sin(bottle_angle) * sin(pitch)
                    # bottle_irl = atan2(d, x_v1)
                    # bottle_irl_degrees = bottle_irl * 180.0 / pi
                    # if (bottle_irl_degrees < 0):
                    #     bottle_irl_degrees += 360.0
                    # bottle_irl_degrees = bottle_irl_degrees - 90
                    # #print("object irl angle: ", bottle_irl_degrees)

                    obj_pose = PoseStamped()
                    obj_pose.pose.position.x = z
                    obj_pose.pose.position.y = x
                    obj_pose.pose.position.z = -y
                    # bottle_irl_rad = self.degrees_to_radians(bottle_irl_degrees)
                    #
                    # roll = 0
                    # pitch = 3.14
                    # yaw = bottle_irl_rad
                    # q = quaternion_from_euler(roll, pitch, yaw)
                    obj_pose.pose.orientation.x = 0.707
                    obj_pose.pose.orientation.y = 0.707
                    obj_pose.pose.orientation.z = 0.0
                    obj_pose.pose.orientation.w = 0.0
                    #TODO: parametrize lin names in launchfiles
                    obj_pose.header.frame_id = "xtion_depth_frame"
                    obj_pose.header.seq = 1
                    obj_pose.header.stamp = rospy.Time.now()
                    objects.append(obj_pose)

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    print ("tf not working")

                # plot.imshow(masked_depth)
                # plot.scatter(u, v, s=10,c="blue")
                # plot.scatter((u_max+u_min)/2, (v_max+v_min)/2, s=10,c="red")
                # plot.show()

                # plot.plot(rows, cols, 'k.')
                # plot.plot([x_v1*-scale*2, x_v1*scale*2],
                #         [y_v1*-scale*2, y_v1*scale*2], color='red')
                # plot.plot([x_v2*-scale, x_v2*scale],
                #         [y_v2*-scale, y_v2*scale], color='blue')

                # plot.axis('equal')
                # plot.gca().invert_yaxis()  # Match the image system with origin at top left
                # plot.show()

            # final_ob = PoseStamped()
            # final_ob.pose.position.z = 1000.0
            # for ob in objects:
            #     if ob.pose.position.z < final_ob.pose.position.z:
            #         final_ob = ob

        if len(objects) > 0:
            return LssnPclSegResponse(objects[0])
        else:
            final_ob = PoseStamped()
            return LssnPclSegResponse(final_ob)

    def degrees_to_radians(self, degrees):
        radians = 2.0 * pi * degrees / 360.0
        return radians

if __name__ == "__main__":
    lssn_pcl_seg_server = LssnPclSegServer()


