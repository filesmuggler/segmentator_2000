#!/usr/bin/env python3
"""
Cuts the image from camera feed using YOLOv5
"""
__author__ = 'Krzysztof Stezala <krzysztof.stezala at student.put.poznan.pl>'
__version__ = '0.1'
__license__ = 'MIT'

## System
import sys, time
import argparse
import os
import shutil
from pathlib import Path

## ROS
import numpy as np
import pandas as pd
from scipy.ndimage import filters
import roslib
import rospy
import ros_numpy

from sensor_msgs.msg import Image as sImage
from geometry_msgs.msg import Pose, PoseStamped
from segmentator_2000.srv import LssnRgbSeg, LssnRgbSegResponse
from segmentator_2000.srv import LssnPclSeg, LssnPclSegResponse

import tf
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import quaternion_from_euler

import cv2

##torch
import torch as th
import torchvision.transforms as T
import requests
from PIL import Image, ImageDraw, ImageFont

th.set_grad_enabled(False)
th.cuda.empty_cache()

model = th.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
model.eval()
model = model.cuda()

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

CONF_THRESH = 0.6
frame_counter = 0


class bootle_detect:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        self.confidence = 0.6
        self.last_frames = []
        self.rate = rospy.Rate(5)
        # topic where to publish
        self.image_pub = rospy.Publisher("/lssn_detect", sImage, queue_size=10)
        self.image_sub = rospy.Subscriber("/xtion/rgb/image_raw", sImage, self.callback, queue_size=1)
        self.posestamped_pub = rospy.Publisher("/lssn_pose", PoseStamped, queue_size=10)
        rospy.wait_for_service('lssn_rgb_seg')
        self.object_class = rospy.get_param('~object_class', 'cup')


    def callback(self, data):
        try:
            np_image = ros_numpy.numpify(data)

            with th.no_grad():
                self.detect(np_image)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def detect(self, np_image):
        # run inference
        now = rospy.Time.now()
        img = Image.fromarray(np_image).resize((800, 600)).convert('RGB')
        img_tens = transform(img).unsqueeze(0).cuda()
        with th.no_grad():
            output = model(img_tens)

        # output['pred_logits'] contains output tensor with
        # not normalized results for each detected object
        #
        # [0] extracts from tensor tensor
        #
        # softmax(-1) normalizes output from 0 to 1
        #
        # [0,:,:-1] removes "no-object" class
        pred_logits = output['pred_logits'].softmax(-1)[0, :, :-1]
        pred_boxes = output['pred_boxes'][0]

        # pred_logits=output['pred_logits'][0][:, :len(CLASSES)]
        # pred_boxes=output['pred_boxes'][0]

        max_output = pred_logits.max(-1)

        m_output_inds = (max_output.values > self.confidence).nonzero(as_tuple=True)[0]

        pred_logits = pred_logits[m_output_inds]
        pred_boxes = pred_boxes[m_output_inds]
        pred_logits.shape

        im2 = img.copy()
        drw = ImageDraw.Draw(im2)

        labels = []

        for logits, box in zip(pred_logits, pred_boxes):
            cls = logits.argmax()
            if cls >= len(CLASSES):
                continue
            label = CLASSES[cls]
            labels.append(label)
            box = box.cpu() * th.Tensor([800, 600, 800, 600])
            x, y, w, h = box
            x0, x1 = x - w // 2, x + w // 2
            y0, y1 = y - h // 2, y + h // 2
            drw.rectangle([x0, y0, x1, y1], outline='red', width=5)
            drw.text((x, y), label, fill='pink')

        if "cup" in labels:
            self.last_frames.append(True)
        else:
            self.last_frames.append(False)
        end = rospy.Time.now()
        print("Time detection: ", (end - now).to_sec())

        if len(self.last_frames) > 10:
            # keep size of the list
            self.last_frames.pop(0)
            # check if bottle appeared in all of them
            if False in self.last_frames:
                print("no object detected")
            else:
                print("detected")
                print("starting service rgb")
                now = rospy.Time.now()
                lssn_rgb_prox = rospy.ServiceProxy('lssn_rgb_seg', LssnRgbSeg)
                resp = LssnRgbSegResponse()
                resp = lssn_rgb_prox(["cup"])
                end = rospy.Time.now()
                print("Time rgb segmentation: ", (end - now).to_sec())
                print("starting service pcl")
                now = rospy.Time.now()
                lssn_pcl_prox = rospy.ServiceProxy('lssn_pcl_seg', LssnPclSeg)
                obj_poses = lssn_pcl_prox(resp.cam_info,
                                          resp.rgb_image,
                                          resp.depth_image,
                                          resp.class_list,
                                          resp.mask_list)
                bottle_pose = obj_poses.obj_poses
                end = rospy.Time.now()
                print("Time pcl segmentation: ", (end - now).to_sec())
                if bottle_pose.header.frame_id == "xtion_depth_frame":
                    try:
                        tf_buffer = tf2_ros.Buffer(rospy.Duration(1.0))  # tf buffer length
                        tf_listener = tf2_ros.TransformListener(tf_buffer)
                        tf_transform = tf_buffer.lookup_transform("odom",
                                                                  bottle_pose.header.frame_id,  # source frame
                                                                  rospy.Time(0),  # get the tf at first available time
                                                                  rospy.Duration(4.0))
                        pose_transformed = tf2_geometry_msgs.do_transform_pose(bottle_pose, tf_transform)
                        br = tf.TransformBroadcaster()

                        # roll=0
                        # pitch=3.14
                        # yaw=0
                        # q = quaternion_from_euler(roll,pitch,yaw)
                        # pose_transformed.pose.orientation.x = q[0]
                        # pose_transformed.pose.orientation.y = q[1]
                        # pose_transformed.pose.orientation.z = q[2]
                        # pose_transformed.pose.orientation.w = q[3]
                        pose_transformed.pose.orientation.x = bottle_pose.pose.orientation.x
                        pose_transformed.pose.orientation.y = bottle_pose.pose.orientation.y
                        pose_transformed.pose.orientation.z = bottle_pose.pose.orientation.z
                        pose_transformed.pose.orientation.w = bottle_pose.pose.orientation.w
                        print(pose_transformed)
                        br.sendTransform((pose_transformed.pose.position.x,
                                          pose_transformed.pose.position.y,
                                          pose_transformed.pose.position.z),
                                         (pose_transformed.pose.orientation.x,
                                          pose_transformed.pose.orientation.y,
                                          pose_transformed.pose.orientation.z,
                                          pose_transformed.pose.orientation.w), rospy.Time.now(), "cup", "odom")

                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        print("tf not working")

                    self.posestamped_pub.publish(pose_transformed)

                self.last_frames = []
                self.rate.sleep()

        msg = sImage()
        msg.header.stamp = rospy.Time.now()
        msg.height = im2.height
        msg.width = im2.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = 3 * im2.width
        msg.data = np.array(im2).tobytes()
        self.image_pub.publish(msg)


def main():
    '''Initializes and cleanup ros node'''
    time.sleep(3)
    rospy.init_node('lssn_detr_detection', anonymous=True)
    rospy.get_param('object_class', 'cup')
    bd = bootle_detect()
    try:
        rospy.loginfo("LSSN Detection DETR -> is RUN")
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image detection module")

    cv2.destroyAllWindows()


# main()


if __name__ == '__main__':
    main()