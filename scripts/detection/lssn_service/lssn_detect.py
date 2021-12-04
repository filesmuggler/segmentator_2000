#!/usr/bin/env python3

from segmentator_2000.srv import LssnDetect, LssnDetectResponse
import rospy
import message_filters
import sensor_msgs.msg

# facebook
from PIL import Image
import io
import torch
import torchvision.transforms as T
import numpy
from math import floor

from sensor_msgs.msg import Image as sImage
from geometry_msgs.msg import Pose, PoseStamped
from segmentator_2000.srv import LssnRgbSeg, LssnRgbSegResponse
from segmentator_2000.srv import LssnPclSeg, LssnPclSegResponse
from segmentator_2000.srv import LssnTf, LssnTfResponse

from panopticapi.utils import rgb2id
from copy import deepcopy

from skimage.transform import resize
import ros_numpy
import matplotlib.pyplot as plot

torch.set_grad_enabled(False)
#torch.cuda.empty_cache()

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True,
                                      return_postprocessor=True, num_classes=250)
model.eval()
#model = model.cuda()

## Classes from COCO
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
PANOPTIC_STUFF = ['things', 'banner', 'blanket', 'bridge', 'cardboard', 'counter',
                  'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house',
                  'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad',
                  'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel',
                  'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water', 'window-blind', 'window',
                  'tree', 'fence', 'ceiling', 'sky', 'cabinet', 'table', 'floor', 'pavement', 'mountain',
                  'grass', 'dirt', 'paper', 'food', 'building', 'rock', 'wall', 'rug']


class lssn_detect:
    def __init__(self):
        '''Initialize ros service'''
        rospy.init_node('lssn_detect_server')
        #TODO
        rgb_topic = "/xtion/rgb/image_raw"
        depth_topic = "/xtion/depth_registered/image_raw"
        camera_topic = "/xtion/depth_registered/camera_info"
        sub_rgb = message_filters.Subscriber(rgb_topic, sensor_msgs.msg.Image)
        self.cache_rgb = message_filters.Cache(sub_rgb, cache_size=1, allow_headerless=True)

        sub_depth = message_filters.Subscriber(depth_topic, sensor_msgs.msg.Image)
        self.cache_depth = message_filters.Cache(sub_depth, cache_size=5, allow_headerless=True)

        sub_info = message_filters.Subscriber(camera_topic, sensor_msgs.msg.CameraInfo)
        self.cache_info = message_filters.Cache(sub_info, cache_size=5, allow_headerless=True)

        self.rgb_img = numpy.empty((640,480))
        self.object_class = rospy.get_param('~object_class', 'cup')
        self.confidence = rospy.get_param('~confidence', 0.5)
        self.flipped_img = rospy.get_param('~flipped', False)
        s = rospy.Service('lssn_detect', LssnDetect, self.handle_data)
        self.posestamped_pub = rospy.Publisher("/lssn_pose", PoseStamped, queue_size=10)
        rospy.wait_for_service('lssn_tf')
        rospy.loginfo("Lssn Detect ready to handle data.")
        rospy.spin()


    def check_usr_classes(self, req):
        user_cls = req.classes
        user_cls = list(map(str.lower, user_cls))
        na_coco = list(set(user_cls) - set(CLASSES))
        na_pano = list(set(user_cls) - set(PANOPTIC_STUFF))
        na_coco_set = set(na_coco)
        missing_cls = list(na_coco_set.intersection(na_pano))
        if len(missing_cls):
            rospy.loginfo("%s will not be found." % missing_cls)
        user_cls = list(set(user_cls) - set(missing_cls))
        return user_cls

    def handle_data(self, req):
        user_cls = self.check_usr_classes(req)
        rospy.loginfo("Looking for: %s" % user_cls)

        data_info = self.cache_info.cache_msgs[-1]
        data_depth = self.cache_depth.cache_msgs[-1]

        data_rgb = self.cache_rgb.cache_msgs[-1]
        image_rgb = ros_numpy.numpify(data_rgb)

        # # mean-std normalize the input image (batch-size: 1)
        img = Image.fromarray(numpy.uint8(image_rgb)).resize((800, 600)).convert('RGB')
        # plot.imshow(img)
        # plot.axis('off')
        # plot.show()
        #img_tens = transform(img).unsqueeze(0).cuda()
        img_tens = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tens)

        pred_logits = output['pred_logits'].softmax(-1)[0, :, :-1]
        pred_boxes = output['pred_boxes'][0]

        max_output = pred_logits.max(-1)

        m_output_inds = (max_output.values > self.confidence).nonzero(as_tuple=True)[0]

        pred_logits = pred_logits[m_output_inds]
        pred_boxes = pred_boxes[m_output_inds]

        labels = []
        real_x_s = []
        real_y_s = []

        for logits, box in zip(pred_logits, pred_boxes):
            cls = logits.argmax()
            if cls >= len(CLASSES):
                continue
            label = CLASSES[cls]
            labels.append(label)
            box = box.cpu() * torch.Tensor([800, 600, 800, 600])
            x, y, w, h = box
            real_x = floor(640 * x.item() / 800)
            real_y = floor(480 * y.item() / 600)
            print(label, real_x, real_y)
            real_x_s.append(real_x)
            real_y_s.append(real_y)

        if "cup" in labels:
            # lssn_rgb_prox = rospy.ServiceProxy('lssn_rgb_seg', LssnRgbSeg)
            # resp = LssnRgbSegResponse()
            # resp = lssn_rgb_prox(["cup"])
            # end = rospy.Time.now()
            # print("Time rgb segmentation: ", (end - now).to_sec())
            # print("starting service pcl")
            # now = rospy.Time.now()
            lssn_pcl_prox = rospy.ServiceProxy('lssn_pcl_seg', LssnPclSeg)
            obj_poses = lssn_pcl_prox(data_info,
                                     data_depth,
                                     labels,
                                     real_x_s,
                                     real_y_s)
            bottle_pose = obj_poses.obj_poses
            # end = rospy.Time.now()
            # print("Time pcl segmentation: ", (end - now).to_sec())
            if bottle_pose.header.frame_id == "xtion_depth_frame":
                lssn_tf_prox = rospy.ServiceProxy('lssn_tf', LssnTf)
                res_pose = lssn_tf_prox(bottle_pose)
                #
                self.posestamped_pub.publish(res_pose.obj_pose)
            return LssnDetectResponse(True)
        else:
            return LssnDetectResponse(False)


if __name__ == "__main__":
    lssn_detect_server = lssn_detect()
