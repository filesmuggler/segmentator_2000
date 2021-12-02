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

from panopticapi.utils import rgb2id
from copy import deepcopy

from skimage.transform import resize
import ros_numpy
import matplotlib.pyplot as plot

torch.set_grad_enabled(False)
torch.cuda.empty_cache()

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True,
                                      return_postprocessor=True, num_classes=250)
model.eval()
model = model.cuda()

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
        sub_rgb = message_filters.Subscriber(rgb_topic, sensor_msgs.msg.Image)
        self.cache_rgb = message_filters.Cache(sub_rgb, cache_size=5, allow_headerless=True)
        self.object_class = rospy.get_param('~object_class', 'cup')
        self.confidence = rospy.get_param('~confidence', 0.5)
        self.flipped_img = rospy.get_param('~flipped', False)
        s = rospy.Service('lssn_detect', LssnDetect, self.handle_data)
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

        data_rgb = self.cache_rgb.cache_msgs[-1]
        image_rgb = ros_numpy.numpify(data_rgb)

        # # mean-std normalize the input image (batch-size: 1)
        now = rospy.Time.now()
        img = Image.fromarray(numpy.uint8(image_rgb)).resize((800, 600)).convert('RGB')
        img_tens = transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            output = model(img_tens)

        pred_logits = output['pred_logits'].softmax(-1)[0, :, :-1]
        pred_boxes = output['pred_boxes'][0]

        max_output = pred_logits.max(-1)

        m_output_inds = (max_output.values > self.confidence).nonzero(as_tuple=True)[0]

        pred_logits = pred_logits[m_output_inds]
        pred_boxes = pred_boxes[m_output_inds]
        pred_logits.shape

        labels = []

        for logits, box in zip(pred_logits, pred_boxes):
            cls = logits.argmax()
            if cls >= len(CLASSES):
                continue
            label = CLASSES[cls]
            labels.append(label)

        end = rospy.Time.now()
        torch.cuda.empty_cache()
        print(labels)
        if "cup" in labels:
            torch.cuda.empty_cache()
            return LssnDetectResponse(True)
        else:
            return LssnDetectResponse(False)


if __name__ == "__main__":
    lssn_detect_server = lssn_detect()