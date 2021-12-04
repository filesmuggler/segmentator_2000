#!/usr/bin/env python3

from segmentator_2000.srv import LssnRgbSeg, LssnRgbSegResponse
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
from detectron2.data import MetadataCatalog
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

model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
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


class LssnRgbSegServer:
    def __init__(self):
        '''Initialize ros service'''
        rospy.init_node('lssn_rgb_seg_server')
        #TODO: parametrize below
        rgb_topic = "/xtion/rgb/image_raw"
        depth_topic = "/xtion/depth_registered/image_raw"
        camera_topic = "/xtion/depth_registered/camera_info"
        sub_rgb = message_filters.Subscriber(rgb_topic, sensor_msgs.msg.Image)
        self.cache_rgb = message_filters.Cache(sub_rgb, cache_size=5, allow_headerless=True)

        sub_depth = message_filters.Subscriber(depth_topic, sensor_msgs.msg.Image)
        self.cache_depth = message_filters.Cache(sub_depth, cache_size=5, allow_headerless=True)

        sub_info = message_filters.Subscriber(camera_topic, sensor_msgs.msg.CameraInfo)
        self.cache_info = message_filters.Cache(sub_info, cache_size=5, allow_headerless=True)

        s = rospy.Service('lssn_rgb_seg', LssnRgbSeg, self.handle_data)
        rospy.loginfo("Lssn Rgb Seg ready to handle data.")
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
        im = Image.fromarray(numpy.uint8(image_rgb))
        ## Image keeps original image resolution
        original_im_shape = im.size

        # plot.imshow(im)
        # plot.axis('off')
        # plot.show()

        # # mean-std normalize the input image (batch-size: 1)
        with torch.no_grad():
            #img = transform(im).unsqueeze(0).cuda()
            img = transform(im).unsqueeze(0)
            out = model(img)

        # the post-processor expects as input the target size of the predictions (which we set here to the image size)
        result = postprocessor(out, torch.as_tensor(img.shape[-2:]).unsqueeze(0))[0]

        # The segmentation is stored in a special-format png
        panoptic_seg = Image.open(io.BytesIO(result['png_string']))

        panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()

        # We retrieve the ids corresponding to each mask
        panoptic_seg_id = rgb2id(panoptic_seg)

        segments_info = deepcopy(result["segments_info"])

        meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")

        id_lst = []

        d = {}

        for i in range(len(segments_info)):
            c = segments_info[i]["category_id"]
            meta_c = ""
            if segments_info[i]["isthing"]:
                segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[c]
                meta_c = meta.thing_classes[segments_info[i]["category_id"]]
            else:
                segments_info[i]["category_id"] = meta.stuff_dataset_id_to_contiguous_id[c]
                meta_c = meta.stuff_classes[segments_info[i]["category_id"]]

            if meta_c in user_cls:
                d[i] = meta_c
                id_lst.append(i)

        mask_list = []
        class_list = []

        # # Finally we color each mask individually
        panoptic_seg[:, :, :] = 0
        for id in range(panoptic_seg_id.max() + 1):
            if id in id_lst:
                segment = deepcopy(panoptic_seg)

                segment[panoptic_seg_id == id] = 255
                segment = resize(segment, (original_im_shape[1], original_im_shape[0]))

                segment = segment * 255
                segment[segment > 127] = 255
                segment[segment < 127] = 0
                #segment[segment > 127] = 255
                class_list.append(d[id])
                # plot.imshow(segment)
                # plot.axis('off')
                # plot.show()

                segment = segment.astype(numpy.uint8)
                # plot.imshow(segment)
                # plot.axis('off')
                # plot.show()

                msg = ros_numpy.msgify(sensor_msgs.msg.Image, segment, encoding='8UC3')

                mask_list.append(msg)
                rospy.loginfo("Adding mask")
        print(class_list)
        rospy.loginfo("Returning data")
        #torch.cuda.empty_cache()
        return LssnRgbSegResponse(data_info, data_rgb, data_depth, class_list, mask_list)


if __name__ == "__main__":
    # ip_rgb_seg_server()
    lssn_rgb_seg_server = LssnRgbSegServer()
