# -*- coding: utf-8 -*-

from numpy.random import permutation as perm
import cv2
import math
import numpy as np
import os

import config
from utils.misc import current_time
#from utils.data_aug import random_color_distort
from utils.data_aug import random_expand, random_flip, random_crop_with_constraints, resize_with_bbox, letterbox_resize


# reference to function parse_data in https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/utils/data_utils.py
def data_aug(img, boxes, resize_w, resize_h):
    # random color jittering
    # NOTE: applying color distort may lead to bad performance sometimes
    #img = random_color_distort(img)

    # random expansion with prob 0.5
    if np.random.uniform(0, 1) > 0.5:
        img, boxes = random_expand(img, boxes, 4)

    # random cropping
    h, w, _ = img.shape
    boxes, crop = random_crop_with_constraints(boxes, (w, h))
    x0, y0, w, h = crop
    img = img[y0: y0 + h, x0: x0 + w]

    # move resize outside, to be able to experiment on training with data augmentation turned off
    """
    # resize with random interpolation
    h, w, _ = img.shape
    interp = np.random.randint(0, 5)
    img, boxes = resize_with_bbox(img, boxes, resize_w, resize_h, interp=interp, letterbox=letterbox_resize)
    """

    # random horizontal flip
    h, w, _ = img.shape
    img, boxes = random_flip(img, boxes, px=0.5)

    return img, boxes


H, W = config.H, config.W
B = config.B
C, classes = config.C, config.CLASSES


def batch(image_dir, chunks, test=False):
    image_batch = []

    cls_batch = []
    conf_batch = []
    coord_batch = []

    for chunk in chunks:
        # preprocess
        jpg = chunk[0]
        img_w, img_h, labels, objs_ = chunk[1]
        objs = np.asarray(objs_)
        path = os.path.join(image_dir, jpg)
        if not os.path.exists(path):
            print("Warning - image %s doesn't exists." % path)
            return None, None

        resize_w, resize_h = config.IMG_W, config.IMG_H

        img = cv2.imread(path)
        interp = 1
        if not test:
            img, objs = data_aug(img, objs, resize_w, resize_h)
            interp = np.random.randint(0, 5)
        img, objs = resize_with_bbox(img, objs, resize_w, resize_h, interp=interp, letterbox=config.KEEP_ASPECT_RATIO)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = img / 255.

        # Calculate placeholders' values
        # https://github.com/wizyoung/YOLOv3_TensorFlow/blob/8776cf7b2531cae83f5fc730f3c70ae97919bfd6/utils/data_utils.py#L51
        cls = np.zeros([H, W, B, C])
        conf = np.zeros([H, W, B, 1])
        coord = np.zeros([H, W, B, 4])

        # Use (resize_w, resize_h) not (image_w, image_h) here, otherwise the image and boxes in it will be inconsistent
        grid_wh = np.array([1. * resize_w / W, 1. * resize_h / H])

        # normalized box center x/y coordinates and width/height to in grid unit
        box_centers = ((objs[:, 0:2] + objs[:, 2:4]) / 2.) / grid_wh
        box_sizes = (objs[:, 2:4] - objs[:, 0:2]) / grid_wh
        anchors = np.reshape(config.anchors, [B, 2])
        best_match_anchors = find_best_match_anchors(box_sizes, anchors)
        for i in range(len(best_match_anchors)):
            anchor_index = best_match_anchors[i]

            cx, cy = box_centers[i]
            grid_cx, grid_cy = int(np.floor(cx)), int(np.floor(cy))
            cls[grid_cy, grid_cx, anchor_index, classes.index(labels[i])] = 1.
            conf[grid_cy, grid_cx, anchor_index, 0] = 1.
            coord[grid_cy, grid_cx, anchor_index, 0:2] = [cx - grid_cx, cy - grid_cy]
            coord[grid_cy, grid_cx, anchor_index, 2:4] = np.log(clip_by_value(box_sizes[i] / anchors[anchor_index], 1e-9, 1e9))

        # collect regression items
        image_batch.append(img)

        cls_batch.append(cls)
        conf_batch.append(conf)
        coord_batch.append(coord)

    inp_feed_val = np.array(image_batch)
    loss_feed_val = {
        'cls': np.array(cls_batch),
        'conf': np.array(conf_batch),
        'coord': np.array(coord_batch)
    }

    return inp_feed_val, loss_feed_val


def clip_by_value(value, lb, ub):
    return np.minimum(np.maximum(value, lb), ub)


def find_best_match_anchors(box_sizes, anchors):
    # [N, 1, 2]
    box_sizes = np.expand_dims(box_sizes, 1)
    # broadcast tricks
    # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
    intersect_mins = np.maximum(- box_sizes / 2., - anchors / 2.)
    intersect_maxs = np.minimum(box_sizes / 2., anchors / 2.)
    intersect_whs = np.maximum(intersect_maxs - intersect_mins, 0.)
    intersect_areas = intersect_whs[:, :, 0] * intersect_whs[:, :, 1]

    box_areas = box_sizes[:, :, 0] * box_sizes[:, :, 1]

    anchor_areas = anchors[:, 0] * anchors[:, 1]

    ious = intersect_areas / (box_areas + anchor_areas - intersect_areas + 1e-9)

    best_anchors = np.argmax(ious, axis=1)

    return best_anchors


def get_batch_num(data, batch_size):
    last_batch_size = 0

    size = len(data)
    batch_num = int(size / batch_size)
    processed_size = batch_num * batch_size
    if processed_size < size:
        batch_num += 1
        last_batch_size = size - processed_size

    return batch_num, last_batch_size

