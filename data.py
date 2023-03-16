# -*- coding: utf-8 -*-

from numpy.random import permutation as perm
import cv2
import numpy as np
import os

import config
from utils.misc import current_time
#from utils.data_aug import random_color_distort
from utils.data_aug import random_expand, random_flip, random_crop_with_constraints, resize_with_bbox


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


RESIZE_H, RESIZE_W = config.IMG_H, config.IMG_W
B = config.B
C, classes = config.C, config.CLASSES


def batch(image_dir, chunks, test=False):
    image_batch = []

    y_true_13_batch = []
    y_true_26_batch = []
    y_true_52_batch = []

    for chunk in chunks:
        # preprocess
        jpg = chunk[0]
        img_w, img_h, labels, objs_ = chunk[1]
        objs = np.asarray(objs_)
        path = os.path.join(image_dir, jpg)
        if not os.path.exists(path):
            print("Warning - image %s doesn't exists." % path)
            return None, None

        img = cv2.imread(path)
        interp = 1
        if not test:
            img, objs = data_aug(img, objs, RESIZE_W, RESIZE_H)
            interp = np.random.randint(0, 5)
        img, objs = resize_with_bbox(img, objs, RESIZE_W, RESIZE_H, interp=interp, letterbox=config.KEEP_ASPECT_RATIO)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = img / 255.

        # https://github.com/wizyoung/YOLOv3_TensorFlow/blob/8776cf7b2531cae83f5fc730f3c70ae97919bfd6/utils/data_utils.py#L51
        y_true_13 = np.zeros((13, 13, B, 4 + 1 + C))
        y_true_26 = np.zeros((26, 26, B, 4 + 1 + C))
        y_true_52 = np.zeros((52, 52, B, 4 + 1 + C))

        y_true = [y_true_13, y_true_26, y_true_52]

        box_centers = (objs[:, 0:2] + objs[:, 2:4]) / 2.
        box_sizes = objs[:, 2:4] - objs[:, 0:2]

        ratio_dict = {0: 8., 1: 16., 2.: 32.}
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        anchors = np.reshape(config.anchors, [B*3, 2])
        best_match_anchors = find_best_match_anchors(box_sizes, anchors)
        for i, anchor in enumerate(best_match_anchors):
            # anchor: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 0
            feature_map_group = 2 - anchor // 3
            ratio = ratio_dict[feature_map_group]
            x = int(np.floor(box_centers[i, 0] / ratio))
            y = int(np.floor(box_centers[i, 1] / ratio))
            k = anchors_mask[feature_map_group].index(anchor)
            c = classes.index(labels[i])
            # print(feature_map_group, '|', y,x,k,c)

            y_true[feature_map_group][y, x, k, 0:2] = box_centers[i]
            y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
            y_true[feature_map_group][y, x, k, 4] = 1.
            y_true[feature_map_group][y, x, k, 5+c] = 1.
            """
            y_true[feature_map_group][y, x, k, 4:5] = 1.
            y_true[feature_map_group][y, x, k, 5+c:5+c+1] = 1.
            """

        # collect regression items
        image_batch.append(img)

        y_true_13_batch.append(y_true_13)
        y_true_26_batch.append(y_true_26)
        y_true_52_batch.append(y_true_52)

    inp_feed_val = np.array(image_batch)
    loss_feed_val = {
        'y_true_13': np.array(y_true_13_batch),
        'y_true_26': np.array(y_true_26_batch),
        'y_true_52': np.array(y_true_52_batch)
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

