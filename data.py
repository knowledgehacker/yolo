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
    #cls_mask_batch = []
    conf_batch = []
    coord_batch = []
    box_mask_batch = []

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
        if not test:
            img, objs = data_aug(img, objs, resize_w, resize_h)
            img, objs = resize_with_bbox(img, objs, resize_w, resize_h, interp=np.random.randint(0, 5), letterbox=letterbox_resize)
        else:
            img, objs = resize_with_bbox(img, objs, resize_w, resize_h, interp=1, letterbox=letterbox_resize)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = img / 255.

        # Calculate placeholders' values
        cls = np.zeros([H, W, B, C])
        #cls_mask = np.zeros([H, W, B, C])
        conf = np.zeros([H, W, B, 1])
        coord = np.zeros([H, W, B, 4])
        box_mask = np.zeros([H, W, B])

        # Calculate regression target, normalize the items in the loss formula
        # Use (resize_w, resize_h) not (image_w, image_h) here, otherwise the image and boxes in it will be inconsistent
        grid_w = 1. * resize_w / W
        grid_h = 1. * resize_h / H

        for label, obj in zip(labels, objs):
            box = np.zeros([4])

            # centrex = 1/2 * (xmin + xmax), centrey = 1/2 * (ymin + ymax)
            centerx = .5 * (obj[0] + obj[2]) #xmin, xmax
            centery = .5 * (obj[1] + obj[3]) #ymin, ymax
            cx = centerx / grid_w
            cy = centery / grid_h
            if cx >= W or cy >= H:
                print("Warning - image %s has bad coordinate!" % path)
                return None, None
            grid_cx = int(cx)
            grid_cy = int(cy)

            box[2] = float(obj[2] - obj[0]) / grid_w
            box[3] = float(obj[3] - obj[1]) / grid_h
            box[0] = cx - grid_cx
            box[1] = cy - grid_cy

            # Calculate placeholders' values
            cls[grid_cy, grid_cx, :, :] = [[0.] * C] * B
            cls[grid_cy, grid_cx, :, classes.index(label)] = 1.
            #cls_mask[grid_cy, grid_cx, :, :] = [[1.] * C] * B
            conf[grid_cy, grid_cx, :] = [[1.]] * B
            anchors = np.reshape(config.anchors, [B, 2])
            coord[grid_cy, grid_cx, :, 0:2] = [box[0:2]] * B
            coord[grid_cy, grid_cx, :, 2:4] = np.log(clip_by_value([box[2:4]] * B / anchors, 1e-9, 1e9))

            best_iou, best_anchor = find_best_anchor(box, anchors)
            if best_iou > 0:
                box_mask[grid_cy, grid_cx, best_anchor] = 1.

        image_batch.append(img)

        cls_batch.append(cls)
        #cls_mask_batch.append(cls_mask)
        conf_batch.append(conf)
        coord_batch.append(coord)
        box_mask_batch.append(box_mask)

    inp_feed_val = np.array(image_batch)
    loss_feed_val = {
        'cls': np.array(cls_batch),
        #'cls_mask': np.array(cls_mask_batch),
        'conf': np.array(conf_batch),
        'coord': np.array(coord_batch),
        'box_mask': np.array(box_mask_batch)
    }

    return inp_feed_val, loss_feed_val


def clip_by_value(value, lb, ub):
    return np.minimum(np.maximum(value, lb), ub)


def find_best_anchor(obj, anchors):
    best_iou = 0
    best_anchor = 0
    for k, anchor in enumerate(anchors):
        # Find IOU between box shifted to origin and anchor box.
        box_maxes = obj[2:4] / 2.
        box_mins = -box_maxes
        anchor_maxes = anchor / 2.
        anchor_mins = -anchor_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[0] * intersect_wh[1]
        box_area = obj[2] * obj[3]
        anchor_area = anchor[0] * anchor[1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        if iou > best_iou:
            best_iou = iou
            best_anchor = k

    return best_iou, best_anchor


def get_batch_num(data, batch_size):
    size = len(data)

    if batch_size > size:
        batch_size = size
    batch_num = int(size / batch_size)

    return batch_size, batch_num

"""
def shuffle(image_dir, data):
    print(current_time(), "Shuffle starts ...")

    batch_size = config.BATCH_SIZE
    batch_num = get_batch_num(data, batch_size)

    shuffle_idx = perm(np.arange(len(data)))
    for b in range(batch_num):
        chunks = [data[i] for i in shuffle_idx[b * batch_size: (b+1) * batch_size]]
        x_batch, feed_batch = _batch(image_dir, chunks)
        if x_batch is None:
            print(current_time(), "batch %d skipped!" % (b+1))
            continue

        print(current_time(), "batch %d data ready!" % (b+1))

        yield x_batch, feed_batch

    print(current_time(), "Shuffle finished!")
"""
