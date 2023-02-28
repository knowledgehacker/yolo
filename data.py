# -*- coding: utf-8 -*-

from numpy.random import permutation as perm
import cv2
from copy import deepcopy
import math
import numpy as np
import os

import config
from utils.misc import current_time
from utils.im_transform import imcv2_affine_trans#, imcv2_recolor


def _fix(obj, dims, scale, offs):
    for i in range(1, 5):
        dim = dims[(i + 1) % 2]
        off = offs[(i + 1) % 2]
        obj[i] = int(obj[i] * scale - off)
        obj[i] = max(min(obj[i], dim), 0)


def resize_input(im):
    h, w, c = config.IMG_H, config.IMG_W, config.IMG_CH
    imsz = cv2.resize(im, (w, h))
    imsz = imsz / 255.
    imsz = imsz[:, :, ::-1] # cv read in BGR, convert it to in RGB

    return imsz


"""
Takes an image, return it as a numpy tensor that is readily to be fed into tfnet.
The image will be transformed with random noise to augment training data,
using scale, translation, flipping and recolor. The accompanied
parsed annotation (allobj) will also be modified accordingly.
"""
def data_augment(im, allobj=None):
    if allobj is not None: # in training mode
        result = imcv2_affine_trans(im)
        im, dims, trans_param = result
        scale, offs, flip = trans_param
        for obj in allobj:
            _fix(obj, dims, scale, offs)
            if not flip:
                continue
            obj_1_ = obj[1]
            obj[1] = dims[0] - obj[3]
            obj[3] = dims[0] - obj_1_
        #im = imcv2_recolor(im)

    return im


H, W = config.H, config.W
B = config.B
C, labels = config.C, config.CLASSES


def batch(image_dir, chunks, test=False):
    image_batch = []

    class_probs_batch = []
    #class_mask_batch = []
    conf_batch = []
    coords_batch = []
    box_mask_batch = []

    for chunk in chunks:
        # preprocess
        jpg = chunk[0]
        w, h, allobj_ = chunk[1]
        allobj = deepcopy(allobj_)
        path = os.path.join(image_dir, jpg)
        if not os.path.exists(path):
            print("Warning - image %s doesn't exists." % path)
            return None, None

        img = cv2.imread(path)
        if not test:
            img = data_augment(img, allobj)
        img = resize_input(img)

        # Calculate placeholders' values
        class_probs = np.zeros([H*W, B, C])
        #class_mask = np.zeros([H*W, B, C])
        conf = np.zeros([H*W, B])
        coords = np.zeros([H*W, B, 4])
        box_mask = np.zeros([H*W, B])

        # Calculate regression target, normalize the items in the loss formula
        grid_w = 1. * w / W
        grid_h = 1. * h / H

        for obj in allobj:
            # centrex = 1/2 * (xmin + xmax), centrey = 1/2 * (ymin + ymax)
            centerx = .5 * (obj[1] + obj[3]) #xmin, xmax
            centery = .5 * (obj[2] + obj[4]) #ymin, ymax
            cx = centerx / grid_w
            cy = centery / grid_h
            if cx >= W or cy >= H:
                print("Warning - image %s has bad coordinate!" % path)
                return None, None
            grid_cx = int(cx)
            grid_cy = int(cy)
            grid_cell = grid_cy * W + grid_cx

            # you should handle obj[3]/[4] first, then obj[1]/[2], since obj[1]/[2] is used in obj[3]/[4]
            obj[3] = float(obj[3] - obj[1]) / grid_w
            obj[4] = float(obj[4] - obj[2]) / grid_h
            obj[1] = cx - grid_cx
            obj[2] = cy - grid_cy

            # Calculate placeholders' values
            class_probs[grid_cell, :, :] = [[0.] * C] * B
            class_probs[grid_cell, :, labels.index(obj[0])] = 1.
            #class_mask[grid_cell, :, :] = [[1.] * C] * B
            conf[grid_cell, :] = [1.] * B
            # coords[grid_cell, :, :] = [obj[1:5]] * B

            anchors = np.reshape(config.anchors, [B, 2])
            best_iou, best_anchor = find_best_anchor(obj, anchors)
            if best_iou > 0:
                # normalize width, height with anchors here
                anchor_w, anchor_h = anchors[best_anchor]
                obj[3] = np.log(obj[3] / anchor_w)
                obj[4] = np.log(obj[4] / anchor_h)
                coords[grid_cell, best_anchor, :] = obj[1:5]
                box_mask[grid_cell, best_anchor] = 1.

        image_batch.append(img)

        class_probs_batch.append(class_probs)
        #class_mask_batch.append(class_mask)
        conf_batch.append(conf)
        coords_batch.append(coords)
        box_mask_batch.append(box_mask)

    inp_feed_val = np.array(image_batch)
    loss_feed_val = {
        'class_probs': np.array(class_probs_batch),
        #'class_mask': np.array(class_mask_batch),
        'conf': np.array(conf_batch),
        'coords': np.array(coords_batch),
        'box_mask': np.array(box_mask_batch)
    }

    return inp_feed_val, loss_feed_val


def find_best_anchor(obj, anchors):
    best_iou = 0
    best_anchor = 0
    for k, anchor in enumerate(anchors):
        # Find IOU between box shifted to origin and anchor box.
        box_maxes = np.array([obj[3] / 2., obj[4] / 2.])
        box_mins = -box_maxes
        anchor_maxes = anchor / 2.
        anchor_mins = -anchor_maxes

        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[0] * intersect_wh[1]
        box_area = obj[3] * obj[4]
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
