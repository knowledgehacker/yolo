# -*- coding: utf-8 -*-

from numpy.random import permutation as perm
import cv2
from copy import deepcopy
import math
import numpy as np
import os

import config
from utils.misc import current_time
from utils.im_transform import imcv2_affine_trans, imcv2_recolor


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
        # TODO: distort with linear instead of exponential
        im = imcv2_recolor(im)

    return im


H, W = config.H, config.W
B = config.B
C, labels = config.C, config.CLASSES


def batch(image_dir, chunks, test=False):
# def _batch(image_dir, chunks):
    image_batch = []

    probs_batch = []
    proids_batch = []
    confs_batch = []
    coord_batch = []

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
        if config.VERSION == "v1":
            probs = np.zeros([H*W, C])
            proid = np.zeros([H*W, C])
        elif config.VERSION == "v2":
            probs = np.zeros([H*W, B, C])
            proid = np.zeros([H*W, B, C])
        else:
            print("Unsupported version: %s" % config.VERSION)
            exit(-1)
        confs = np.zeros([H*W, B])
        coord = np.zeros([H*W, B, 4])

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

            obj[3] = float(obj[3] - obj[1]) / w
            obj[4] = float(obj[4] - obj[2]) / h
            obj[3] = math.sqrt(obj[3])
            obj[4] = math.sqrt(obj[4])
            grid_cx = int(cx)
            grid_cy = int(cy)
            obj[1] = cx - grid_cx  # centerx
            obj[2] = cy - grid_cy  # centery

            grid_cell = grid_cy * W + grid_cx

            # Calculate placeholders' values
            if config.VERSION == "v1":
                probs[grid_cell, :] = [0.] * C
                probs[grid_cell, labels.index(obj[0])] = 1.
                proid[grid_cell, :] = [1.] * C
            elif config.VERSION == "v2":
                probs[grid_cell, :, :] = [[0.] * C] * B
                probs[grid_cell, :, labels.index(obj[0])] = 1.
                proid[grid_cell, :, :] = [[1.] * C] * B
            else:
                print("Unsupported version: %s" % config.VERSION)
                exit(-1)
            confs[grid_cell, :] = [1.] * B
            coord[grid_cell, :, :] = [obj[1:5]] * B

        image_batch.append(img)

        probs_batch.append(probs)
        proids_batch.append(proid)
        confs_batch.append(confs)
        coord_batch.append(coord)

    inp_feed_val = np.array(image_batch)
    loss_feed_val = {
        'class_probs': np.array(probs_batch),
        'class_proids': np.array(probs_batch),
        'object_proids': np.array(confs_batch),
        'coords': np.array(coord_batch)
    }

    return inp_feed_val, loss_feed_val


def get_batch_num(data, batch_size):
    size = len(data)

    if batch_size > size:
        batch_size = size
    batch_num = int(size / batch_size)

    return batch_num

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
