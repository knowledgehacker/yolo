# -*- coding: utf-8 -*-

from numpy.random import permutation as perm
import cv2
from copy import deepcopy
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


def preprocess(im, allobj=None):
    """
    Takes an image, return it as a numpy tensor that is readily to be fed into tfnet.
    If there is an accompanied annotation (allobj),
    meaning this preprocessing is serving the train process, then this
    image will be transformed with random noise to augment training data,
    using scale, translation, flipping and recolor. The accompanied
    parsed annotation (allobj) will also be modified accordingly.
    """
    if type(im) is not np.ndarray:
        im = cv2.imread(im)

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
        #im = imcv2_recolor(im)

    im = resize_input(im)
    if allobj is None:
        return im

    return im#, np.array(im) # for unit testing


S, B = config.S, config.B
C, labels = config.C, config.CLASSES


def batch(image_dir, chunks):
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

        img = preprocess(path, allobj)

        # Calculate regression target, normalize the items in the loss formula
        grid_w = 1. * w / S
        grid_h = 1. * h / S

        for obj in allobj:
            # centrex = 1/2 * (xmin + xmax), centrey = 1/2 * (ymin + ymax)
            centerx = .5 * (obj[1] + obj[3]) #xmin, xmax
            centery = .5 * (obj[2] + obj[4]) #ymin, ymax
            cx = centerx / grid_w
            cy = centery / grid_h
            if cx >= S or cy >= S:
                print("Warning - image %s has bad coordinate!" % path)
                return None, None

            obj[3] = float(obj[3] - obj[1]) / w
            obj[4] = float(obj[4] - obj[2]) / h
            obj[3] = np.sqrt(obj[3])
            obj[4] = np.sqrt(obj[4])
            obj[1] = cx - np.floor(cx)  # centerx
            obj[2] = cy - np.floor(cy)  # centery
            obj += [int(np.floor(cy) * S + np.floor(cx))]

        # show(im, allobj, S, w, h, cellx, celly) # unit test

        # Calculate placeholders' values
        probs = np.zeros([S*S, C])
        confs = np.zeros([S*S, B])
        coord = np.zeros([S*S, B, 4])
        proid = np.zeros([S*S, C])
        for obj in allobj:
            probs[obj[5], :] = [0.] * C
            probs[obj[5], labels.index(obj[0])] = 1.
            proid[obj[5], :] = [1] * C
            coord[obj[5], :, :] = [obj[1:5]] * B
            confs[obj[5], :] = [1.] * B

        image_batch.append(img)

        probs_batch.append(probs)
        proids_batch.append(proid)
        confs_batch.append(confs)
        coord_batch.append(coord)

    inp_feed_val = np.array(image_batch)
    #print("--- inp_feed_val.shape")
    #print(inp_feed_val.shape)
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