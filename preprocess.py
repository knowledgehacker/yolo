# -*- coding: utf-8 -*-

import os
import numpy as np
from math import sqrt
from multiprocessing.pool import ThreadPool

import config
from utils.parser import parse_annotation, resize
from utils.misc import current_time, load_image_indexes

import matplotlib.pyplot as plt

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

S, B, C = config.S, config.B, config.C
SS = S * S

class_to_index = dict(zip(config.CLASSES, range(C)))


"""
an image is divided into config.S x config.S grids.
if the centre of a bounding box is in a grid, we say the grid hits the object in the bounding box.
the grid is responsible for prediction of the class of the hit object, 
and predicts config.B bounding boxes(including x, y, w, h and confidence for (whether) hit an object) for the target.
"""


def build_dataset(input, output):
    print(current_time(), "Build dataset: %s starts ..." % input)

    image_index2names = load_image_indexes("%s-%s.txt" % (config.IMAGE_INDEX_FILE, "train"))
    image_files = ["%s/%s" % (input, name) for index, name in image_index2names]

    pool = ThreadPool(processes=4)

    image_num = len(image_files)
    batch_num = int(image_num / config.BATCH_SIZE)
    if batch_num * config.BATCH_SIZE != image_num:
        batch_num += 1
    print("image_num: %d, batch_num: %d" % (image_num, batch_num))

    batch_sample_nums = []
    for batch_idx in range(batch_num):
        start_idx = batch_idx * config.BATCH_SIZE
        end_idx = min((batch_idx + 1) * config.BATCH_SIZE, image_num)
        image_file_batch = image_files[start_idx:end_idx]
        batch_sample_num = pool.apply(generate_batch, (batch_idx, image_file_batch, output))
        batch_sample_nums.append(batch_sample_num)

    pool.close()
    pool.join()

    total_batch_sample_num = sum(batch_sample_nums)
    print("Total %d samples generated!" % total_batch_sample_num)

    print(current_time(), "Build dataset: %s finished!" % input)


def generate_batch(batch_idx, image_file_batch, output):
    print(current_time(), "Batch %d starts ..." % batch_idx)

    batch_sample_num = 0

    writer = tf.python_io.TFRecordWriter("%s/image-%03d.tfrecords" % (output, batch_idx))
    for i in range(len(image_file_batch)):
        # image.shape: (h, w, c)
        image_file = image_file_batch[i]
        image_name = os.path.basename(image_file)

        class_names, recs = get_objs(image_name)
        if len(class_names) == 0:
            continue

        probs = np.zeros((SS, C))
        proids = np.zeros((SS, C))
        confs = np.zeros((SS, B))
        coords = np.zeros((SS, B, 4))

        image = plt.imread(image_file)
        resized_image, resized_recs = resize(image, recs, config.IMG_W, config.IMG_H)
        for class_name, resized_rec in zip(class_names, resized_recs):
            x, y, w, h = resized_rec.corner_to_centre()

            grid_w, grid_h = float(config.IMG_W / S), float(config.IMG_H / S)
            grid = int(y / grid_h) * S + int(x / grid_w)
            """
            print("--- resized_rec")
            print("left: %f, top: %f, right: %f, bottom: %f"
                  % (resized_rec.left, resized_rec.top, resized_rec.right, resized_rec.bottom))
            print("x: %f, y: %f, w: %f, h: %f" % (x, y, w, h))
            print("grid_x: %d, grid_y: %d, grid: %d" % (int(x / grid_w), int(y / grid_h), grid))
            """

            probs[grid, class_to_index[class_name]] = 1.0
            proids[grid, :] = [1.0] * C
            confs[grid, :] = [1.0] * B  # for ground truth bounding box, replicate B copies
            """
            --- normalization
            (x, y): divided by grid size, then relative to the grid the object in.
            (w, h): sqrt(w / config.IMG_W), sqrt(h / config.IMG_H)
            """
            norm_x, norm_y = x / grid_w, y / grid_h
            norm_x, norm_y = norm_x - int(norm_x), norm_y - int(norm_y)
            norm_w, norm_h = sqrt(float(w / config.IMG_W)), float(sqrt(h / config.IMG_H))
            coords[grid, :, :] = [[norm_x, norm_y, norm_w, norm_h]] * B
            """
            for i in range(B):
                confs[grid, i] = 1.0
                coords[grid, i, 0], coords[grid, i, 1] = norm_x, norm_y
                coords[grid, i, 2], coords[grid, i, 3] = norm_w, norm_h
            """

            """
            print("--- probs")
            print(probs)
            print("--- confs")
            print(confs)
            print("--- coords")
            print(coords)
            """

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image_idx': int64_feature([batch_idx * config.BATCH_SIZE + i]),
                'content': float_feature(resized_image.flatten()),
                'probs': float_feature(probs.flatten()),
                'proids': float_feature(proids.flatten()),
                'confs': float_feature(confs.flatten()),
                'coords': float_feature(coords.flatten())
            }))
        writer.write(example.SerializeToString())

        batch_sample_num += 1
    writer.close()

    print(current_time(), "Batch %d: %d samples generated!" % (batch_idx, batch_sample_num))

    return batch_sample_num


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_objs(image_name):
    class_names = []
    recs = []
    annotation_file = "%s/%s.xml" % (config.ANNOT_DIR, image_name[:image_name.rindex('.')])
    # print("--- annotation_file: %s" % annotation_file)
    obj_infos = parse_annotation(annotation_file)
    for obj_info in obj_infos:
        class_name, difficult, rec, image_shape = obj_info
        if difficult != 1:
            class_names.append(class_name)
            recs.append(rec)
            """
            print("--- rec")
            print("left: %f, top: %f, right: %f, bottom: %f" % (rec.left, rec.top, rec.right, rec.bottom))
            """

    return class_names, recs


build_dataset(config.IMAGE_TRAIN_DIR, config.TF_IMAGE_TRAIN_DIR)
#build_dataset(config.IMAGE_TEST_DIR, config.TF_IMAGE_TEST_DIR)
