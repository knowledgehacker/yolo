# -*- coding: utf-8 -*-

import os
import numpy as np
from math import sqrt

import config
from utils.parser import parse_annotation
from utils.misc import current_time, load_files

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

S, B, C = config.S, config.B, config.C
SS = S * S

class_to_index = dict(zip(config.CLASSES, range(config.C)))


"""
an image is divided into config.S x config.S grids.
if the centre of a bounding box is in a grid, we say the grid hits the object in the bounding box.
the grid is responsible for prediction of the class of the hit object, 
and predicts config.B bounding boxes(including x, y, w, h and confidence for (whether) hit an object) for the target.
"""


def build_dataset(image_dir, annotation_dir, output, data_type):
    print(current_time(), "Build %s dataset: %s starts ..." % (data_type, image_dir))

    sample_num = 0
    bad_sample_num = 0

    writer = tf.python_io.TFRecordWriter(output)

    fout = open("%s-%s.txt" % (config.IMAGE_INDEX_FILE, data_type), 'w')

    # load image files
    image_files = load_files(image_dir)
    image_file_names = [image_file[image_file.rindex("/") + 1:] for image_file in image_files]
    for i in range(len(image_file_names)):
        # image.shape is of format (h, w, c)
        image_file_name = image_file_names[i]
        image_name = image_file_name[:image_file_name.rindex('.')]
        image_w, image_h, class_names, recs = get_objs(annotation_dir, image_name)
        if len(class_names) == 0:
            print(image_file_name)
            bad_sample_num += 1
            continue

        fout.write("%s\t%d\n" % (image_file_name, i))

        probs = np.zeros((SS, C))
        proids = np.zeros((SS, C))
        confs = np.zeros((SS, B))
        coords = np.zeros((SS, B, 4))

        for class_name, rec in zip(class_names, recs):
            # a bounding box's coordinates
            x, y, w, h = rec.corner_to_centre()

            # grid size is calculated using original instead of resized image's size
            grid_w, grid_h = float(image_w / S), float(image_h / S)
            grid_x, grid_y = int(x / grid_w), int(y / grid_h)
            grid = grid_y * S + grid_x

            probs[grid, class_to_index[class_name]] = 1.0
            proids[grid, :] = [1.0] * C

            # for a ground truth bounding box, replicate B copies
            confs[grid, :] = [1.0] * B

            """
            normalization as follows:
            (x, y): divided by grid size, then relative to the object grid.
            (w, h): sqrt(w / image_w), sqrt(h / image_h)
            """
            norm_x, norm_y = x / grid_w, y / grid_h
            norm_x, norm_y = norm_x - np.floor(norm_x), norm_y - np.floor(norm_y)
            norm_w, norm_h = sqrt(float(w / image_w)), float(sqrt(h / image_h))
            coords[grid, :, :] = [[norm_x, norm_y, norm_w, norm_h]] * B

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image_idx': int64_feature([i]),
                'probs': float_feature(probs.flatten()),
                'proids': float_feature(proids.flatten()),
                'confs': float_feature(confs.flatten()),
                'coords': float_feature(coords.flatten())
            }))
        writer.write(example.SerializeToString())

        sample_num += 1

        if sample_num % config.BATCH_SIZE == 0:
            print(current_time(), "batch %d: %s samples generated!" % (int(sample_num / config.BATCH_SIZE), sample_num))

    fout.close()

    writer.close()

    if sample_num % config.BATCH_SIZE != 0:
        print("batch %d: %s samples generated!" % (int(i / config.BATCH_SIZE + 1), sample_num))

    print("sample_num: %d, bad_sample_num: %d" % (sample_num, bad_sample_num))

    print(current_time(), "Build %s dataset: %s finished!" % (data_type, image_dir))

    return sample_num


def get_objs(annotation_dir, image_name):
    class_names = []
    recs = []
    annotation_file = "%s/%s.xml" % (annotation_dir, image_name)
    w, h, obj_infos = parse_annotation(annotation_file)
    for obj_info in obj_infos:
        class_name, difficult, rec = obj_info
        if difficult != 1:
            class_names.append(class_name)
            recs.append(rec)

    return w, h, class_names, recs


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


train_sample_num = build_dataset(config.IMAGE_TRAIN_DIR, config.ANNOTATION_TRAIN_DIR, config.TF_IMAGE_TRAIN_FILE, "train")
print("train_sample_num: %d" % train_sample_num)
test_sample_num = build_dataset(config.IMAGE_TEST_DIR, config.ANNOTATION_TEST_DIR, config.TF_IMAGE_TEST_FILE, "test")
print("test_sample_num: %d" % test_sample_num)

