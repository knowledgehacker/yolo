# -*- coding: utf-8 -*-

import os
import numpy as np
from math import sqrt

import config
from utils.parser import parse_annotation, resize
from utils.misc import current_time, load_files, load_image_indexes

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


def build_dataset(input, output, data_type):
    print(current_time(), "Build %s dataset: %s starts ..." % (data_type, input))

    sample_num = 0
    bad_sample_num = 0

    examples = []

    fout = open("%s-%s.txt" % (config.IMAGE_INDEX_FILE, data_type), 'w')

    # load image files
    image_files = load_files(input)
    image_file_names = [image_file[image_file.rindex("/") + 1:] for image_file in image_files]
    for i in range(len(image_file_names)):
        # image.shape is of format (h, w, c)
        image_file_name = image_file_names[i]
        image_name = image_file_name[:image_file_name.rindex('.')]
        #print(image_name)

        w, h, class_names, recs = get_objs(image_name)
        if len(class_names) == 0:
            print(image_file_name)
            bad_sample_num += 1
            continue

        fout.write("%s\t%d\n" % (image_file_name, i))

        probs = np.zeros((SS, C))
        proids = np.zeros((SS, C))
        confs = np.zeros((SS, B))
        coords = np.zeros((SS, B, 4))

        resized_recs = resize(recs, w, h, config.IMG_W, config.IMG_H)
        for class_name, resized_rec in zip(class_names, resized_recs):
            x, y, w, h = resized_rec.corner_to_centre()

            grid_w, grid_h = float(config.IMG_W / S), float(config.IMG_H / S)
            grid_x, grid_y = int(x / grid_w), int(y / grid_h)
            grid = grid_y * S + grid_x
            """
            print("--- resized_rec")
            print("left: %f, top: %f, right: %f, bottom: %f"
                  % (resized_rec.left, resized_rec.top, resized_rec.right, resized_rec.bottom))
            print("x: %f, y: %f, w: %f, h: %f" % (x, y, w, h))
            print("grid_x: %d, grid_y: %d, grid: %d" % (grid_x, grid_y, grid))
            """

            probs[grid, class_to_index[class_name]] = 1.0
            proids[grid, :] = [1.0] * C

            # TODO: for ground truth bounding box, replicate B copies???
            confs[grid, :] = [1.0] * B

            """
            --- normalization
            (x, y): divided by grid size, then relative to the object grid.
            (w, h): sqrt(w / config.IMG_W), sqrt(h / config.IMG_H)
            """
            norm_x, norm_y = x / grid_w, y / grid_h
            norm_x, norm_y = norm_x - np.floor(norm_x), norm_y - np.floor(norm_y)
            norm_w, norm_h = sqrt(float(w / config.IMG_W)), float(sqrt(h / config.IMG_H))
            coords[grid, :, :] = [[norm_x, norm_y, norm_w, norm_h]] * B

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
                'image_idx': int64_feature([i]),
                'probs': float_feature(probs.flatten()),
                'proids': float_feature(proids.flatten()),
                'confs': float_feature(confs.flatten()),
                'coords': float_feature(coords.flatten())
            }))
        examples.append(example)

        sample_num += 1
        """
        if sample_num % config.BATCH_SIZE == 0:
            print(current_time(), "batch %d: %s samples generated!" % (int(sample_num / config.BATCH_SIZE), sample_num))
        """

    fout.close()

    writer = tf.python_io.TFRecordWriter(output)
    for i in range(sample_num):
        writer.write(examples[i].SerializeToString())
    writer.close()
    """
    if sample_num % config.BATCH_SIZE != 0:
        print("batch %d: %s samples generated!" % (int(i / config.BATCH_SIZE + 1), sample_num))
    """
    print("sample_num: %d, bad_sample_num: %d" % (sample_num, bad_sample_num))

    print(current_time(), "Build %s dataset: %s finished!" % (data_type, input))

    return sample_num


def get_objs(image_name):
    class_names = []
    recs = []
    annotation_file = "%s/%s.xml" % (config.ANNOT_DIR, image_name)
    # print("--- annotation_file: %s" % annotation_file)
    w, h, obj_infos = parse_annotation(annotation_file)
    for obj_info in obj_infos:
        class_name, difficult, rec, image_shape = obj_info
        if difficult != 1:
            class_names.append(class_name)
            recs.append(rec)
            """
            print("--- rec")
            print("left: %f, top: %f, right: %f, bottom: %f" % (rec.left, rec.top, rec.right, rec.bottom))
            """

    return w, h, class_names, recs


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


train_sample_num = build_dataset(config.IMAGE_TRAIN_DIR, config.TF_IMAGE_TRAIN_FILE, "train")
print("train_sample_num: %d" % train_sample_num)
#test_sample_num = build_dataset(config.IMAGE_TEST_DIR, config.TF_IMAGE_TEST_FILE, "test")
#print("test_sample_num: %d" % test_sample_num)

