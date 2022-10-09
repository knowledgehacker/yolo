# -*- coding: utf-8 -*-

import os
import numpy as np
from math import sqrt

import config
from utils.parser import parse_annotation, resize
from utils.misc import current_time, load_files

import matplotlib.pyplot as plt

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
def build_dataset(input):
    print(current_time(), "build dataset: %s starts..." % input)

    sample_num = 0

    image_name_list = []
    resized_image_list = []
    probs_list = []
    proids_list = []
    confs_list = []
    coords_list = []

    image_files = load_files(input)
    for image_file in image_files:
        # image.shape is of format (h, w, c)
        image_name = os.path.basename(image_file)
        image_name_list.append(image_name)

        image = plt.imread(image_file)

        class_names = []
        recs = []
        annotation_file = "%s/%s.xml" % (config.ANNOT_DIR, image_name[:image_name.rindex('.')])
        #print("--- annotation_file: %s" % annotation_file)
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

        if len(class_names) == 0:
            continue

        probs = np.zeros((SS, C))
        proids = np.zeros((SS, C))
        confs = np.zeros((SS, B))
        coords = np.zeros((SS, B, 4))

        resized_image, resized_recs = resize(image, recs, config.IMG_W, config.IMG_H)
        for i in range(len(resized_recs)):
            resized_rec = resized_recs[i]
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

            probs[grid, class_to_index[class_names[i]]] = 1.0
            proids[grid, :] = [1] * C

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

        # TODO: does unflatten ones work?
        """
        resized_image_list.append(resized_image.flatten())
        probs_list.append(probs.flatten())
        confs_list.append(confs.flatten())
        coords_list.append(coords.flatten())
        """
        resized_image_list.append(resized_image)
        probs_list.append(probs)
        proids_list.append(proids)
        confs_list.append(confs)
        coords_list.append(coords)

        sample_num += 1

    print("sample_num=%d" % sample_num)

    print(current_time(), "build dataset: %s finishes..." % input)

    image_name_array = np.asarray(image_name_list)
    resized_image_array = np.asarray(resized_image_list)
    probs_array = np.asarray(probs_list)
    proids_array = np.asarray(proids_list)
    confs_array = np.asarray(confs_list)
    coords_array = np.asarray(coords_list)

    return image_name_array, resized_image_array, probs_array, proids_array, confs_array, coords_array


"""
resized_images, probs, confs, coords = build_dataset(config.IMAGE_DIR)
print("--- image")
print(resized_images[0])
"""
