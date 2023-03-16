# coding=utf-8

import numpy as np

import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

H, W = config.H, config.W
B = config.B
C = config.C


def create_cell_xy():
    # use some broadcast tricks to get the mesh coordinates
    h, w = config.H, config.W

    grid_x = tf.range(w, dtype=tf.int32)
    grid_y = tf.range(h, dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(grid_x, (-1, 1))
    y_offset = tf.reshape(grid_y, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.cast(tf.reshape(x_y_offset, [h, w, 1, 2]), tf.float32)

    return x_y_offset


def restore_coord(coord, cell_xy, anchors):
    grid_wh = np.reshape([config.IMG_W / W, config.IMG_H / H], [1, 1, 1, 2])

    coord_xy = coord[..., 0:2] + cell_xy
    coord_xy = coord_xy * grid_wh

    coord_wh = tf.clip_by_value(tf.exp(coord[..., 2:4]), 1e-9, 1e9) * anchors
    coord_wh = coord_wh * grid_wh

    orig_coord = tf.concat([coord_xy, coord_wh], -1)

    return orig_coord


def normalize_coord(coord, cell_xy, anchors):
    grid_wh = np.reshape([config.IMG_W / W, config.IMG_H / H], [1, 1, 1, 2])

    coord_xy = coord[..., 0:2]
    coord_xy = (coord_xy / grid_wh) - cell_xy

    coord_wh = coord[..., 2:4]
    coord_wh = (coord_wh / grid_wh) / anchors
    #coord_wh = tf.where(condition=tf.equal(coord_wh, 0), x=tf.ones_like(coord_wh), y=coord_wh)
    coord_wh = tf.log(tf.clip_by_value(coord_wh, 1e-9, 1e9))

    normalized_coord = tf.concat([coord_xy, coord_wh], -1)

    return normalized_coord


def cal_iou(pred_boxes, valid_true_boxes):
    """
    param:
    pred_boxes: [13, 13, 3, 4], (center_x, center_y, w, h)
    valid_true_boxes: [V, 4]
    """

    # [H, W, B, 2]
    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    # shape: [B, W, B, 1, 2]
    pred_box_xy = tf.expand_dims(pred_box_xy, -2)
    pred_box_wh = tf.expand_dims(pred_box_wh, -2)

    # [V, 2]
    true_box_xy = valid_true_boxes[:, 0:2]
    true_box_wh = valid_true_boxes[:, 2:4]

    # [H, W, B, 1, 2] & [V, 2] ==> [H, W, B, V, 2]
    intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                true_box_xy - true_box_wh / 2.)
    intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                true_box_xy + true_box_wh / 2.)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [H, W, B, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [H, W, B, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # shape: [V]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    # shape: [1, V]
    true_box_area = tf.expand_dims(true_box_area, axis=0)

    # [H, W, B, V]
    iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

    return iou
