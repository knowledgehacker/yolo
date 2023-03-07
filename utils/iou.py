
import numpy as np

import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def _extract_coords(coords, cell_coords, anchors):
    # unit: grid cell, w = resize_w / grid_w, h = resize_h / grid_h.
    wh = tf.exp(coords[:, :, :, :, 2:4]) * anchors
    area = wh[:, :, :, :, 0] * wh[:, :, :, :, 1]  # unit: grid cell^2
    xy_centre = coords[:, :, :, :, 0:2] + cell_coords  # [batch, W, H, B, 2]
    left_top = xy_centre - (wh * 0.5)  # [batch, W, H, B, 2]
    right_bottom = xy_centre + (wh * 0.5)  # [batch, W, H, B, 2]

    return left_top, right_bottom, area


def _calc_intersects(left_top_1, right_bottom_1, left_top_2, right_bottom_2):
    left_top_intersect = tf.maximum(left_top_1, left_top_2)
    right_bottom_intersect = tf.minimum(right_bottom_1, right_bottom_2)
    wh_intersect = tf.maximum(right_bottom_intersect - left_top_intersect, 0.)
    area_intersect = wh_intersect[:, :, :, :, 0] * wh_intersect[:, :, :, :, 1]

    return left_top_intersect, right_bottom_intersect, area_intersect


def cal_iou(coords_predict, coords_true, cell_coords, anchors):
    left_top_predict, right_bottom_predict, area_predict = _extract_coords(coords_predict, cell_coords, anchors)
    left_top_true, right_bottom_true, area_true = _extract_coords(coords_true, cell_coords, anchors)
    _, _, area_intersect = _calc_intersects(left_top_predict, right_bottom_predict,
                                            left_top_true, right_bottom_true)

    #iou = tf.truediv(area_intersect, area_predict + area_true - area_intersect)
    iou = area_intersect / (area_predict + area_true - area_intersect)

    return iou

