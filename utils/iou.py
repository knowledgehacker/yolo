
import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

S, B, C = config.S, config.B, config.C
SS = S * S


def _extract_coords(coords):
    # w, h is normalized to sqrt(w/S), sqrt(h/S) when preprocess, we should restore w, h to calculate area
    wh = tf.pow(coords[:, :, :, 2:4], 2) * S  # unit: grid cell
    area = wh[:, :, :, 0] * wh[:, :, :, 1]  # unit: grid cell^2
    xy_centre = coords[:, :, :, 0:2]  # [batch, SS, B, 2]
    top_left = xy_centre - (wh * 0.5)  # [batch, SS, B, 2]
    bottom_right = xy_centre + (wh * 0.5)  # [batch, SS, B, 2]

    return top_left, bottom_right, area


def _calc_intersects(top_left_1, bottom_right_1, top_left_2, bottom_right_2):
    top_left_intersect = tf.maximum(top_left_1, top_left_2)
    bottom_right_intersect = tf.minimum(bottom_right_1, bottom_right_2)
    wh_intersect = bottom_right_intersect - top_left_intersect
    wh_intersect = tf.maximum(wh_intersect, 0.0)
    # * is equivalent to tf.multiply
    area_intersect = wh_intersect[:, :, :, 0] * wh_intersect[:, :, :, 1]

    return top_left_intersect, bottom_right_intersect, area_intersect


def _find_best_box_iou(area_1, area_2, area_intersect):
    #iou = tf.truediv(area_intersect, area_1 + area_2 - area_intersect)
    iou = area_intersect / (area_1 + area_2 - area_intersect)
    best_box_iou = tf.to_float(tf.equal(iou, tf.reduce_max(iou, [2], True)))

    return best_box_iou


def calc_best_box_iou(coords_predict, coords_true):
    top_left_predict, bottom_right_predict, area_predict = _extract_coords(coords_predict)
    top_left_true, bottom_right_true, area_true = _extract_coords(coords_true)
    _, _, area_intersect = _calc_intersects(top_left_predict, bottom_right_predict,
                                            top_left_true, bottom_right_true)
    best_box_iou = _find_best_box_iou(area_predict, area_true, area_intersect)

    return best_box_iou
