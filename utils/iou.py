
import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


def _extract_coords(coords):
    # w, h is normalized to sqrt(w/config.IMG_W), sqrt(h/config.IMG_W) upon preprocess, we should restore w, h to calculate area
    # grid_w = config.IMG_W / config.S, grid_h = config.IMG_H / config.S
    wh = coords[:, :, :, 2:4] ** 2 * config.S  # unit: grid cell
    area = wh[:, :, :, 0] * wh[:, :, :, 1]  # unit: grid cell^2
    xy_centre = coords[:, :, :, 0:2]  # [batch, SS, B, 2]
    left_top = xy_centre - (wh * 0.5)  # [batch, SS, B, 2]
    right_bottom = xy_centre + (wh * 0.5)  # [batch, SS, B, 2]

    return left_top, right_bottom, area


def _calc_intersects(left_top_1, right_bottom_1, left_top_2, right_bottom_2):
    left_top_intersect = tf.maximum(left_top_1, left_top_2)
    right_bottom_intersect = tf.minimum(right_bottom_1, right_bottom_2)
    wh_intersect = right_bottom_intersect - left_top_intersect
    wh_intersect = tf.maximum(wh_intersect, 0.0)
    area_intersect = wh_intersect[:, :, :, 0] * wh_intersect[:, :, :, 1]

    return left_top_intersect, right_bottom_intersect, area_intersect


def _find_best_box_iou(area_1, area_2, area_intersect):
    #iou = tf.truediv(area_intersect, area_1 + area_2 - area_intersect)
    iou = area_intersect / (area_1 + area_2 - area_intersect)
    best_box_iou = tf.to_float(tf.equal(iou, tf.reduce_max(iou, [2], True)))

    return best_box_iou


def calc_best_box_iou(coords_predict, coords_true):
    left_top_predict, right_bottom_predict, area_predict = _extract_coords(coords_predict)
    left_top_true, right_bottom_true, area_true = _extract_coords(coords_true)
    _, _, area_intersect = _calc_intersects(left_top_predict, right_bottom_predict,
                                            left_top_true, right_bottom_true)
    best_box_iou = _find_best_box_iou(area_predict, area_true, area_intersect)

    return best_box_iou
