# coding=utf-8

import config

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

B = config.B
C = config.C


def create_xy_offset(grid_size):
    # use some broadcast tricks to get the mesh coordinates
    grid_x = tf.range(grid_size[1], dtype=tf.int32)
    grid_y = tf.range(grid_size[0], dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(grid_x, (-1, 1))
    y_offset = tf.reshape(grid_y, (-1, 1))
    xy_offset = tf.concat([x_offset, y_offset], axis=-1)
    # shape: [13, 13, 1, 2]
    xy_offset = tf.cast(tf.reshape(xy_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

    return xy_offset


def restore_coord(feature_map, anchors, ratio):
    # NOTE: size in [h, w] format, anchor is in [w, h] format! don't get messed up!
    grid_size = tf.shape(feature_map)[1:3]  # [13, 13]

    feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], B, 4 + 1 + C])
    # split the feature_map along the last dimension
    box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, C], axis=-1)

    x_y_offset = create_xy_offset(grid_size)
    box_centers = tf.nn.sigmoid(box_centers)
    box_centers = box_centers + x_y_offset
    box_centers = box_centers * ratio[::-1]

    rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]
    print("--- rescaled_anchors")
    print(rescaled_anchors)
    box_sizes = tf.exp(box_sizes) * rescaled_anchors
    box_sizes = box_sizes * ratio[::-1]

    boxes = tf.concat([box_centers, box_sizes], axis=-1)

    return x_y_offset, boxes, conf_logits, prob_logits


def cal_iou(pred_boxes, valid_true_boxes):
    '''
    param:
        pred_boxes: [13, 13, 3, 4], (center_x, center_y, w, h)
        valid_true: [V, 4]
    '''

    # [13, 13, 3, 2]
    pred_box_xy = pred_boxes[..., 0:2]
    pred_box_wh = pred_boxes[..., 2:4]

    # shape: [13, 13, 3, 1, 2]
    pred_box_xy = tf.expand_dims(pred_box_xy, -2)
    pred_box_wh = tf.expand_dims(pred_box_wh, -2)

    # [V, 2]
    true_box_xy = valid_true_boxes[:, 0:2]
    true_box_wh = valid_true_boxes[:, 2:4]

    # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
    intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2., true_box_xy - true_box_wh / 2.)
    intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2., true_box_xy + true_box_wh / 2.)
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

    # shape: [13, 13, 3, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # shape: [13, 13, 3, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    # shape: [V]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
    # shape: [1, V]
    true_box_area = tf.expand_dims(true_box_area, axis=0)

    # [13, 13, 3, V]
    iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

    return iou
