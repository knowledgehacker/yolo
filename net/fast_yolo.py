# -*- coding: utf-8 -*-

import numpy as np

import config
from net.small import Small

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import Model
from keras.layers import Input

H, W = config.H, config.W
B = config.B
C = config.C


class FastYolo(object):
    def __init__(self):
        #print("FastYolo")
        self.net = Small()

    def forward(self, image_batch, input_shape, data_format, dropout_keep_prob, trainable=True):
        input_image = Input(shape=input_shape, name="input_image")
        output, pretrained_model, _ = self.net.build(input_image, data_format, dropout_keep_prob, trainable)

        model = Model(input_image, output)
        #model.summary()

        net_out = model.call(image_batch)
        if data_format == 'channels_first':
            net_out = tf.transpose(net_out, [0, 2, 3, 1])
        net_out = tf.identity(net_out, name="net_out")

        return net_out, pretrained_model

    """
    reference to loss_layer in wizyoung/YOLOv3_TensorFlow/model.py
    https://github.com/wizyoung/YOLOv3_TensorFlow/blob/8776cf7b2531cae83f5fc730f3c70ae97919bfd6/model.py#L192
    """
    def opt(self, net_out, nd_cls, nd_conf, nd_coord):
        # parameters
        class_scale = config.class_scale
        obj_scale = config.object_scale
        noobj_scale = config.noobject_scale
        coord_scale = config.coord_scale
        print('scales  = {}'.format([class_scale, obj_scale, noobj_scale, coord_scale]))

        batch_size = tf.cast(tf.shape(net_out)[0], tf.float32)

        # shape of [H, W, B, 2] instead of [?, H, W, B, 2]
        anchors = np.reshape(config.anchors, [1, 1, B, 2])
        # shape of [H, W, B, 2] instead of [?, H, W, B, 2]
        cell_xy = tf.concat(B * [create_cell_xy()], 2)

        # handle net output feature map
        nd_net_out = tf.reshape(net_out, [-1, H, W, B, (C + 1 + 4)])

        cls_pred, conf_pred, coord_pred = tf.split(nd_net_out, [C, 1, 4], axis=-1)

        # coord_pred[:, :, :, :, 0:2] is (t_x, t_y)
        coord_pred_xy = tf.sigmoid(coord_pred[:, :, :, :, 0:2])
        # adjusted_coord[:, :, :, :, 2:4] is (t_w, t_h)
        coord_pred_wh = coord_pred[:, :, :, :, 2:4]
        coord_pred = tf.concat([coord_pred_xy, coord_pred_wh], -1)

        """
        confidence part, some samples are neither positive nor negative ones
        (the boxes with highest iou >= threshold, but not responsible for detection on any ground truth object).
        """
        # positive samples(bounding boxes with the highest iou)' ground truth confidence is iou
        positive = nd_conf

        # negative samples(bounding boxes with highest iou < threshold)' ground truth confidence is 0
        orig_coord_pred = restore_coord(coord_pred, cell_xy, anchors)
        ignore_mask = cal_ignore_mask(batch_size, orig_coord_pred, nd_coord, positive)
        negative = ignore_mask * (1. - positive)

        bce_conf = tf.nn.sigmoid_cross_entropy_with_logits(labels=nd_conf, logits=conf_pred)
        conf_obj_loss = positive * bce_conf
        conf_noobj_loss = negative * bce_conf
        conf_loss = obj_scale * conf_obj_loss + noobj_scale * conf_noobj_loss
        # Whether to apply focal loss on the conf loss
        if config.USE_FOCAL_LOSS:
            focal_mask = config.ALPHA * tf.pow(tf.abs(positive - tf.sigmoid(conf_pred)), config.GAMMA)
            conf_loss *= focal_mask
        conf_loss = tf.reduce_sum(conf_loss) / batch_size

        """
        class part, multiply positive to get the positive samples.
        """
        # whether to use label smooth
        if config.USE_LABEL_SMOOTH:
            delta = 0.01
            label_target = (1 - delta) * nd_cls + delta * 1. / C
        else:
            label_target = nd_cls
        bce_class = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target, logits=cls_pred)
        class_loss = class_scale * tf.reduce_sum(positive * bce_class) / batch_size

        """
        coordinate part, multiply positive to get the positive samples.
        """
        # punish the box size, the bigger the box is, the smaller its weight is. the trick to improve detection on small objects???
        # note: nd_coord[:, :, :, :, 2:3] is of shape [?, H, W, B, 1], while nd_coord[:, :, :, :, 2] is of shape [?, H, W, B]
        coord_ratio = 2 - (nd_coord[:, :, :, :, 2:3] / config.IMG_W) * (nd_coord[:, :, :, :, 3:4] / config.IMG_H)

        se_coord = (coord_pred - normalize_coord(nd_coord, cell_xy, anchors)) ** 2
        coord_loss = coord_scale * tf.reduce_sum(se_coord * positive * coord_ratio) / batch_size

        # total loss
        total_loss = conf_loss + class_loss + coord_loss
        loss_op = tf.identity(total_loss, name="loss")

        return loss_op


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


def box_iou(pred_boxes, valid_true_boxes):
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


def cal_ignore_mask(batch_size, coord_pred, coord_gt, positive):
    # the calculation of ignore mask if referred from
    # https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c#L179
    ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

    def loop_cond(idx, ignore_mask):
        return tf.less(idx, tf.cast(batch_size, tf.int32))

    def loop_body(idx, ignore_mask):
        # shape: [H, W, B, 4] & [H, W, B]  ==>  [V, 4]
        # V: num of true gt box of each image in a batch
        valid_true_boxes = tf.boolean_mask(coord_gt[idx], tf.cast(positive[idx, ..., 0], 'bool'))
        # shape: [H, W, B, 4] & [V, 4] ==> [H, W, B, V]
        iou = box_iou(coord_pred[idx], valid_true_boxes)
        # shape: [H, W, B]
        best_iou = tf.reduce_max(iou, axis=-1)
        # shape: [H, W, B]
        ignore_mask_tmp = tf.cast(best_iou < config.IOU_THRESHOLD, tf.float32)
        # finally will be shape: [?, H, W, B]
        ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
        return idx + 1, ignore_mask

    _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
    ignore_mask = ignore_mask.stack()
    # shape: [?, H, W, B, 1]
    ignore_mask = tf.expand_dims(ignore_mask, -1)

    return ignore_mask
