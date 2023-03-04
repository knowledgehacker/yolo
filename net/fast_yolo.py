# -*- coding: utf-8 -*-

import numpy as np

import config
from net.small import Small
from utils.iou import cal_iou, create_cell_xy

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

        net_out = tf.identity(model.call(image_batch), name="net_out")

        return net_out, pretrained_model

    """
    reference to loss_layer in wizyoung/YOLOv3_TensorFlow/model.py
    https://github.com/wizyoung/YOLOv3_TensorFlow/blob/8776cf7b2531cae83f5fc730f3c70ae97919bfd6/model.py#L192
    """
    def opt(self, net_out, nd_cls, nd_conf, nd_coord, box_mask):
        # parameters
        class_scale = config.class_scale
        obj_scale = config.object_scale
        noobj_scale = config.noobject_scale
        coord_scale = config.coord_scale
        print('scales  = {}'.format([class_scale, obj_scale, noobj_scale, coord_scale]))

        batch_num = tf.cast(tf.shape(net_out)[0], tf.float32)

        anchors = np.reshape(config.anchors, [1, 1, 1, B, 2])
        # shape of [H, W, B, 2] instead of [?, H, W, B, 2]
        cell_xy = tf.concat(B * [create_cell_xy()], 2)

        # handle net output feature map
        nd_net_out = tf.reshape(net_out, [-1, H, W, B, (C + 1 + 4)])

        adjusted_cls, adjusted_conf, adjusted_coord = tf.split(nd_net_out, [C, 1, 4], axis=-1)

        # adjusted_coord[:, :, :, :, 0:2] is (t_x, t_y)
        adjusted_coord_xy = tf.sigmoid(adjusted_coord[:, :, :, :, 0:2])
        # adjusted_coord[:, :, :, :, 2:4] is (t_w, t_h)
        adjusted_coord_wh = tf.clip_by_value(adjusted_coord[:, :, :, :, 2:4], -9, 9)
        adjusted_coord = tf.concat([adjusted_coord_xy, adjusted_coord_wh], -1)

        """
        confidence part, some samples are neither positive nor negative ones
        (the boxes with highest iou >= threshold, but not responsible for detection on any ground truth object).
        """
        # positive samples(bounding boxes with the highest iou)' ground truth confidence is iou
        positive = tf.expand_dims(box_mask, -1)

        # negative samples(bounding boxes with highest iou < threshold)' ground truth confidence is 0
        iou = cal_iou(adjusted_coord, nd_coord, cell_xy, anchors)
        best_box = tf.equal(iou, tf.reduce_max(iou, axis=-1, keepdims=True))
        best_box_ge_thres = tf.equal(best_box, tf.greater_equal(iou, config.IOU_THRESHOLD))
        negative = (1.0 - tf.expand_dims(tf.to_float(best_box_ge_thres), -1)) * (1.0 - positive)

        bce_conf = tf.nn.sigmoid_cross_entropy_with_logits(logits=adjusted_conf, labels=nd_conf)
        conf_obj_loss = obj_scale * tf.reduce_sum(positive * bce_conf) / batch_num
        conf_noobj_loss = noobj_scale * tf.reduce_sum(negative * bce_conf) / batch_num
        conf_loss = conf_obj_loss + conf_noobj_loss

        """
        class part, multiply positive to get the positive samples.
        """
        cls_mask = tf.concat(C * [positive], -1)

        bce_class = tf.nn.sigmoid_cross_entropy_with_logits(logits=adjusted_cls, labels=nd_cls)
        class_loss = class_scale * tf.reduce_sum(cls_mask * bce_class) / batch_num

        """
        coordinate part, multiply positive to get the positive samples.
        use (b_x - c_x, b_y - c_y), (b_w / p_w, b_h / p_h) in coordinate loss, is it correct???
        I think so, (width, height) loss should be irrelevant to anchor size (width, height).
        """
        coord_mask = tf.concat(4 * [positive], -1)
        # the bigger the box is, the smaller its weight is. the trick to improve detection on small objects???
        true_wh = tf.exp(nd_coord[:, :, :, :, 2:4]) * anchors
        coord_ratio = 2 - (true_wh[:, :, :, :, 0] / W) * (true_wh[:, :, :, :, 1] / H)
        coord_ratio = tf.concat(4 * [tf.expand_dims(coord_ratio, -1)], -1)

        se_coord = (adjusted_coord - nd_coord) ** 2
        coord_loss = coord_scale * tf.reduce_sum(coord_ratio * coord_mask * se_coord) / batch_num

        # total loss
        loss_op = tf.add_n([conf_loss, class_loss, coord_loss], name="loss")

        return loss_op


def flat(x):
    return tf.layers.flatten(x)
