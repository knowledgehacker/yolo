# -*- coding: utf-8 -*-

import numpy as np

import config

from net.darknet19 import DarkNet19
from utils.compose import ConvBatchLReLu, ConvBatchLReLu_loop, Conv, space_to_depth_x2, depth_concat
from utils.box import create_cell_xy, restore_coord, normalize_coord, cal_iou

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import Model
from keras.layers import Input, Lambda

H, W = config.H, config.W
B = config.B
C = config.C


class FastYolo(object):
    def __init__(self):
        #print("FastYolo")
        self.net = DarkNet19()

    def forward(self, image_batch, input_shape, data_format, dropout_keep_prob, trainable=True):
        input_image = Input(shape=input_shape, name="input_image")

        net_output = self.net.build(input_image, data_format, dropout_keep_prob, trainable)
        pretrained_model = Model(inputs=input_image, outputs=net_output)
        #pretrained_model.summary()

        layer_13 = pretrained_model.get_layer("relu_13")
        layer_13_output = layer_13.output

        layer_18 = pretrained_model.get_layer("relu_18")
        layer_18_output = layer_18.output

        # yolo head
        padding_mode = 'same'

        # Layer 19 - 20
        convs = [(1024, 3, 1), (1024, 3, 1)]
        x = ConvBatchLReLu_loop(layer_18_output, convs, padding_mode, data_format, 19, trainable) # (?, 1024, 13, 13) in NCHW?

        # reorg layer
        """
        reorg_layer reorganizes the output from conv13 as the shape of conv20.
        Then the second route layer will concat the two output together. Its function is mostly shape transform.

        tf.space_to_depth converts conv13(?, 512, 26, 26) to (?, 512 * block_size * block_size, 26 / block_size, 26 / block_size).
        to be able to concatenate with conv20(?, 1024, 13, 13), block_size = 2, that is what space_to_depth_x2 does.
        """
        passthrough = Lambda(space_to_depth_x2, arguments={"data_format": data_format, "name": "reorg"})(layer_13_output)
        #passthrough = space_to_depth_x2(layer_13_output, data_format, "reorg")  # (?, 2048, 13, 13) in NCHW?

        x = depth_concat([passthrough, x], data_format)

        # Layer 21
        x = ConvBatchLReLu(x, 1024, 3, 1, padding_mode, data_format, 21, trainable)

        # layer 22, use bias
        output = Conv(x, B * (C + 1 + 4), 1, 1, padding_mode, data_format, 22, trainable)

        model = Model(input_image, output)
        #model.summary()
        net_out = model.call(image_batch)
        if data_format == 'channels_first':
            net_out = tf.transpose(net_out, [0, 2, 3, 1])
        net_out = tf.identity(net_out, name="net_out")

        return net_out, pretrained_model

    def cal_ignore_mask(self, batch_size, coord_pred, coord_gt, positive):
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
            iou = cal_iou(coord_pred[idx], valid_true_boxes)
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
        ignore_mask = self.cal_ignore_mask(batch_size, orig_coord_pred, nd_coord, positive)
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
