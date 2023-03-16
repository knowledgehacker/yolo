# coding=utf-8

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

from keras.models import Model
from keras.layers import Input

import config
from net.darknet53 import DarkNet53
from utils.layer import conv2d, yolo_block, upsample_layer, concatenate
from utils.box import restore_coord, cal_iou

B = config.B
C = config.C


class FastYolo(object):
    def __init__(self):
        # print("FastYolo")
        self.net = DarkNet53()

    def forward(self, image_batch, input_shape, data_format, is_training=False, reuse=False):
        self.img_size = tf.shape(image_batch)[1:3]
        if data_format == 'channels_first':
            self.img_size = tf.shape(image_batch)[2:4]

        input_image = Input(shape=input_shape, name="input_image")

        with tf.variable_scope('darknet53_body'):
            route_1, route_2, route_3, darknet53_out = self.net.build(input_image, data_format, trainable=is_training)
            pretrained_model = Model(inputs=input_image, outputs=darknet53_out)

        with tf.variable_scope('yolov3_head'):
            feature_map_out_size = B * (4 + 1 + C)

            inter1, net = yolo_block(route_3, 512, data_format, trainable=is_training)
            feature_map_1 = conv2d(net, feature_map_out_size, 1, 1, data_format, trainable=is_training)
            if data_format == 'channels_first':
                feature_map_1 = tf.transpose(feature_map_1, [0, 2, 3, 1])
            feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

            inter1 = conv2d(inter1, 256, 1, 1, data_format, trainable=is_training)
            #inter1 = upsample_layer(inter1, route_2.get_shape().as_list() if self.use_static_shape else tf.shape(route_2))
            inter1 = upsample_layer(inter1, 2, data_format)
            concat1 = concatenate(inter1, route_2, data_format)

            inter2, net = yolo_block(concat1, 256, data_format, trainable=is_training)
            feature_map_2 = conv2d(net, feature_map_out_size, 1, 1, data_format, trainable=is_training)
            if data_format == 'channels_first':
                feature_map_2 = tf.transpose(feature_map_2, [0, 2, 3, 1])
            feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

            inter2 = conv2d(inter2, 128, 1, 1, data_format, trainable=is_training)
            #inter2 = upsample_layer(inter2, route_1.get_shape().as_list() if self.use_static_shape else tf.shape(route_1))
            inter2 = upsample_layer(inter2, 2, data_format)
            concat2 = concatenate(inter2, route_1, data_format)

            _, net = yolo_block(concat2, 128, data_format, trainable=is_training)
            feature_map_3 = conv2d(net, feature_map_out_size, 1, 1, data_format, trainable=is_training)
            if data_format == 'channels_first':
                feature_map_3 = tf.transpose(feature_map_3, [0, 2, 3, 1])
            feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            model = Model(inputs=input_image, outputs=[feature_map_1, feature_map_2, feature_map_3])
            feature_maps = model.call(image_batch)

        return feature_maps, pretrained_model

    def cal_loss(self, feature_map_i, y_true, anchors):
        '''
        calc loss function from a certain scale
        input:
            feature_map_i: feature maps of a certain scale. shape: [N, 13, 13, 3*(5 + num_class)] etc.
            y_true: y_ture from a certain scale. shape: [N, 13, 13, 3, 5 + num_class + 1] etc.
            anchors: shape [9, 2]
        '''

        batch_size = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)

        # pred_boxes = {xy = (sigmoid(txy) + xy_offset) * ratio, wh = (exp(twh) * anchor)}
        xy_offset, pred_boxes, pred_conf_logits, pred_prob_logits = restore_coord(feature_map_i, anchors, ratio)

        # shape: take 416x416 input image and 13*13 feature_map for example:
        object_mask = y_true[..., 4:5]
        ignore_mask = self.cal_ignore_mask(batch_size, y_true[..., 0:4], pred_boxes, object_mask)

        # sigmoid(tx/ty)
        true_xy = y_true[..., 0:2] / ratio[::-1] - xy_offset
        pred_xy = pred_boxes[..., 0:2] / ratio[::-1] - xy_offset

        # tw/th
        true_wh = y_true[..., 2:4] / anchors
        pred_wh = pred_boxes[..., 2:4] / anchors
        """
        # for numerical stability
        true_wh = tf.where(condition=tf.equal(true_wh, 0), x=tf.ones_like(true_wh), y=true_wh)
        pred_wh = tf.where(condition=tf.equal(pred_wh, 0), x=tf.ones_like(pred_wh), y=pred_wh)
        """
        true_wh = tf.log(tf.clip_by_value(true_wh, 1e-9, 1e9))
        pred_wh = tf.log(tf.clip_by_value(pred_wh, 1e-9, 1e9))

        # box size punishment: box with smaller area has bigger weight. from the yolo darknet C source code.
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (
                    y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        """ loss_part """
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale) / batch_size
        wh_loss = tf.reduce_sum(tf.square(true_wh - pred_wh) * object_mask * box_loss_scale) / batch_size

        # shape: [N, 13, 13, 3, 1]
        conf_bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        # TODO: may need to balance the pos-neg by multiplying some weights
        conf_loss = object_mask * conf_bce + (1. - object_mask) * ignore_mask * conf_bce
        if config.USE_FOCAL_LOSS:
            # TODO: alpha should be a mask array if needed
            focal_mask = config.ALPHA * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), config.GAMMA)
            conf_loss *= focal_mask
        conf_loss = tf.reduce_sum(conf_loss) / batch_size

        # whether to use label smooth
        if config.USE_LABEL_SMOOTH:
            delta = 0.01
            label_target = (1 - delta) * y_true[..., 5:] + delta * 1. / C
        else:
            label_target = y_true[..., 5:]
        class_bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target, logits=pred_prob_logits)
        class_loss = object_mask * class_bce
        class_loss = tf.reduce_sum(class_loss) / batch_size

        return xy_loss, wh_loss, conf_loss, class_loss

    # calculation of ignore mask, reference to https://github.com/pjreddie/darknet/blob/master/src/yolo_layer.c#L179
    def cal_ignore_mask(self, batch_size, coord_pred, coord_gt, positive):
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

    def opt(self, y_pred, y_true):
        '''
        param:
            y_pred: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
            y_true: input ground true boxes
        '''
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [config.anchors[6:9], config.anchors[3:6], config.anchors[0:3]]

        # calc loss in 3 scales
        for i in range(len(y_pred)):
            result = self.cal_loss(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        total_loss = tf.identity(total_loss, name='loss')

        return total_loss, loss_xy, loss_wh, loss_conf, loss_class
