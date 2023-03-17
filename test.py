# -*- coding: utf-8 -*-

import numpy as np

import config
from utils.voc_parser import parse
from data import get_batch_num, batch
from predict import gpu_nms, predict, draw_detection_on_image#, save_detection_as_json
from utils.misc import current_time, with_prefix, load_model

"""
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
"""

import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()


cfg = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
cfg.gpu_options.allow_growth = True


def test():
    print(current_time(), "Testing starts ...")

    with tf.device("/cpu:0"):
        print(current_time(), "Parse starts ...")
        data = parse(config.ANNOTATION_TEST_DIR, config.CLASSES)
        print(current_time(), "Parse finished!")

    outputs = ["feature_map_0", "feature_map_1", "feature_map_2", "loss"]
    g = load_model(config.MODLE_DIR, config.MODEL_NAME, outputs)
    with tf.Session(graph=g, config=cfg) as sess:
        #load_ckpt_model(sess, config.CKPT_DIR)

        # get prediction and other dependent tensors from the graph in the trained model for inference
        """
        for operation in g.get_operations():
            print(operation.name)
        """

        image_ph = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "image_ph:0"))
        box_ph_dict = {
            "y_true_13": g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "y_true_13_ph:0")),
            "y_true_26": g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "y_true_26_ph:0")),
            "y_true_52": g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "y_true_52_ph:0"))
        }

        # get feature maps and loss
        feature_map_0_op = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "feature_map_0:0"))
        feature_map_1_op = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "feature_map_1:0"))
        feature_map_2_op = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "feature_map_2:0"))
        feature_maps_op = [feature_map_0_op, feature_map_1_op, feature_map_2_op]
        boxes_op, confs_op, probs_op = predict(feature_maps_op)
        loss_op = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "loss:0"))

        batch_size = config.TEST_BATCH_SIZE
        batch_num, last_batch_size = get_batch_num(data, batch_size)
        print("batch_num: %d, batch_size: %d, last_batch_size: %d" % (batch_num, batch_size, last_batch_size))
        for b in range(batch_num):
            step = b + 1

            if (step == batch_num) and (last_batch_size > 0):
                batch_size = last_batch_size

            # get data
            print(current_time(), "batch %d get data starts ..." % step)
            chunks = [data[idx] for idx in range(b * batch_size, (b + 1) * batch_size)]
            images, box_dict = batch(config.IMAGE_TEST_DIR, chunks, test=True)
            if images is None:
                print(current_time(), "batch %d skipped!" % step)
                continue

            if config.DEVICE_TYPE == "gpu":
                images = np.transpose(images, [0, 3, 1, 2])

            # forward
            #print(current_time(), "batch %d forward starts ..." % step)
            feed_dict = dict()
            feed_dict[image_ph] = images
            for key in box_ph_dict:
                feed_dict[box_ph_dict[key]] = box_dict[key]

            boxes_batch, confs_batch, probs_batch, loss = sess.run([boxes_op, confs_op, probs_op, loss_op], feed_dict=feed_dict)

            # predict
            print(current_time(), "batch %d predict starts ..." % step)
            for i in range(len(chunks)):
                chunk = chunks[i]
                boxes, confs, probs = boxes_batch[i], confs_batch[i], probs_batch[i]
                image_file = "%s/%s" % (config.IMAGE_TEST_DIR, chunk[0])
                gpu_nms_op = gpu_nms(boxes, confs * probs, config.C, 50, config.THRESHOLD, config.IOU_THRESHOLD)
                boxes_pred, scores_pred, labels_pred = sess.run(gpu_nms_op)
                # !!!Important, we can not exchange the call below, since draw_detection_on_image changes the raw images
                # save_detection_as_json(image_file, [boxes_pred, scores_pred, labels_pred])
                draw_detection_on_image(image_file, [boxes_pred, scores_pred, labels_pred], chunk[1])

            # print loss message
            print(current_time(), "step %d, loss: %.3f" % (step, loss))

    print(current_time(), "Testing finished!")


if __name__ == "__main__":
    test()
