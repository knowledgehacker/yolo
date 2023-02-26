# -*- coding: utf-8 -*-

import numpy as np

import config
from utils.voc_parser import parse
from data import get_batch_num, batch
from predict import postprocess, draw_detection_on_image, save_detection_as_json
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

    outputs = ["net_out", "loss"]
    g = load_model(config.MODLE_DIR, config.MODEL_NAME, outputs)
    with tf.Session(graph=g, config=cfg) as sess:
        #load_ckpt_model(sess, config.CKPT_DIR)

        # get prediction and other dependent tensors from the graph in the trained model for inference
        """
        for operation in g.get_operations():
            print(operation.name)
        """

        image_ph = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "image_ph:0"))
        bounding_box_ph_dict = {
            "class_probs": g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "class_probs_ph:0")),
            #"class_mask": g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "class_mask_ph:0")),
            "conf": g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "conf_ph:0")),
            "coords": g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "coords_ph:0")),
            "box_mask": g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "box_mask_ph:0"))
        }
        #dropout_keep_prob_ph = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "dropout_keep_prob_ph:0"))

        # get net_out and loss
        net_out_op = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "net_out:0"))
        loss_op = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "loss:0"))

        batch_size, batch_num = get_batch_num(data, config.TEST_BATCH_SIZE)
        print("batch_size: %d, batch_num: %d" % (batch_size, batch_num))
        for b in range(batch_num):
            step = b + 1

            # get data
            print(current_time(), "batch %d get data starts ..." % step)
            chunks = [data[idx] for idx in range(b * batch_size, (b + 1) * batch_size)]
            images, bounding_box_dict = batch(config.IMAGE_TEST_DIR, chunks, test=True)
            if images is None:
                print(current_time(), "batch %d skipped!" % step)
                continue

            if config.DEVICE_TYPE == "gpu":
                images = np.transpose(images, [0, 3, 1, 2])

            # forward
            #print(current_time(), "batch %d forward starts ..." % step)
            feed_dict = dict()
            feed_dict[image_ph] = images
            for key in bounding_box_ph_dict:
                feed_dict[bounding_box_ph_dict[key]] = bounding_box_dict[key]
            #feed_dict[dropout_keep_prob_ph] = config.TEST_KEEP_PROB

            net_outs, loss = sess.run([net_out_op, loss_op], feed_dict=feed_dict)
            #print("--- net_outs.shape")
            #print(net_outs.shape)

            # predict
            print(current_time(), "batch %d predict starts ..." % step)
            """
            image_fnames = [chunk[0] for chunk in chunks]
            predict(config.IMAGE_TEST_DIR, image_fnames, net_outs)
            """
            with tf.device("/cpu:0"):
                for chunk, net_out in zip(chunks, net_outs):
                    image_file = "%s/%s" % (config.IMAGE_TEST_DIR, chunk[0])
                    # !!!Important, we can not exchange the call below, since draw_detection_on_image changes the raw images
                    #save_detection_as_json(image_file, net_out)
                    draw_detection_on_image(image_file, net_out)

            # print loss message
            print(current_time(), "step %d, loss: %.3f" % (step, loss))

    print(current_time(), "Testing finished!")


"""
def predict(image_dir, image_fnames, net_outs):
    for (image_fname, net_out) in zip(image_fnames, net_outs):
        image_file = "%s/%s" % (image_dir, image_fname)
        #print("image_file: %s" % image_file)
        draw_detection_on_image(image_file, net_out)
"""

if __name__ == "__main__":
    test()
