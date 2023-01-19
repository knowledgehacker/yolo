# -*- coding: utf-8 -*-

import config
from input_feed import create_dataset
from predict import postprocess
from utils.misc import current_time, load_image_indexes, with_prefix, load_model

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

    # TODO: test images in tmp directory
    index_file = "%s-%s.txt" % (config.IMAGE_INDEX_FILE, "tmp")
    print("index_file: %s" % index_file)
    image_index2names = load_image_indexes(index_file)
    print("--- image_index2names")
    print(image_index2names)

    outputs = ["net_out", "loss"]
    g = load_model(config.MODLE_DIR, config.MODEL_NAME, outputs)
    with tf.Session(graph=g, config=cfg) as sess:
        #load_ckpt_model(sess, config.CKPT_DIR)

        # get prediction and other dependent tensors from the graph in the trained model for inference
        """
        for operation in g.get_operations():
            print(operation.name)
        """

        content_ph = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "content_ph:0"))
        #image_idx_ph, \
        class_probs_ph, class_proids_ph, object_proids_ph, coords_ph = (
            #g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "image_idx_ph:0")),
            g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "class_probs_ph:0")),
            g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "class_proids_ph:0")),
            g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "object_proids_ph:0")),
            g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "coords_ph:0"))
        )
        dropout_keep_prob_ph = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "dropout_keep_prob:0"))

        net_out_op = g.get_tensor_by_name(with_prefix(config.MODEL_NAME, "net_out:0"))

        # create iterator for test dataset
        #handle_ph = g.get_tensor_by_name("handle_ph:0")
        # TODO: test images in tmp directory
        test_dataset = create_dataset(config.IMAGE_TMP_DIR, config.TF_IMAGE_TMP_FILE, image_index2names, test=True)
        test_iterator = test_dataset.make_initializable_iterator()
        content, (image_idx, probs, proids, confs, coords) = test_iterator.get_next("next_batch")
        if config.DEVICE_TYPE == "gpu":
            content = tf.transpose(content, [0, 3, 1, 2])

        #test_handle = sess.run(test_iterator.string_handle())

        # important!!! Don't call 'tf.global_variables_initializer().run()' when doing inference using trained model
        #tf.global_variables_initializer().run()
        sess.run(test_iterator.initializer)

        try:
            while True:
                # we don't need to feed test_handle to handle_ph here
                content_ts, image_idx_ts, class_probs_ts, class_proids_ts, object_proids_ts, coords_ts = sess.run([content, image_idx, probs, proids, confs, coords])
                net_out_ts = sess.run([net_out_op], feed_dict={#image_idx_ph: image_idx_ts,
                                                               content_ph: content_ts,
                                                               class_probs_ph: class_probs_ts,
                                                               class_proids_ph: class_proids_ts,
                                                               object_proids_ph: object_proids_ts,
                                                               coords_ph: coords_ts,
                                                               dropout_keep_prob_ph: config.TEST_KEEP_PROB})
                print("--- net_out_ts")
                print(net_out_ts)
                write_preds(image_idx_ts, net_out_ts, dict(image_index2names))
        except tf.errors.OutOfRangeError:
            pass

    print(current_time(), "Testing finished!")


def load_ckpt_model(sess, ckpt_dir):
    ckpt_file = tf.train.latest_checkpoint(ckpt_dir)
    print("ckpt_file: %s" % ckpt_file)
    saver = tf.train.import_meta_graph("{}.meta".format(ckpt_file))
    saver.restore(sess, ckpt_file)


def write_preds(image_idx_ts, net_out_ts, image_idx_to_name):
    for (image_idx, net_out) in zip(image_idx_ts, net_out_ts[0]):
        # TODO: test images in tmp directory
        image_file = "%s/%s" % (config.IMAGE_TMP_DIR, image_idx_to_name[image_idx])
        #image_file = "%s/%s" % (config.IMAGE_TEST_DIR, image_idx_to_name[image_idx])
        print("image_idx: %d, image_file: %s" % (image_idx, image_file))
        postprocess(image_file, net_out)


if __name__ == "__main__":
    test()
