# -*- coding: utf-8 -*-

import os

import numpy as np
import json
import cv2

import config
from cython_utils.cy_yolo3_findboxes import box_constructor


def process_box(b, h, w, threshold):
	max_indx = np.argmax(b.probs)
	max_prob = b.probs[max_indx]
	label = config.CLASSES[max_indx]
	if max_prob > threshold:
		left  = int ((b.x - b.w/2.) * w)
		right = int ((b.x + b.w/2.) * w)
		top   = int ((b.y - b.h/2.) * h)
		bot   = int ((b.y + b.h/2.) * h)
		if left  < 0    :  left = 0
		if right > w - 1: right = w - 1
		if top   < 0    :   top = 0
		if bot   > h - 1:   bot = h - 1
		mess = '{}'.format(label)
		return (left, right, top, bot, mess, max_indx, max_prob)
	return None


# net
def find_boxes(net_out):
	meta = {
		"anchors": config.anchors,

		"classes": config.C,
		"num": config.B,
		"out_size": (config.H, config.W, config.B * (config.C + 1 + 4))
	}
	boxes = box_constructor(meta, net_out, config.THRESHOLD, config.IOU_THRESHOLD)

	return boxes


"""
    def predict(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, config.anchors[6:9]),
                               (feature_map_2, config.anchors[3:6]),
                               (feature_map_3, config.anchors[0:3])]
        reorg_results = [restore_coord(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = tf.shape(x_y_offset)[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, C])

            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs
"""

"""
    Perform NMS on GPU using TensorFlow.
    params:
        boxes: tensor of shape [1, 10647, 4] # 10647=(13*13+26*26+52*52)*3, for input 416*416 image
        scores: tensor of shape [1, 10647, num_classes], score=conf*prob
        num_classes: total number of classes
        max_boxes: integer, maximum number of predicted boxes you'd like, default is 50
        score_thresh: if [ highest class probability score < score_threshold]
                        then get rid of the corresponding box
        nms_thresh: real value, "intersection over union" threshold used for NMS filtering
"""
"""
def gpu_nms(boxes, scores, num_classes, max_boxes=50, score_thresh=0.5, nms_thresh=0.5):
    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # since we do nms for single image, then reshape it
    boxes = tf.reshape(boxes, [-1, 4]) # '-1' means we don't konw the exact number of boxes
    score = tf.reshape(scores, [-1, num_classes])

    # Step 1: Create a filtering mask based on "box_class_scores" by using "threshold".
    mask = tf.greater_equal(score, tf.constant(score_thresh))
    # Step 2: Do non_max_suppression for each class
    for i in range(num_classes):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:,i])
        filter_score = tf.boolean_mask(score[:,i], mask[:,i])
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                   scores=filter_score,
                                                   max_output_size=max_boxes,
                                                   iou_threshold=nms_thresh, name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label
"""


def get_image_boxes(image_file, net_out):
	image = cv2.imread(image_file)
	h, w, _ = image.shape

	boxes = find_boxes(net_out)

	return image, h, w, boxes


def draw_box(image, h, w, left, top, right, bot, cls, color):
	thick = int((h + w) // 300)

	cv2.rectangle(image,
				  (left, top), (right, bot),
				  color, thick)
	cv2.putText(
		image, cls, (left, top - 12),
		0, 1e-3 * h, color,
		   thick // 3)


def draw_detection_on_image(image_file, net_out, gt, save=True):
	image, h, w, boxes = get_image_boxes(image_file, net_out)

	# draw detected boxes
	color = (0, 255, 255)
	for b in boxes:
		boxResults = process_box(b, h, w, config.THRESHOLD)
		if boxResults is None:
			continue
		left, right, top, bot, cls, max_indx, confidence = boxResults
		draw_box(image, h, w, left, top, right, bot, cls, color)

	# draw ground truth boxes
	color = (0, 0, 255)
	_, _, labels, objs = gt
	for obj, label in zip(objs, labels):
		left, top, right, bot = obj
		draw_box(image, h, w, left, top, right, bot, label, color)

	if not save:
		return image

	image_file_name = os.path.basename(image_file)
	print("image_file_name: %s" % image_file_name)
	image_file = os.path.join(config.IMAGE_OUT_DIR, image_file_name)
	cv2.imwrite(image_file, image)


def save_detection_as_json(image_file, net_out):
	image, h, w, boxes = get_image_boxes(image_file, net_out)

	resultsForJSON = []
	for b in boxes:
		boxResults = process_box(b, h, w, config.THRESHOLD)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})

	image_file_name = os.path.basename(image_file)
	textJSON = json.dumps(resultsForJSON)
	textFile = os.path.join(config.JSON_OUT_DIR, os.path.splitext(image_file_name)[0] + ".json")
	with open(textFile, 'w') as f:
		f.write(textJSON)


def postprocess(image_file, net_out, save=True):
	"""
	Takes net output, draw predictions, save to disk
	"""
	image = cv2.imread(image_file)
	h, w, _ = image.shape

	boxes = findboxes(net_out)

	resultsForJSON = []
	for b in boxes:
		boxResults = process_box(b, h, w, config.THRESHOLD)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		thick = int((h + w) // 300)
		if config.JSON:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue

		color = (248, 0, 124)
		cv2.rectangle(image,
			(left, top), (right, bot),
			color, thick)
		cv2.putText(
			image, mess, (left, top - 12),
			0, 1e-3 * h, color,
			thick // 3)

	if not save: return image

	image_file_name = os.path.basename(image_file)
	if config.JSON:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.join(config.JSON_OUT_DIR, os.path.splitext(image_file_name)[0] + ".json")
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	image_file = os.path.join(config.IMAGE_OUT_DIR, image_file_name)
	cv2.imwrite(image_file, image)
