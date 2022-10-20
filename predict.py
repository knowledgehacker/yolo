# -*- coding: utf-8 -*-

import os

import numpy as np
import json
import cv2

import config
from cython_utils.cy_yolo_findboxes import yolo_box_constructor


def process_box(b, h, w, threshold):
	max_indx = np.argmax(b.probs)
	max_prob = b.probs[max_indx]
	label = config.CLASSES[max_indx]
	if max_prob > threshold:
		left = int(b.x - b.w / 2. * w)
		right = int(b.x + b.w / 2. * w)
		top = int(b.y - b.h / 2. * h)
		bot = int(b.y + b.h / 2. * h)
		if left < 0:  left = 0
		if right > w - 1: right = w - 1
		if top < 0:   top = 0
		if bot > h - 1:   bot = h - 1
		mess = '{}'.format(label)
		return (left, right, top, bot, mess, max_indx, max_prob)
	return None


def findboxes(net_out, w, h):
	meta = {
		"classes": config.C,
		"num": config.B,
		"side": config.S
	}
	threshold = config.THRESHOLD
	boxes = yolo_box_constructor(meta, net_out, w, h, threshold)

	return boxes


def postprocess(image_file, net_out, save=True):
	"""
	Takes net output, draw predictions, save to disk
	"""
	image = cv2.imread(image_file)
	h, w, _ = image.shape

	boxes = findboxes(net_out, w, h)

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

		color = (127, 127, 127)
		cv2.rectangle(image,
			(left, top), (right, bot),
			color, thick)
		cv2.putText(
			image, mess, (left, top - 12),
			0, 1e-3 * h, color,
			thick // 3)

	if not save: return image

	image_name = os.path.join(config.IMAGE_OUT_DIR, os.path.basename(image_file))
	if config.JSON:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(image_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(image_name, image)
