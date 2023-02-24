# -*- coding: utf-8 -*-

import sys
import os
import glob
import xml.etree.ElementTree as ET
import json

import config

map_input_dir = 'mAP/input'
"""
# generate ground truth input from VOC annotations
ann_out_dir = '%s/ground-truth' % map_input_dir

if not os.path.exists(ann_out_dir):
    os.makedirs(ann_out_dir)

ann_in_dir = config.ANNOTATION_TEST_DIR
xml_list = glob.glob('%s/*.xml' % ann_in_dir)
if len(xml_list) == 0:
    print("Error: no .xml files found in %s" % ann_in_dir)
    sys.exit()

print("Convert voc .xml files starts...")

for ann_file in xml_list:
    ann_file_name = ann_file[ann_file.rindex('/')+1:]
    box_file = "%s/%s.txt" % (ann_out_dir, ann_file_name[:ann_file_name.rindex('.')])
    print("ann_file: %s, box_file: %s" % (ann_file, box_file))
    with open(box_file, "a") as f:
        root = ET.parse(ann_file).getroot()
        for obj in root.findall('object'):
            obj_name = obj.find('name').text
            bndbox = obj.find('bndbox')
            left = bndbox.find('xmin').text
            top = bndbox.find('ymin').text
            right = bndbox.find('xmax').text
            bottom = bndbox.find('ymax').text
            f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

print("Convert voc .xml files completed!")
"""
########################################################

# generate detection input from json files
detect_out_dir = '%s/detection-results' % map_input_dir

if not os.path.exists(detect_out_dir):
    os.makedirs(detect_out_dir)

pred_in_dir = config.JSON_OUT_DIR
json_list = glob.glob('%s/*.json' % pred_in_dir)
if len(json_list) == 0:
    print("Error: no .json files found in %s" % pred_in_dir)
    sys.exit()

print("Convert predict .json files starts...")

for pred_file in json_list:
    pred_file_name = pred_file[pred_file.rindex('/')+1:]
    box_file = "%s/%s.txt" % (detect_out_dir, pred_file_name[:pred_file_name.rindex('.')])
    print("pred_file: %s, box_file: %s" % (pred_file, box_file))
    with open(box_file, "a") as f:
        data = json.load(open(pred_file))
        for obj in data:
          obj_name = obj['label']
          conf = obj['confidence']
          left = obj['topleft']['x']
          top = obj['topleft']['y']
          right = obj['bottomright']['x']
          bottom = obj['bottomright']['y']
          f.write(obj_name + " " + str(conf) + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom) + '\n')

print("Convert predict .json completed!")

