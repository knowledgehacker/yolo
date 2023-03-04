# -*- coding: utf-8 -*-
"""
parse PASCAL VOC xml annotations
"""

import os
import sys
import xml.etree.ElementTree as ET
import glob

import config


def _pp(l): # pretty printing 
    for i in l: print('{}: {}'.format(i,l[i]))


def parse(ann_dir, classes, exclusive=False):
    print('Parsing for {} {}'.format(classes, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ann_dir)
    anns = os.listdir('.')
    anns = glob.glob(str(anns)+'*.xml')
    size = len(anns)

    for i, file in enumerate(anns):
        # progress bar      
        sys.stdout.write('\r')
        percentage = 1. * (i+1) / size
        progress = int(percentage * 20)
        bar_arg = [progress*'=', ' '*(19-progress), percentage*100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()
        
        # actual parsing 
        in_file = open(file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        jpg = str(root.find('filename').text)
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        #all = list()
        labels = list()
        objs = list()

        for obj in root.iter('object'):
                #current = list()
                name = obj.find('name').text
                if name not in classes:
                        continue

                xmlbox = obj.find('bndbox')
                xn = int(float(xmlbox.find('xmin').text))
                xx = int(float(xmlbox.find('xmax').text))
                yn = int(float(xmlbox.find('ymin').text))
                yx = int(float(xmlbox.find('ymax').text))
                #current = [name, xn, yn, xx, yx]
                # all += [current]
                labels += [name]
                objs += [[xn, yn, xx, yx]]

        #add = [[jpg, [w, h, all]]]
        add = [[jpg, [w, h, labels, objs]]]
        dumps += add
        in_file.close()

    # gather all stats
    gather_stats(dumps, classes)

    os.chdir(cur_dir)

    return dumps


def gather_stats(dumps, classes):
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in classes:
                if current[0] in stat:
                    stat[current[0]] += 1
                else:
                    stat[current[0]] = 1

    print('Statistics:')
    _pp(stat)
    print('Dataset size: {}'.format(len(dumps)))
