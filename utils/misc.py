# -*- coding: utf-8 -*-

import os
import time

import config


def current_time():
    return time.strftime('%H:%M:%S', time.localtime(time.time()))


def load_files(input):
    files = []
    for dir in os.listdir(input):
        path = os.path.join(input, dir)
        if os.path.isdir(path):
            for file in os.listdir(path):
                if not file.startswith("._"):
                    files.append(os.path.join(path, file))
        else:
            if not path.startswith("._"):
                files.append(path)

    return files


def load_image_indexes(index_file):
    image_index2names = []

    fin = open(index_file, 'r')
    for line in fin:
        line = line.strip()
        name, idx_str = line.split('\t')
        idx = int(idx_str)
        image_index2names.append((idx, name))
    fin.close()

    return image_index2names


def with_prefix(prefix, op):
    return "%s/%s" % (prefix, op)
    #return op

