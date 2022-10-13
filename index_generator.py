# -*- coding: utf-8 -*-
import os

import config
from utils.misc import current_time, load_files


def generate(input, data_type):
    # load image files
    image_files = load_files(input)

    print(current_time(), "Generate %s image indexes starts ..." % data_type)
    fout = open("%s-%s.txt" % (config.IMAGE_INDEX_FILE, data_type), 'w')
    for image_idx in range(len(image_files)):
        # image.shape: (h, w, c)
        image_name = os.path.basename(image_files[image_idx])
        fout.write("%s\t%d\n" % (image_name, image_idx))
    fout.close()
    print(current_time(), "Generate %s image indexes finished!" % data_type)

    return image_files


generate(config.IMAGE_TRAIN_DIR, "train")
generate(config.IMAGE_TEST_DIR, "test")
