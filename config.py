MODEL_NAME = 'fast_yolo'

MODLE_DIR = "models"

CKPT_DIR = 'ckpt'
CKPT_PATH = '%s/%s' % (CKPT_DIR, MODEL_NAME)

DATA_DIR = 'data/VOC2012'
#ANNOT_DIR = '%s/Annotations' % DATA_DIR
#IMAGE_DIR = '%s/JPEGImages' % DATA_DIR
# test
ANNOT_DIR = '%s/Annotations' % DATA_DIR
IMAGE_TRAIN_DIR = '%s/tmp/JPEGImages' % DATA_DIR
IMAGE_TEST_DIR = '%s/tmp/JPEGImages' % DATA_DIR
#IMAGE_TRAIN_DIR = '%s/train' % DATA_DIR
#IMAGE_TEST_DIR = '%s/test' % DATA_DIR

TF_IMAGE_TRAIN_DIR = '%s/tf/train' % DATA_DIR
TF_IMAGE_TEST_DIR = '%s/tf/test' % DATA_DIR
"""
TRAIN_PATH = '%s/train' % DATA_DIR
TEST_PATH = '%s/test' % DATA_DIR

RAW_TRAIN_PATH = '%s/cnews.train.txt' % TRAIN_PATH
RAW_TEST_PATH = '%s/cnews.test.txt' % TEST_PATH
TF_TRAIN_PATH = '%s/cnews.train.tfrecords' % TRAIN_PATH
TF_TEST_PATH = '%s/cnews.test.tfrecords' % TEST_PATH
"""

JSON = False

# term scale in loss formula
object_scale = 1
noobject_scale = .5
class_scale = 1
coord_scale = 5

# image
IMG_H = 448
IMG_W = 448
IMG_CH = 3

IMG_IDX_FILE = '%s/image_indexes.txt' % DATA_DIR

# classes
"""
CLASSES = ['person',
#               0
              'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
#               1       2       3       4       5       6
              'aeroplane,', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
#               7            8            9     10      11      12          13
              'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tv/monitor']
#               14          15      16                  17          18      19
"""

CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle",
#               0           1       2       3       4
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
#   5       6       7       8       9       10          11
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
#   12          13          14          15          16          17
    "train", "tvmonitor"]
#   18          19

# grid num: S x S
#S = 7
S = 2
# Bounding box each grid
B = 2
# class num
C = len(CLASSES)

THRESHOLD = 0.17

NUM_EPOCH = 2

STEPS_PER_CKPT = 250

VALIDATE = False

# shuffle size affects convergence greatly, it should be big enough
SHUFFLE_SIZE = 5000

# large batch, ex 200, does not work, I don't know why
#BATCH_SIZE = 128
BATCH_SIZE = 2
TEST_BATCH_SIZE = 300

LEARNING_RATE = 1e-3

TRAIN_KEEP_PROB = 0.5
TEST_KEEP_PROB = 1.0

if MODEL_NAME == 'fast_yolo':
    HIDDEN_SIZE = 64

    CONV_FILTER_NUM = 128
    CONV_FILTER_KERNEL_SIZES = [2, 3, 4]
else:
    print("Unsupported model %s" % MODEL_NAME)
    exit(-1)




