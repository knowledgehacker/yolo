MODEL_NAME = 'fast_yolo'

DEVICE_TYPE = "cpu"

MODLE_DIR = "models"

CKPT_DIR = 'ckpt'
CKPT_PATH = '%s/%s' % (CKPT_DIR, MODEL_NAME)

PROF_DIR = "prof"

DATA_DIR = 'data/VOC2012'
#ANNOT_DIR = '%s/Annotations' % DATA_DIR
#IMAGE_DIR = '%s/JPEGImages' % DATA_DIR
# test
ANNOT_DIR = '%s/Annotations' % DATA_DIR
#IMAGE_TRAIN_DIR = '%s/train' % DATA_DIR
#IMAGE_TEST_DIR = '%s/test' % DATA_DIR
IMAGE_TRAIN_DIR = '%s/tmp/JPEGImages' % DATA_DIR
IMAGE_TEST_DIR = '%s/tmp/JPEGImages' % DATA_DIR

TF_IMAGE_TRAIN_DIR = '%s/tf/train' % DATA_DIR
TF_IMAGE_TEST_DIR = '%s/tf/test' % DATA_DIR

TF_IMAGE_TRAIN_FILE = '%s/tf/train/objects.tfrecords' % DATA_DIR
TF_IMAGE_TEST_FILE = '%s/tf/test/objects.tfrecords' % DATA_DIR

IMAGE_INDEX_FILE = "%s/image_indexes" % DATA_DIR

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

if DEVICE_TYPE == "gpu":
    data_format = "channels_first"
    placeholder_image_shape = (None, IMG_CH, IMG_H, IMG_W)
    input_shape = (IMG_CH, IMG_H, IMG_W)
elif DEVICE_TYPE == "cpu":
    data_format = "channels_last"
    placeholder_image_shape = (None, IMG_H, IMG_W, IMG_CH)
    input_shape = (IMG_H, IMG_W, IMG_CH)
else:
    print("Unsupported device type - %s" % DEVICE_TYPE)
    exit(-1)

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

#NUM_EPOCH = 135
NUM_EPOCH = 1

STEPS_PER_CKPT = 10

VALIDATE = False

# shuffle size affects convergence greatly, it should be big enough
SHUFFLE_SIZE = 500

# large batch, ex 200, does not work, I don't know why
#BATCH_SIZE = 128
#TEST_BATCH_SIZE = 300
BATCH_SIZE = 2
TEST_BATCH_SIZE = 2

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
