MODEL_NAME = 'fast_yolo'

DEVICE_TYPE = "gpu"

MODLE_DIR = "models"

CKPT_DIR = 'ckpt'
CKPT_PATH = '%s/%s' % (CKPT_DIR, MODEL_NAME)

PROF_DIR = "prof"

DATASET = 'VOC2007'
DATA_DIR = 'data/%s' % DATASET

ANNOTATION_TRAIN_DIR = '%s/train/Annotations' % DATA_DIR
ANNOTATION_TEST_DIR = '%s/test/Annotations' % DATA_DIR
ANNOTATION_TMP_DIR = '%s/tmp/Annotations' % DATA_DIR
IMAGE_TRAIN_DIR = '%s/train/JPEGImages' % DATA_DIR
IMAGE_TEST_DIR = '%s/test/JPEGImages' % DATA_DIR
IMAGE_TMP_DIR = '%s/tmp/JPEGImages' % DATA_DIR

TF_IMAGE_TRAIN_FILE = '%s/tf/train/objects.tfrecords' % DATA_DIR
TF_IMAGE_TEST_FILE = '%s/tf/test/objects.tfrecords' % DATA_DIR
TF_IMAGE_TMP_FILE = '%s/tf/tmp/objects.tfrecords' % DATA_DIR

IMAGE_INDEX_FILE = "%s/image_indexes" % DATA_DIR

IMAGE_OUT_DIR = "%s/out" % DATA_DIR

JSON = False

# term scale in loss formula
#object_scale = 1
noobject_scale = 0.5
class_scale = 1.0
coord_scale = 5.0

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
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle",
#               0           1       2       3       4
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
#   5       6       7       8       9       10          11
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
#   12          13          14          15          16          17
    "train", "tvmonitor"]
#   18          19

# grid num: S x S
S = 7
#S = 2
# Bounding box each grid
B = 2
# class num
C = len(CLASSES)

#THRESHOLD = 0.17
THRESHOLD = 0.0

#NUM_EPOCH = 90
NUM_EPOCH = 5

STEPS_PER_CKPT = 10
#STEPS_PER_CKPT = 1

VALIDATE = False

# shuffle size affects convergence greatly, it should be big enough
SHUFFLE_SIZE = 500

# large batch, ex 200, does not work, I don't know why
BATCH_SIZE = 64
TEST_BATCH_SIZE = 300
"""
BATCH_SIZE = 2
TEST_BATCH_SIZE = 2
"""

OPTIMIZER = 'rmsprop'
# !!!Important, for tiny model, use lr 1e-4, for large model, use lr 1e-3
LR = 1e-4
MOMENTUM = 0.9
DECAY = 5e-4

TRAIN_KEEP_PROB = 0.5
TEST_KEEP_PROB = 1.0

