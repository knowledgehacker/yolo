MODEL_NAME = 'fast_yolo'

#VERSION = "v1"
VERSION = "v2"

DEVICE_TYPE = "gpu"

MODLE_DIR = "models"

CKPT_DIR = 'ckpt/%s' % VERSION

PROF_DIR = "prof"

DATASET = 'VOC2007'

#DATA_DIR = 'data/%s' % DATASET
V1_DIR = "../yolo_v1"
DATA_DIR = '%s/data/%s' % (V1_DIR, DATASET)


ANNOTATION_TRAIN_DIR = '%s/train/Annotations' % DATA_DIR
#ANNOTATION_TRAIN_DIR = '%s/tmp/Annotations' % DATA_DIR
#ANNOTATION_TRAIN_DIR = '%s/data/VOC2012/train/Annotations' % V1_DIR
#ANNOTATION_TEST_DIR = '%s/test/Annotations' % DATA_DIR
ANNOTATION_TEST_DIR = '%s/tmp/Annotations' % DATA_DIR

IMAGE_TRAIN_DIR = '%s/train/JPEGImages' % DATA_DIR
#IMAGE_TRAIN_DIR = '%s/tmp/JPEGImages' % DATA_DIR
#IMAGE_TRAIN_DIR = '%s/data/VOC2012/train/JPEGImages' % V1_DIR
#IMAGE_TEST_DIR = '%s/test/JPEGImages' % DATA_DIR
IMAGE_TEST_DIR = '%s/tmp/JPEGImages' % DATA_DIR

IMAGE_OUT_DIR = "%s/out/%s" % (DATA_DIR, VERSION)

JSON = False

# image
if VERSION == "v1":
    IMG_H, IMG_W = 448, 448
elif VERSION == "v2":
    IMG_H, IMG_W = 416, 416
else:
    print("Unsupported version: %s" % VERSION)
    exit(-1)

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

# box priors for voc2012, (w, h), based on not coordinates but grid
anchors = [1.32,1.73, 3.19,4.01, 5.05,8.10, 9.47,4.84, 11.23,10.01]

# feature map H x W
if VERSION == "v1":
    H, W = 7, 7
    B = 2

    # term scale in loss formula
    object_scale = 1.0
    noobject_scale = 0.5
    class_scale = 1.0
    coord_scale = 5.0
elif VERSION == "v2":
    H, W = 13, 13
    B = 5

    # term scale in loss formula
    object_scale = 5.0
    noobject_scale = 1.0
    class_scale = 1.0
    coord_scale = 1.0
else:
    print("Unsupported version: %s" % VERSION)
    exit(-1)

# class num
C = len(CLASSES)

# P(object) * P(class|object), hope P(class|object) > THRESHOLD, P(object) ~ 1.0
THRESHOLD = 0.3
#THRESHOLD = 0.0

NUM_EPOCH = 90
#NUM_EPOCH = 5

#STEPS_PER_CKPT = 10
STEPS_PER_CKPT = 1

VALIDATE = False

# large batch, ex 200, does not work, I don't know why
BATCH_SIZE = 16
TEST_BATCH_SIZE = 1
#TEST_BATCH_SIZE = 300

OPTIMIZER = 'rmsprop'
# !!!Important, for tiny model, use lr 1e-4, for large model, use lr 1e-3
LR = 1e-5
MOMENTUM = 0.9
DECAY = 5e-4

TRAIN_KEEP_PROB = 0.5
TEST_KEEP_PROB = 1.0

