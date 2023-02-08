MODEL_NAME = 'fast_yolo'

#VERSION = "v1"
VERSION = "v2"

DEVICE_TYPE = "gpu"

MODLE_DIR = "models/%s" % VERSION

CKPT_DIR = 'ckpt/%s' % VERSION

PROF_DIR = "prof"


DATASET = 'VOC2007'

#DATA_DIR = 'data/%s' % DATASET
V1_DIR = "../yolo_v1"
DATA_DIR = '%s/data/%s' % (V1_DIR, DATASET)


ANNOTATION_TRAIN_DIR = '%s/train/Annotations' % DATA_DIR
#ANNOTATION_TRAIN_DIR = '%s/tmp/Annotations' % DATA_DIR
ANNOTATION_TRAIN_DIR = '/content/data/VOC2012/train/Annotations'
#ANNOTATION_TEST_DIR = '%s/test/Annotations' % DATA_DIR
ANNOTATION_TEST_DIR = '%s/tmp/Annotations' % DATA_DIR

IMAGE_TRAIN_DIR = '%s/train/JPEGImages' % DATA_DIR
#IMAGE_TRAIN_DIR = '%s/tmp/JPEGImages' % DATA_DIR
IMAGE_TRAIN_DIR = '/content/data/VOC2012/train/JPEGImages'
#IMAGE_TEST_DIR = '%s/test/JPEGImages' % DATA_DIR
IMAGE_TEST_DIR = '%s/tmp/JPEGImages' % DATA_DIR


IMAGE_OUT_DIR = "out/%s" % VERSION

JSON = False


# classes
CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle",
#               0           1       2       3       4
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
#   5       6       7       8       9       10          11
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
#   12          13          14          15          16          17
    "train", "tvmonitor"]
#   18          19

# class num
C = len(CLASSES)

# feature map H x W
if VERSION == "v1":
    IMG_H, IMG_W, IMG_CH = 448, 448, 3

    H, W = 7, 7
    B = 2

    # term scale in loss formula
    object_scale = 1.0
    noobject_scale = 0.5
    class_scale = 1.0
    coord_scale = 5.0

    OPTIMIZER = 'rmsprop'
    #LR = 1e-5
    MOMENTUM = 0.9
    DECAY = 5e-4

    BOUNDARIES = [30, 60]
    LRS = [2.5e-5, 5e-6, 1e-6]

    # large batch, ex 200, does not work, I don't know why
    BATCH_SIZE = 32
    TEST_BATCH_SIZE = 1

    # pretrain network
    pt_net = "extraction"
elif VERSION == "v2":
    IMG_H, IMG_W, IMG_CH = 416, 416, 3

    H, W = 13, 13
    B = 5

    # term scale in loss formula
    object_scale = 5.0
    noobject_scale = 1.0
    class_scale = 1.0
    coord_scale = 1.0

    OPTIMIZER = 'adam'
    #LR = 1e-5   # starts with 1e-5 gets nan after ~18 steps
    MOMENTUM = 0.9
    DECAY = 5e-4

    BOUNDARIES = [30, 60]
    LRS = [1e-5, 2.5e-6, 0.5e-6]

    # large batch, ex 200, does not work, I don't know why
    BATCH_SIZE = 32
    TEST_BATCH_SIZE = 1

    # pretrain network
    pt_net = "darknet19_448"

    # box priors for voc2012, (w, h), based on not coordinates but grid
    # https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg
    anchors = [1.32,1.73, 3.19,4.01, 5.05,8.10, 9.47,4.84, 11.23,10.01]
else:
    print("Unsupported version: %s" % VERSION)
    exit(-1)

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

# P(object) * P(class|object), hope P(class|object) > THRESHOLD, P(object) ~ 1.0
THRESHOLD = 0.3
#THRESHOLD = 0.0

NUM_EPOCH = 90
#NUM_EPOCH = 5

#STEPS_PER_CKPT = 10
STEPS_PER_CKPT = 1

TRAIN_KEEP_PROB = 0.5
TEST_KEEP_PROB = 1.0
