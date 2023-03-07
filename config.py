MODEL_NAME = 'fast_yolo'

VERSION = "v3"

DEVICE_TYPE = "gpu"

MODLE_DIR = "models/%s" % VERSION

CKPT_DIR = 'ckpt/%s' % VERSION

PROF_DIR = "prof"

DATA_DIR = 'data'

VOC2007_DIR = '%s/%s' % (DATA_DIR, 'VOC2007')
VOC2012_2007_DIR = '%s/%s' % (DATA_DIR, 'VOC2012')

ANNOTATION_TRAIN_DIR = '/content/%s/train/Annotations' % VOC2012_2007_DIR
IMAGE_TRAIN_DIR = '/content/%s/train/JPEGImages' % VOC2012_2007_DIR

"""
# kaggle
ANNOTATION_TRAIN_DIR = '%s/train/Annotations' % VOC2012_2007_DIR
IMAGE_TRAIN_DIR = '%s/train/JPEGImages' % VOC2012_2007_DIR
"""

#ANNOTATION_TEST_DIR = '%s/test/Annotations' % VOC2007_DIR
#IMAGE_TEST_DIR = '%s/test/JPEGImages' % VOC2007_DIR

#ANNOTATION_TEST_DIR = '/content/%s/test/Annotations' % VOC2007_DIR
#IMAGE_TEST_DIR = '/content/%s/test/JPEGImages' % VOC2007_DIR

ANNOTATION_TEST_DIR = 'data/tmp/Annotations'
IMAGE_TEST_DIR = 'data/tmp/JPEGImages'

OUT_DIR = "out/%s" % VERSION
IMAGE_OUT_DIR = "%s/Image" % OUT_DIR
JSON_OUT_DIR = "%s/Json" % OUT_DIR

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
IMG_H, IMG_W, IMG_CH = 416, 416, 3

H, W = 13, 13
B = 5

# term scale in loss formula
object_scale = 5.0
noobject_scale = 1.0
class_scale = 1.0
coord_scale = 1.0

OPTIMIZER = 'adam'

BOUNDARIES = [0, 45, 90]
LRS = [1e-4, 1e-5, 1e-6]
BATCH_SIZES = [32, 32, 40]

TEST_BATCH_SIZE = 1

# pretrain network
pt_net = "darknet19_448"

# box priors for voc2012, (w, h), based on not coordinates but grid
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg
anchors = [1.32, 1.73, 3.19, 4.01, 5.05, 8.10, 9.47, 4.84, 11.23, 10.01]

IOU_THRESHOLD = 0.5

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
THRESHOLD = 0.6

NUM_EPOCH = 90

STEPS_PER_CKPT = 1

TRAIN_KEEP_PROB = 0.5
TEST_KEEP_PROB = 1.0
