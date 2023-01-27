MODEL_NAME = 'fast_yolo'

DEVICE_TYPE = "gpu"

MODLE_DIR = "models"

CKPT_DIR = 'ckpt'

PROF_DIR = "prof"

DATASET = 'VOC2007'

#DATA_DIR = 'data/%s' % DATASET
V1_DIR = "../yolo_v1"
DATA_DIR = '%s/data/%s' % (V1_DIR, DATASET)


#ANNOTATION_TRAIN_DIR = '%s/train/Annotations' % DATA_DIR
ANNOTATION_TRAIN_DIR = '%s/tmp/Annotations' % DATA_DIR
#ANNOTATION_TRAIN_DIR = 'data/VOC2012/train/Annotations'
#ANNOTATION_TEST_DIR = '%s/test/Annotations' % DATA_DIR
ANNOTATION_TEST_DIR = '%s/tmp/Annotations' % DATA_DIR

#IMAGE_TRAIN_DIR = '%s/train/JPEGImages' % DATA_DIR
IMAGE_TRAIN_DIR = '%s/tmp/JPEGImages' % DATA_DIR
#IMAGE_TRAIN_DIR = 'data/VOC2012/train/JPEGImages'
#IMAGE_TEST_DIR = '%s/test/JPEGImages' % DATA_DIR
IMAGE_TEST_DIR = '%s/tmp/JPEGImages' % DATA_DIR

IMAGE_OUT_DIR = "%s/out" % DATA_DIR

JSON = False


# image
IMG_H, IMG_W, IMG_CH = 416, 416, 3


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

# class num
C = len(CLASSES)

# feature map H x W
H = 13
W = 13
# Bounding box each feature map location
B = 5

# term scale in loss formula
object_scale = 5.0
noobject_scale = 1.0
class_scale = 1.0
coord_scale = 1.0

# box priors, (w, h), based on not coordinates but grid
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg
anchors = [1.32,1.73, 3.19,4.01, 5.05,8.10, 9.47,4.84, 11.23,10.01]

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
LR = 1e-6   # starts with 1e-5 gets nan after ~18 steps
MOMENTUM = 0.9
DECAY = 5e-4

TRAIN_KEEP_PROB = 0.5
TEST_KEEP_PROB = 1.0
