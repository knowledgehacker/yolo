import xml.etree.ElementTree as et

import cv2
import matplotlib.pyplot as plt

import config


class Rectangle(object):
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def corner_to_centre(self):
        centre_x = (self.left + self.right) / 2.0
        centre_y = (self.top + self.bottom) / 2.0
        box_width = self.right - self.left
        box_height = self.bottom - self.top

        return centre_x, centre_y, box_width, box_height

    """
    def corner_to_centre_normalized(self, image_width, image_height):
        centre_x = (self.left + self.right) / 2.0
        centre_y = (self.top + self.bottom) / 2.0
        box_width = self.right - self.left
        box_height = self.bottom - self.top

        centre_x = centre_x / image_width
        centre_y = centre_y / image_height
        box_width = box_width / image_width
        box_height = box_height / image_height

        return centre_x, centre_y, box_width, box_height
    """

    def rescale(self, rescale_x, rescale_y):
        rec = Rectangle(self.left * rescale_x, self.top * rescale_y,
                        self.right * rescale_x, self.bottom * rescale_y)

        return rec


def parse_annotation(annotation_file):
    obj_infos = []

    f = open(annotation_file, 'rt', encoding='utf-8')

    tree = et.parse(f)
    root = tree.getroot()

    size = root.find('size')
    iw = int(size.find("width").text)
    ih = int(size.find("height").text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        #print("class: %s" % cls)
        difficult = int(obj.find('difficult').text)
        #print("difficult: %d" % difficult)
        bounding_box = obj.find('bndbox')
        left = float(bounding_box.find('xmin').text)
        top = float(bounding_box.find('ymin').text)
        right = float(bounding_box.find('xmax').text)
        bottom = float(bounding_box.find('ymax').text)
        #print("left: %f, top: %f, right: %f, bottom: %f" % (left, top, right, bottom))

        obj_infos.append((cls, difficult, Rectangle(left, top, right, bottom), (iw, ih)))

    f.close()

    return obj_infos


"""
def parse_annotations(annotation_dir, label_dir):
    files = load_files(annotation_dir)
    for file in files:
        label_file = open("%s/%s.txt" % (label_dir, file[file.rindex('/')+1:file.rindex('.')]), 'w')

        obj_infos = parse_annotation(file)
        for obj_info in obj_infos:
            cls, difficult, rec, image_shape = obj_info
            if cls not in config.CLASSES or difficult == 1:
                continue
            image_with, image_height = image_shape
            x, y, w, h = rec.corner_to_centre_normalized(image_with, image_height)
            label_file.write(str(config.CLASSES.index(cls)) + " " + "%f %f %f %f\n" % (x, y, w, h))

        label_file.close()
"""


def resize(image, recs, resized_w, resized_h):
    # resized_h, resized_w = 350, 900
    h, w, _ = image.shape
    rescale_x, rescale_y = resized_w / w, resized_h / h
    resized_image = cv2.resize(image, (resized_w, resized_h), rescale_x, rescale_y)

    resized_recs = [rec.rescale(rescale_x, rescale_y) for rec in recs]

    return resized_image, resized_recs


def draw_box_with_image(image, rec):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.imshow(image)

    h, w, _ = image.shape
    thickness = int((h + w) // 250)
    annotation = cv2.rectangle(image, (int(rec.left), int(rec.top)), (int(rec.right), int(rec.bottom)),
                               color=(255, 0, 0), thickness=thickness)
    ax2.imshow(annotation)
    # plt.axis("off")
    plt.show()


def draw_box(image_file, recs):
    image = plt.imread(image_file)
    """
    for rec in recs:
        draw_box_with_image(image, rec)
    """

    resized_image, resized_recs = resize(image, recs, config.IMG_W, config.IMG_H)
    for resized_rec in resized_recs:
        draw_box_with_image(resized_image, resized_rec)


"""
image_name = "2007_000027"
image_file = "data/VOC2012/JPEGImages/%s.jpg" % image_name
print(image_file)

annotation_file = "data/VOC2012/Annotations/%s.xml" % image_name
print(annotation_file)
obj_infos = parse_annotation(annotation_file)
recs = []
for obj_info in obj_infos:
    cls, difficult, rec, image_shape = obj_info
    if difficult != 1:
        recs.append(rec)
draw_box(image_file, recs)
"""

"""
annotation_dir = "data/VOC2012/Annotations"
label_dir = "data/VOC2012/YOLOLabels"
parse_annotations(annotation_dir, label_dir)
"""
