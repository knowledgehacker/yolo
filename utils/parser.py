import xml.etree.ElementTree as et

import cv2
import matplotlib.pyplot as plt


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
    def rescale(self, rescale_x, rescale_y):
        rec = Rectangle(self.left * rescale_x, self.top * rescale_y,
                        self.right * rescale_x, self.bottom * rescale_y)

        return rec
    """


def parse_annotation(annotation_file):
    obj_infos = []

    f = open(annotation_file, 'rt', encoding='utf-8')

    tree = et.parse(f)
    root = tree.getroot()

    size = root.find('size')
    image_w = int(size.find("width").text)
    image_h = int(size.find("height").text)
    #print("image_w: %d, image_h" % image_w, image_h)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        difficult = int(obj.find('difficult').text)
        bounding_box = obj.find('bndbox')
        left = float(bounding_box.find('xmin').text)
        top = float(bounding_box.find('ymin').text)
        right = float(bounding_box.find('xmax').text)
        bottom = float(bounding_box.find('ymax').text)
        #print("class: %s, difficult: %d" % (cls, difficult))
        #print("left: %f, top: %f, right: %f, bottom: %f" % (left, top, right, bottom))

        obj_infos.append((cls, difficult, Rectangle(left, top, right, bottom)))

    f.close()

    return image_w, image_h, obj_infos


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
    for rec in recs:
        draw_box_with_image(image, rec)


