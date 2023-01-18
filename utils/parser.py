import xml.etree.ElementTree as et


class Rectangle(object):
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def corner_to_centre(self):
        centre_x = 0.5 * (self.left + self.right)
        centre_y = 0.5 * (self.top + self.bottom)
        box_width = self.right - self.left
        box_height = self.bottom - self.top

        return centre_x, centre_y, box_width, box_height


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
