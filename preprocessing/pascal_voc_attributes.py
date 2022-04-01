"""
Copyright 2017-2018 Fizyr (https://fizyr.com)
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ..preprocessing.generator_attributes import Generator
from ..utils.image import read_image_bgr
from ..utils.anchors_attributes import bbox_transform

import os
import numpy as np
from six import raise_from
from PIL import Image
import keras

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

voc_classes = {
    'aeroplane'   : 0,
    'bicycle'     : 1,
    'bird'        : 2,
    'boat'        : 3,
    'bottle'      : 4,
    'bus'         : 5,
    'car'         : 6,
    'cat'         : 7,
    'chair'       : 8,
    'cow'         : 9,
    'diningtable' : 10,
    'dog'         : 11,
    'horse'       : 12,
    'motorbike'   : 13,
    'person'      : 14,
    'pottedplant' : 15,
    'sheep'       : 16,
    'sofa'        : 17,
    'train'       : 18,
    'tvmonitor'   : 19
}


def _findNode(parent, name, debug_name=None, parse=None):
    if debug_name is None:
        debug_name = name

    result = parent.find(name)
    if result is None:
        raise ValueError('missing element \'{}\''.format(debug_name))
    if parse is not None:
        try:
            return parse(result.text)
        except ValueError as e:
            raise_from(ValueError('illegal value for \'{}\': {}'.format(debug_name, e)), None)
    return result


class PascalVocGenerator(Generator):
    """ Generate data for a Pascal VOC dataset.
    See http://host.robots.ox.ac.uk/pascal/VOC/ for more information.
    """

    def __init__(
        self,
        data_dir,
        set_name,
        classes=voc_classes,
        image_extension='.jpg',
        skip_truncated=False,
        skip_difficult=False,
        **kwargs
    ):
        """ Initialize a Pascal VOC data generator.
        Args
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
            csv_class_file: Path to the CSV classes file.
        """
        self.data_dir             = data_dir
        self.set_name             = set_name
        self.classes              = classes
        self.attributes = self.read_attributes()
        self.image_names          = [l.strip().split(None, 1)[0] for l in open(os.path.join(data_dir, 'ImageSets', 'Layout', set_name + '.txt')).readlines()]
        self.image_extension      = image_extension
        self.skip_truncated       = skip_truncated
        self.skip_difficult       = skip_difficult

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        self.attr_labels = {}
        for key, value in self.attributes.items():
            self.attr_labels[value] = key

        super(PascalVocGenerator, self).__init__(**kwargs)

    def read_attributes(self):
        attributes = {}
        attr = [l.strip() for l in open(os.path.join(self.data_dir, 'attribute_names.txt')).readlines()]
        for label, name in enumerate(attr):
            attributes[name] = label
        return attributes

    def num_attributes(self):
        return len(self.attributes)

    def attr_to_label(self, attr):
        return self.attributes[attr]

    def label_to_attr(self, label):
        return self.attr_labels[label]

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return len(self.classes)

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        path  = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        image = Image.open(path)
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        return read_image_bgr(path)

    def compute_targets(self, image_group, annotations_group):
        """ Compute target outputs for the network using images and their annotations.
        """
        # get the max image shape
        max_shape = tuple(max(image.shape[x] for image in image_group) for x in range(3))

        labels_group = [None] * self.batch_size
        regression_group = [None] * self.batch_size
        attribute_group = [None] * self.batch_size

        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # compute regression targets
            labels_group[index], annotations, anchors, attribute_group[index] = self.compute_anchor_targets(
                max_shape,
                annotations,
                self.num_classes(),
                mask_shape=image.shape,
                num_attributes=self.num_attributes()
            )
        regression_group[index] = bbox_transform(anchors, annotations)

        anchor_states = np.max(labels_group[index], axis=1, keepdims=True)
        regression_group[index] = np.append(regression_group[index], anchor_states, axis=1)

        labels_batch     = np.zeros((self.batch_size,) + labels_group[0].shape, dtype=keras.backend.floatx())
        regression_batch = np.zeros((self.batch_size,) + regression_group[0].shape, dtype=keras.backend.floatx())
        attributes_batch = np.zeros((self.batch_size,) + attribute_group[0].shape, dtype=keras.backend.floatx())

        for index, (labels, regression, attributes) in enumerate(zip(labels_group, regression_group, attribute_group)):
            labels_batch[index, ...]     = labels
            regression_batch[index, ...] = regression
            attributes_batch[index, ...] = attributes

        return [regression_batch, labels_batch, attributes_batch]

    def __parse_annotation(self, element):
        """ Parse an annotation given an XML element.
        """
        truncated = _findNode(element, 'truncated', parse=int)
        difficult = _findNode(element, 'difficult', parse=int)

        class_name = _findNode(element, 'name').text
        if class_name not in self.classes:
            raise ValueError('class name \'{}\' not found in classes: {}'.format(class_name, list(self.classes.keys())))
        temp_attribute = np.zeros(64)
        for attribute in element.iter('attribute'):
            label = self.attr_to_label(attribute.text)
            temp_attribute[label] = 1

        box = np.zeros((1, 69))
        box[0, 4] = self.name_to_label(class_name)

        bndbox    = _findNode(element, 'bndbox')
        box[0, 0] = _findNode(bndbox, 'xmin', 'bndbox.xmin', parse=float)
        box[0, 1] = _findNode(bndbox, 'ymin', 'bndbox.ymin', parse=float)
        box[0, 2] = _findNode(bndbox, 'xmax', 'bndbox.xmax', parse=float)
        box[0, 3] = _findNode(bndbox, 'ymax', 'bndbox.ymax', parse=float)

        box[0, 5:] = temp_attribute

        return truncated, difficult, box

    def __parse_annotations(self, xml_root):
        """ Parse all annotations under the xml_root.
        """
        boxes = np.zeros((0, 69))
        for i, element in enumerate(xml_root.iter('object')):
            try:
                truncated, difficult, box = self.__parse_annotation(element)
            except ValueError as e:
                raise_from(ValueError('could not parse object #{}: {}'.format(i, e)), None)

            if truncated and self.skip_truncated:
                continue
            if difficult and self.skip_difficult:
                continue
            boxes = np.append(boxes, box, axis=0)

        return boxes

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        filename = self.image_names[image_index] + '.xml'
        try:
            tree = ET.parse(os.path.join(self.data_dir, 'xml', filename))
            return self.__parse_annotations(tree.getroot())
        except ET.ParseError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError('invalid annotations file: {}: {}'.format(filename, e)), None)