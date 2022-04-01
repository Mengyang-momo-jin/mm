import matplotlib
matplotlib.use('Agg')
from keras.models import Model
import os
import sys
import numpy as np
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401

    __package__ = "keras_retinanet.bin"

from .. import models

backbone_name = "resnet50"
filepath = './snapshots/resnet50_attributes_macro.h5'
model = models.load_model(filepath, backbone_name=backbone_name, convert=True)
print(model.summary())


image = read_image_bgr('./results/test3/1.jpg')
image = preprocess_image(image)
#image, scale = resize_image(image)

# The name of the layer we want to visualize
# (see model definition in vggnet.py)
layer_name = 'res3d'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
intermediate_output = intermediate_layer_model.predict(np.expand_dims(image, axis=0))

print(intermediate_output.shape)
#plt.imshow(intermediate_output[0, :, :, 0])
for i in range(256):
    im_resize = cv2.resize(intermediate_output[0, :, :, i], (640, 480), interpolation=cv2.INTER_LANCZOS4)
    #cv2.imshow("Image", im_resize)
    #cv2.waitKey()
    cv2.imwrite("./results/test3/image{}.jpg".format(i), np.asarray(im_resize, dtype=np.uint8))