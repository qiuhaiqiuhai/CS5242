import numpy as np
np.random.seed(0)
from tensorflow import set_random_seed
set_random_seed(0)

import keras
from keras.models import Model
from keras.layers import Input, Conv2D
from matplotlib import pyplot as plt
from PIL import Image

plt.set_cmap('gray')
# The input image is a 224x224x1 grayscale image
a = Input(shape=(224,224,1))
# Read Keras' document about Conv2D:
# b = Conv2D(...)(a)
# Combine things together
# model = Model(...)

import pickle as pk
with open('example.pkl', 'rb') as f:
 example = pk.load(f)

image = example['img']
image_f = example['img_f']
plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(image_f)