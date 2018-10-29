import numpy as np
np.random.seed(0)
from tensorflow import set_random_seed
from sklearn.metrics import mean_squared_error
set_random_seed(0)

import keras
from keras.models import Model
from keras.layers import Input, Conv2D
from matplotlib import pyplot as plt
from PIL import Image

import pickle as pk

def load_image(path):
    im = Image.open(path)
    im = im.resize((224,224), Image.ANTIALIAS)
    im = im.convert('L')
    return np.array(im) / 255.0

def save_image(path, arr):
    plt.imsave(path, arr)
    
def test_model_with_weight(weights, x, conv_shape=(3,3)):
    """Test the weights you get using input image
    
    Arguments:
        weights {np.array} -- filter weights obtained from the conv layer, should have shape (3,3,1,1)
        x {np.array} -- test image in shape (B,H,W,C), in our case B=1, H=W=224, C=1
    
    Returns:
        y_pred -- filtered image
    """
    a = Input(shape=(224,224,1))
    b = Conv2D(1, conv_shape, padding='valid', use_bias=False)(a)
    model = Model(inputs=a, outputs=b)
    model.get_layer(index=1).set_weights([weights])
    y_pred = model.predict(x)
    return y_pred

def validate(answer_file):
    with open('val.pkl', 'rb') as f:
        val = pk.load(f)
    with open(answer_file, 'rb') as f:
        ans = pk.load(f)
    x = np.expand_dims(np.expand_dims(val['val_img'],0),3)

    weights = ans['answer']['filter']
    val_img_output = []
    plt.set_cmap('gray')
    plt.title('Original Photo')
    plt.imshow(val['val_img'])
    plt.show()

    fig, axs = plt.subplots(2, 3, dpi=200)
    for i in range(0, 3):
        val_img_output.append(test_model_with_weight(np.expand_dims(np.expand_dims(weights[i],2),3), x, conv_shape=weights[i].shape))
        img_output = val_img_output[i][0, :, :, 0]
        img_val = val['val_img_f'][i][0, :, :, 0]
        # plt.subplot(2,3,i+1)
        axs[0][i].imshow(img_output)
        # plt.subplot(2,3,i+4)
        axs[1][i].imshow(img_val)

        err = mean_squared_error(img_val, img_output)
        axs[0][i].set_title('Image ' +str(i+1))
        axs[1][i].set_xlabel('MSE {:.6f}'.format(err))
        print('MSE between real filter and learned filter is {:.6f}'.format(err))

    axs[0][0].set_ylabel('Test')
    axs[1][0].set_ylabel('Target')

    plt.show()