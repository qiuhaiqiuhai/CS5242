import csv
import numpy as np
# np.random.seed(0)
from tensorflow import set_random_seed
# set_random_seed(0)

import keras
from keras.models import Model
from keras.layers import Input, Conv2D
from matplotlib import pyplot as plt
from keras import optimizers, losses
from PIL import Image
from keras.utils.vis_utils import plot_model


plt.set_cmap('gray')

MatricNum = 'A0191508A'
data_set = [
    {'name':'first', 'shape':(3,3)},
    {'name':'second', 'shape':(3,3)},
    {'name':'third', 'shape':(5,5)}
]

def save_filter(name, learned_filter):

    with open(name+".csv", 'w') as csvfile:
        myWriter = csv.writer(csvfile, lineterminator='\n')
        myWriter.writerows(learned_filter)

def load_filter(name):
    with open(name+'.csv') as csvfile:
        rows = csv.reader(csvfile)
        res = list(rows)
    return np.asarray(res, dtype=np.float64)

def generate_answer(filters):
    answer = {
        # your name as shown in IVLE
        'Name': 'QIU HAI',
        # your Matriculation Number, starting with letter 'A'
        'MatricNum': MatricNum,
        # do check the size of filters
        'answer': {'filter': filters}
    }
    with open(MatricNum+'.pkl', 'wb') as f:
        pk.dump(answer, f)


def plot_filter(filter_name, learned_filter=None):
    if(learned_filter is None):
        learned_filter = load_filter(filter_name)
    fig = plt.figure()
    plt.xticks([i for i in range(learned_filter.shape[0])])
    plt.yticks([i for i in range(learned_filter.shape[1])])
    cs = plt.imshow(learned_filter)
    cbaxes = fig.add_axes([0.1, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(cs, cax=cbaxes)
    plt.title(filter_name + ' filter')
    plt.show()





import pickle as pk

def plot_load_origin(file_name):
    with open(file_name, 'rb') as f:
        example = pk.load(f)

        images = example['img']
        image_fs = example['img_f']
        fig, ax = plt.subplots(2,3, dpi=200)

        for i in range(3):
            ax[0][i].imshow(images[i])
            ax[1][i].imshow(image_fs[i])
            ax[0][i].set_title('Image '+str(i+1))

        ax[0][0].set_ylabel('Origin')
        ax[1][0].set_ylabel('Filtered')

        plt.show()
    return images, image_fs

def plot_mse(histories, opt_labels, title):
    nb_of_epochs = histories[0].params['epochs']
    epochs_x = np.linspace(1, nb_of_epochs, num=nb_of_epochs)
    for i in range(len(histories)):
        optimizer = histories[i].model.optimizer
        plt.plot(histories[i].history['loss'], 'C'+str(i), linewidth=1.5, label=opt_labels[i])
    plt.legend(loc=1)
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title(title)
    plt.show()




    # best_val = np.min(val_costs);
    # best_epoch = np.argmin(val_costs) + 1;
    # plt.xlabel('{} epochs'.format(nb_of_epochs))
    # plt.ylabel('Cross Entropy Cost')
    # plt.title(module_name + ' Best Validation Performance is {:.4f} at epoch {}'.format(best_val, best_epoch))
    #
    # plt.axhline(y=best_val, color="blue", linestyle="--", linewidth=0.5)
    # plt.axvline(x=best_epoch, color="blue", linestyle="--", linewidth=0.5)
    #
    # plt.axis((0, nb_of_epochs, 0, 2.5))
    # plt.grid()
    # plt.show()


def train(model, name, image, image_f, epochs=50, verbose=2, save=False, plot_filter=False):
    set_random_seed(1)
    np.random.seed(1)
    x = np.expand_dims(np.expand_dims(image, 0), 3)
    y = np.expand_dims(np.expand_dims(image_f, 0), 3)
    history = model.fit(x=x, y=y, epochs=epochs, verbose=verbose)
    plot_model(model, to_file='model.png', show_shapes=True)


    conv_weights = model.get_layer(index=1).get_weights()[0]
    learned_filter = conv_weights[:, :, 0, 0]
    if(save):
        save_filter(name, learned_filter)

    if(plot_filter):
        plot_filter(name, learned_filter=learned_filter)

    return history

def network(filter=None, filter_shape=(3,3), optimizer=optimizers.SGD()):
    set_random_seed(1)
    np.random.seed(1)

    a = Input(shape=(224,224,1))
    b = Conv2D(1, filter_shape, padding='valid', use_bias=False)(a)
    model = Model(inputs=a, outputs=b)
    model.compile(loss=losses.mean_squared_error, optimizer=optimizer)
    if(filter is not None):
        model.get_layer(index=1).set_weights([np.expand_dims(np.expand_dims(load_filter(filter),2),3)])
    return model


images, image_fs = plot_load_origin('problems.pkl')

# with open('val.pkl', 'rb') as f:
#     val = pk.load(f)
# images, image_fs = [val['val_img'],val['val_img'],val['val_img']] , val['val_img_f']

num=2;
name = data_set[num]['name']
histories = []
for optimizer in [optimizers.SGD(lr=0.1)]:
    cnn = network(filter=None, filter_shape=data_set[num]['shape'], optimizer=optimizer )
    history = train(cnn, name, images[num], image_fs[num], epochs=5000, verbose=0, save=False)
    histories.append(history)
plot_mse(histories, ["SGD","Adam","Adagrad","Adadelta"], 'Performance of different optimizer')
# train('second', images[1], image_fs[1], epochs=2000, verbose=1)
# train('third', images, image_fs[2][0,:,:,0], epochs=200, verbose=1)

# plot_filter('third')
# filters = [load_filter('first'), load_filter('second'), load_filter('third')]
# generate_answer(filters)
# # #
# from assign2_utils import validate
# validate('A0191508A.pkl')