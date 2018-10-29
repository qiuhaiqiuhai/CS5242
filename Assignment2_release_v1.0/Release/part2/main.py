from assign2_utils_p2 import mnist_reader
from assign2_utils_p2 import cifar10_reader
from keras.utils.vis_utils import plot_model
from keras.callbacks import Callback as iCallback
from keras.callbacks import CSVLogger
import plot
import numpy as np
import csv

MatricNum = 'A0191508A'
mnist = {'input_shape':(28,28,1)}
cifar = {'input_shape':(32,32,3)}

# plt.set_cmap('gray')
# plt.imshow(train_x[0,:,:,0])
# plt.show()
# print('The label is {}'.format(class_name[list(train_y[0]).index(1)]))

# plt.imshow(train_x[1,:,:,:])
# plt.show()
# print('The label is {}'.format(class_name[list(train_y[1]).index(1)]))

class LossHistory(iCallback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.avg_loss = []
        self.avg_acc = []
        self.epoch = 0

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.avg_loss.append(np.mean(self.losses[-batch-1:]))
        self.avg_acc.append(np.mean(self.acc[-batch-1:]))

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch

    def on_train_end(self, logs={}):
        with open('history.csv', 'w') as csvfile:
            historywriter = csv.writer(csvfile, lineterminator='\n')
            # historywriter.writeheader()
            historywriter.writerows([self.avg_loss, self.avg_acc, self.losses, self.acc])



from keras import optimizers, losses
from network import my_network as test_network

# data_chosen = mnist
# if(data_chosen == mnist):
#     train_x, train_y, test_x, test_y, class_name = mnist_reader()
# else:
#     train_x, train_y, test_x, test_y, class_name = cifar10_reader()

f = open(MatricNum+".json", "r")
json_model = f.read()
f.close()

from keras.models import model_from_json
model = model_from_json(json_model)
model.save('example.h5')
plot_model(model, to_file='model.png', show_shapes=True)
# model = test_network(input_shape=data_chosen['input_shape'])
# f = open(MatricNum+".json", "w")
# f.write(model.to_json())
# model.save_weights('rand_network_weights.h5')
# plot_model(model, to_file='model.png', show_shapes=True)
#
# history_log = LossHistory()
# csv_logger = CSVLogger('log.csv')
# histories = []
# for optimizer in [optimizers.adadelta()]:
#     # set_random_seed(1)
#     # np.random.seed(1)
#     # model.load_weights('rand_network_weights.h5', by_name=True)
#     model.load_weights('my_network_weights.h5', by_name=True)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     history = model.fit(batch_size=32, x=train_x, y=train_y, epochs=10, verbose=1, callbacks=[history_log, csv_logger])
#     model.save_weights('my_network_weights.h5')
#     histories.append(history)
#     loss, acc = model.evaluate(x=test_x, y=test_y)
#     print('Test accuracy is {:.4f}, loss is {:.4f}'.format(acc, loss))
# plot.plot_compare(histories, ["SGD","Adam","Adagrad","Adadelta"], 'Performance of different optimizer', type='loss')
# plot.plot_compare(histories, ["SGD","Adam","Adagrad","Adadelta"], 'Performance of different optimizer', type='acc')
#
# model.save(MatricNum+'_cifar10.h5')
# model.load_weights('my_network_weights.h5', by_name=True)
#
# plot.plot_prediction(test_x[:320], model.predict(x=test_x[:320]), class_name)

# from keras.models import load_model
# model = load_model('A0191508A_mnist.h5')
# loss, acc = model.evaluate(x=test_x, y=test_y)
# print('Test accuracy is {:.4f}, loss is {:.4f}'.format(acc, loss))