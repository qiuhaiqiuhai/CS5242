from sklearn import metrics
from layer import *
import csv

class Nnet:

    def __init__(self, layer_sizes, random=True, W=None, B=None):
        layers = []

        if random:
            for i in range(len(layer_sizes) - 2):
                layers.append(FullyConnectedLayer(layer_sizes[i], layer_sizes[i + 1]))
                layers.append(ReLULayer())

            layers.append(FullyConnectedLayer(layer_sizes[-2], layer_sizes[-1]))
            layers.append(Softmax_CrossEntropyLayer())
        else:
            for i in range(len(layer_sizes) - 2):
                layers.append(FullyConnectedLayer(random=False, W=W[i], B=B[i]))
                layers.append(ReLULayer())

            layers.append(FullyConnectedLayer(random=False, W=W[-1], B=B[-1]))
            layers.append(Softmax_CrossEntropyLayer())

        self.layers = layers
        self.set_train_param()

    def save_wb(self):
        W=[]
        b=[]
        for layer in self.layers:
            if type(layer) is FullyConnectedLayer:
                W.append(layer.get_w())
                b.append(layer.get_b())

        with open("weight.csv", 'w') as csvfile:
            myWriter = csv.writer(csvfile, lineterminator='\n')
            for rows in W:
                myWriter.writerows(rows.tolist())

        with open("bias.csv", 'w') as csvfile:
            myWriter = csv.writer(csvfile, lineterminator='\n')
            myWriter.writerows(b)


    def set_train_param(self, batch_size = 20, max_nb_of_epochs = 1000, learning_rate = 0.001, momentum = 0.1, lrate_drop = 0.8, epochs_drop = 40):
        self.batch_size = batch_size
        self.max_nb_of_epochs = max_nb_of_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.lrate_drop = lrate_drop
        self.epochs_drop = epochs_drop

    def forward_prop(self, input):

        outputs = [input]
        X = input

        for layer in self.layers:
            Y = layer.output(X)
            outputs.append(Y)
            X = outputs[-1]
        return outputs


    def back_prop(self, outputs, targets):

        w_grads, b_grads = [], []
        next_grad = None

        for layer in reversed(self.layers):
            Y = outputs.pop()
            X = outputs[-1]
            w_grads.insert(0, layer.get_w_grad(X, next_grad))
            b_grads.insert(0, layer.get_b_grad(X, next_grad))

            if type(layer) is Softmax_CrossEntropyLayer:
                cur_grad = layer.calc_grad(Y, targets)
            else:
                cur_grad = layer.calc_grad(Y, next_grad)

            next_grad = cur_grad
        return w_grads, b_grads



    def update_wb(self, w_grads, b_grads, learning_rate):
        # if old_w_grads is None:
        for layer, w_grad_layer, b_grad_layer\
            in zip(self.layers, w_grads, b_grads):
            for weight, bias, w_grad_neuron, b_grad_neuron in \
                    zip(layer.get_w(), layer.get_b(), w_grad_layer, b_grad_layer):
                weight -= learning_rate * w_grad_neuron
                bias -= learning_rate * b_grad_neuron

        # else:
        #     for layer, w_grad_layer, b_grad_layer,  old_w_grad_layer, old_b_grad_layer\
        #         in zip(self.layers, w_grads, b_grads, old_w_grads, old_b_grads):
        #         for weight, bias, w_grad_neuron, b_grad_neuron, old_w_grad_neuron, old_b_grad_neuron  in \
        #                 zip(layer.get_w(), layer.get_b(), w_grad_layer, b_grad_layer, old_w_grad_layer, old_b_grad_layer):
        #             weight -= self.learning_rate * w_grad_neuron + self.learning_rate * self.momentum*old_w_grad_neuron
        #             bias -= self.learning_rate * b_grad_neuron + self.learning_rate * self.momentum*old_b_grad_neuron

    def step_decay(self, epoch):
        initial_lrate = self.learning_rate
        drop = self.lrate_drop
        epochs_drop = self.epochs_drop
        cyc = 100
        lrate = initial_lrate*2**(epoch//cyc) * np.power(drop, np.floor(epoch / epochs_drop))
        return lrate

    def calc_cost_acc (self, X, T, X_label):
        outputs = self.forward_prop(X)
        cost = self.layers[-1].calc_cost(outputs[-1], T)
        y_label = np.argmax(outputs[-1], axis=1)
        acc = metrics.accuracy_score(X_label, y_label)

        return cost, acc

    def train(self, X_train, T_train, X_validation, T_validation, X_test, T_test):

        nb_of_batches = X_train.shape[0] // self.batch_size

        XT_batches = list(zip(
            np.array_split(X_train, nb_of_batches, axis=0),
            np.array_split(T_train, nb_of_batches, axis=0)))

        minibatch_costs = []
        train_costs = []
        train_acc = []
        val_costs = []
        val_acc = []
        test_costs = []
        test_acc = []
        lrate = []

        x_train_label = np.argmax(T_train, axis=1)
        x_val_label = np.argmax(T_validation, axis=1)
        x_test_label = np.argmax(T_test, axis=1)


        for epoch in range(self.max_nb_of_epochs):
            # old_w_grads, old_b_grads = None, None
            learning_rate = self.step_decay(epoch)
            lrate.append(learning_rate)
            np.random.shuffle(XT_batches)
            # batch training
            for X, T in XT_batches:
                outputs = self.forward_prop(X)
                minibatch_cost = self.layers[-1].calc_cost(outputs[-1], T)
                minibatch_costs.append(minibatch_cost)
                w_grads, b_grads = self.back_prop(outputs, T)
                self.update_wb(w_grads, b_grads, learning_rate)
                # self.update_wb(w_grads, b_grads, old_w_grads, old_b_grads)
                # old_w_grads, old_b_grads = w_grads, b_grads

            train_cost, train_accuracy = self.calc_cost_acc(X_train, T_train, x_train_label)
            train_costs.append(train_cost)
            train_acc.append(train_accuracy)

            validation_cost, validation_accuracy = self.calc_cost_acc(X_validation, T_validation, x_val_label)
            val_costs.append(validation_cost)
            val_acc.append(validation_accuracy)

            test_cost, test_accuracy = self.calc_cost_acc(X_test, T_test, x_test_label)
            test_costs.append(test_cost)
            test_acc.append(test_accuracy)

            print('epoch {}: train loss {:.4f} acc {:.4f}, val loss {:.4f} acc {:.4f}, test loss {:.4f} acc {:.4f}'.format(epoch + 1, train_cost, train_accuracy,
                                                                                             validation_cost, validation_accuracy, test_cost, test_accuracy))

            # print('epoch {}: train loss {:.4f} acc {:.4f}'.format(epoch + 1, train_cost, train_accuracy))

            # if len(val_costs) > 3:
            #     # Stop training if the cost on the validation set doesn't decrease
            #     #  for 3 iterations
            #     if val_costs[-1] >= val_costs[-2] >= val_costs[-3]:
            #         break

        return {"nb_of_epochs": epoch + 1,
                "nb_of_batches": nb_of_batches,
                "minibatch_costs": minibatch_costs,
                "train_costs": train_costs,
                "train_acc": train_acc,
                "val_costs": val_costs,
                "val_acc": val_acc,
                "test_costs": test_costs,
                "test_acc":test_acc,
                "lrate": lrate}

