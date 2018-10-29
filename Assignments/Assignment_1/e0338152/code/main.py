import plot
from loaddata import *
from nnet import Nnet

modules = {'100-40-4': [14, 100, 40, 4],
           '28-6-4': [14] + [28]*6 + [4],
           '14-28-4': [14] + [14]*28 + [4]}
question_2_path = 'Question_2'

calc_grad_X = [[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]]
calc_grad_T = [[0,0,0,1]]

def calc_gradient(module_name, layer_sizes):
    W, B = load_WB_data(module_name, layer_sizes)

    net1 = Nnet(layer_sizes, random = False, W=W, B=B)

    outputs = net1.forward_prop(np.asarray(calc_grad_X))
    w_grads, b_grads = net1.back_prop(outputs, np.asarray(calc_grad_T))

    with open(question_2_path + "/dw-" + module_name + ".csv", 'w') as csvfile:
        myWriter = csv.writer(csvfile, lineterminator = '\n')
        for rows in list(w_grads)[::2]:
            myWriter.writerows(rows.tolist())

    with open(question_2_path + "/db-" + module_name + ".csv", 'w') as csvfile:
        myWriter = csv.writer(csvfile, lineterminator = '\n')
        myWriter.writerows(list(b_grads)[::2])


def train_plot_net(module_name):
    layer_sizes = modules[module_name]
    X_train, T_train, X_validation, T_validation, X_test, T_test = load_training_data()

    # W_raw = read_file('weight.csv')
    # B_raw = read_file('bias.csv')
    # W, B = [], []
    #
    # cur = 0
    # for i in range(len(layer_sizes) - 1):
    #     W.append(np.asarray(W_raw[cur:cur + layer_sizes[i]], dtype=np.float32))
    #     B.append(np.asarray(B_raw[i], dtype=np.float32))
    #     cur += layer_sizes[i]
    #
    # net1 = Nnet(layer_sizes, random = False, W=W, B=B)
    net1 = Nnet(layer_sizes)
    net1.set_train_param(batch_size = 40, max_nb_of_epochs = 800, learning_rate = 0.05, momentum = 0.1, lrate_drop = 0.75, epochs_drop = 20)
    res = net1.train(X_train, T_train, X_validation, T_validation, X_test, T_test)
    net1.save_wb()


    plot.plot_accuracys(module_name, res["nb_of_epochs"], res["train_acc"], res["val_acc"], res["test_acc"])
    plot.plot_costs(module_name, res["nb_of_epochs"], res["train_costs"], res["val_costs"], res["test_costs"])
    plot.plot_lrate(module_name, res["nb_of_epochs"], res["lrate"])


# if __name__ == '__main__':

# for key, module in modules.items():
#     calc_gradient(key, module)
train_plot_net('100-40-4')
# train_plot_net('28-6-4') # batch_size = 20, max_nb_of_epochs = 500, learning_rate = 0.01, momentum = 0.1, lrate_drop = 0.9, epochs_drop = 10
# train_plot_net('14-28-4')