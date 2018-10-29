import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import csv
import matplotlib.ticker as mtick

def plot_prediction(x, y, class_name):
    nrow = 10
    ncol = 5

    fig = plt.figure(figsize=(ncol + 1, nrow + 1))
    fig.suptitle("Prediction")
    gs = gridspec.GridSpec(nrow, ncol,
                           wspace=0.0, hspace=0.0,
                           top=1. - 0.5 / (nrow + 1), bottom=0.5 / (nrow + 1),
                           left=0.5 / (ncol + 1), right=1 - 0.5 / (ncol + 1))

    image_mat = np.zeros(10).astype(int)
    for i in range(len(x)):
        cat = np.argmax(y[i])
        if(image_mat[cat] == ncol):
            continue
        ax = plt.subplot(gs[cat, image_mat[cat]])
        # ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0 )

        if ax.is_first_col():
            ax.set_ylabel(class_name[cat], fontsize=9)
        ax.imshow(x[i,:,:,:])
        image_mat[cat]+=1

    plt.show()

def plot_costs(minibatch, epoch, nb_of_epochs = 10):


    minibatch_x = np.linspace(0, nb_of_epochs, num=len(minibatch[0]))
    epochs_x = np.linspace(1, nb_of_epochs, num=nb_of_epochs)
    plt.plot(minibatch_x, np.asarray(minibatch[2], dtype=float), 'C1', linewidth=2, label='Minibatches', alpha=0.5)
    plt.plot(epochs_x, np.asarray(list(list(zip(*epoch[1:]))[2]), dtype=float), 'C2', linewidth=2, label='Train')
    # plt.plot(epochs_x, val_costs, 'b-', linewidth=1.5, label='Validation')
    # plt.plot(epochs_x, test_costs, 'g-', linewidth=1.5, label='Test')


    # best_val = np.min(val_costs);
    # best_epoch = np.argmin(val_costs) + 1;
    plt.xlabel('{} epochs'.format(nb_of_epochs))
    plt.ylabel('Cross Entropy Cost')
    # plt.title(module_name + ' Best Validation Performance is {:.4f} at epoch {}'.format(best_val, best_epoch))
    plt.legend()

    # plt.axhline(y=best_val, color="blue", linestyle="--", linewidth=0.5)
    # plt.axvline(x=best_epoch, color="blue", linestyle="--", linewidth=0.5)
    #
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    plt.axis((0, nb_of_epochs, 0, 2.5))
    # plt.grid()
    plt.show()


def plot_accuracys(minibatch, epoch, nb_of_epochs = 10):
    minibatch_x = np.linspace(0, nb_of_epochs, num=len(minibatch[0]))
    epochs_x = np.linspace(1, nb_of_epochs, num=nb_of_epochs)
    plt.plot(minibatch_x, np.asarray(minibatch[3], dtype=float), 'C1', linewidth=2, label='Minibatches', alpha=0.5)
    plt.plot(epochs_x, np.asarray(list(list(zip(*epoch[1:]))[1]), dtype=float), 'C2', linewidth=2, label='Train')

    # best_val = np.max(val_acc);
    # best_epoch = np.argmax(val_acc) + 1;
    plt.xlabel('{} epochs'.format(nb_of_epochs))
    plt.ylabel('Accuracy')
    # plt.title(module_name + ' Best Validation Accuracy is {:.4f} at epoch {}'.format(best_val, best_epoch))
    plt.legend()

    # plt.axhline(y=best_val, color="blue", linestyle="--", linewidth=0.5)
    # plt.axvline(x=best_epoch, color="blue", linestyle="--", linewidth=0.5)

    plt.axis((0, nb_of_epochs, 0, 1.0))
    plt.grid()
    plt.show()

def plot_compare(histories, opt_labels, title, type="loss"):
    nb_of_epochs = histories[0].params['epochs']
    epochs_x = np.linspace(1, nb_of_epochs, num=nb_of_epochs)
    for i in range(len(histories)):
        optimizer = histories[i].model.optimizer
        plt.plot(epochs_x, histories[i].history[type], 'C'+str(i), linewidth=1.5, label=opt_labels[i])

    plt.xlabel('Epochs')
    plt.title(title)
    if type == 'loss':
        plt.ylabel('Cross Entropy Error')
        plt.axis((1, nb_of_epochs, 0, 2.5))
        plt.legend(loc=1)
    else:
        plt.ylabel('Accuracy')
        plt.axis((1, nb_of_epochs, 0, 1.0))
        plt.legend(loc=4)
    plt.show()

# with open('history.csv') as csvfile:
#     minibatch = list(csv.reader(csvfile))
#     minibatch = [line for line in minibatch if line]
# with open('log.csv') as csvfile:
#     epoch = list(csv.reader(csvfile))
#     epoch = [line for line in epoch if line]
#
# plot_costs(minibatch, epoch)
# plot_accuracys(minibatch, epoch)